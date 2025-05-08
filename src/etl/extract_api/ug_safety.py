"""underground_safety_api.py
Utility layer for MOLIT Underground Safety Info API (지하안전정보).

Design principles
-----------------
1. **Single core fetcher** – handles HTTP, pagination, JSON→GeoDataFrame.
2. **Operation registry** – maps *internal* operation keys → endpoint paths.
3. **Parameter spec validation** – each operation declares `required` / `optional` keys.
4. **Thin wrapper helpers** – Python‑friendly functions per frequently‑used operation.
5. **Region helper** – support `region` argument to auto‑resolve `emdCd` via GIS lookup.

This keeps low‑level logic DRY while exposing an ergonomic interface.
"""
from __future__ import annotations

import certifi
import time
import pyproj
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from src.utils.region import load_emdcd_from_region

# ---------------------------------------------------------------------------
# Operation registry & parameter specifications
# ---------------------------------------------------------------------------

OPERATIONS = {
    "impact_list": "getImpatEvalutionList",
    "impact_info": "getImpatEvalutionInfo",
    "small_impact_list": "getSmallImpactEvalutionList",
    "small_impact_info": "getSmallImpactEvalutionInfo",

    "utility_list": "getUndergroundUtilityList",
    "utility_info": "getUndergroundUtilityInfo",

    "subsidence_eval_list": "getSubsidenceEvalutionList",
    "subsidence_action": "getSubsidenceResult",
    "subsidence_emergency": "getSubsidenceExpediency",
    "accident_list": "getSubsidenceList",
    "accident_info": "getSubsidenceInfo",
}


# 공통 에러코드 매핑
ERROR_MAP = {
    "0":  ("NORMAL_CODE", "정상"),
    "1":  ("APPLICATION_ERROR", "어플리케이션 에러"),
    "2":  ("DB_ERROR", "데이터베이스 에러"),
    "3":  ("NODATA_ERROR", "데이터없음 에러"),
    "4":  ("HTTP_ERROR", "HTTP 에러"),
    "5":  ("SERVICETIMEOUT_ERROR", "서비스 연결실패 에러"),
    "10": ("INVALID_REQUEST_PARAMETER_ERROR", "잘못된 요청 파라메터 에러"),
    "11": ("NO_MANDATORY_REQUEST_PARAMETERS_ERROR", "필수요청 파라메터가 없음"),
    "12": ("NO_OPENAPI_SERVICE_ERROR", "해당 오픈API서비스가 없거나 폐기됨"),
    "20": ("SERVICE_ACCESS_DENIED_ERROR", "서비스 접근거부"),
    "21": ("TEMPORARILY_DISABLE_THE_SERVICEKEY_ERROR", "일시적으로 사용할 수 없는 서비스 키"),
    "22": ("LIMITED_NUMBER_OF_SERVICE_REQUESTS_EXCEEDS_ERROR", "서비스 요청제한횟수 초과에러"),
    "30": ("SERVICE_KEY_IS_NOT_REGISTERED_ERROR", "등록되지 않은 서비스키"),
    "31": ("DEADLINE_HAS_EXPIRED_ERROR", "기한만료된 서비스키"),
    "32": ("UNREGISTERED_IP_ERROR", "등록되지 않은 IP"),
    "33": ("UNSIGNED_CALL_ERROR", "서명되지 않은 호출"),
    "99": ("UNKNOWN_ERROR", "기타에러"),
}


PARAM_SPECS = {
    "impact_list": {
        "required": ["sysRegDateFrom", "sysRegDateTo"],
        "optional": [
            "geomLon", "geomLat", "buffer", "numOfRows", "pageNo", "emdCd"
        ],
    },
    "impact_info": {
        "required": ["evalNo"],
        "optional": ["numOfRows", "pageNo"],
    },
    "small_impact_list": {
        "required": ["sysRegDateFrom", "sysRegDateTo"],
        "optional": [
            "geomLon", "geomLat", "buffer", "numOfRows", "pageNo", "emdCd"
        ],
    },
    "small_impact_info": {
        "required": ["evalNo"],
        "optional": ["numOfRows", "pageNo"],
    },


    "utility_list": {
        "required": ["startYmd", "endYmd"],
        "optional": ["numOfRows", "pageNo"],
    },
    "utility_info": {
        "required": ["facilNo"],
        "optional": ["numOfRows", "pageNo"],
    },




    "subsidence_eval_list": {
        "required": ["startYmd", "endYmd"],
        "optional": ["numOfRows", "pageNo", "emdCd"],
    },
    "subsidence_action": {
        "required": ["evalNo"],
        "optional": ["numOfRows", "pageNo"],
    },
    "subsidence_emergency": {
        "required": ["evalNo"],
        "optional": ["numOfRows", "pageNo"],
    },
    "accident_list": {
        "required": ["sagoDateFrom", "sagoDateTo"],
        "optional": [
            "geomLon", "geomLat", "buffer", "numOfRows", "pageNo", "emdCd"
        ],
    },
    "accident_info": {
        "required": ["sagoNo"],
        "optional": ["numOfRows", "pageNo"],
    },
}

# ---------------------------------------------------------------------------
# Core helpers (validate/json_to_gdf unchanged)
# ---------------------------------------------------------------------------

def _validate_params(operation: str, params: dict) -> None:
    spec = PARAM_SPECS.get(operation)
    if spec is None:
        raise ValueError(f"Unknown operation: {operation}")
    missing = [k for k in spec["required"] if k not in params or params[k] is None]
    if missing:
        raise ValueError(f"Missing required parameters for '{operation}': {', '.join(missing)}")
    unexpected = [k for k in params if k not in spec["required"] + spec["optional"]]
    if unexpected:
        raise ValueError(f"Unexpected parameter(s) for '{operation}': {', '.join(unexpected)}")


def _json_to_gdf(features: list[dict], crs: str):
    if not features:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    if features[0].get("geometry"):
        rows = []
        for feat in features:
            geom = shape(feat["geometry"])
            props = feat.get("properties", {}).copy()
            props["geometry"] = geom
            rows.append(props)
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    return pd.DataFrame([feat.get("properties", feat) for feat in features])

def _items_to_gdf(items: list[dict], crs: str = "EPSG:4326"):
    """
    Convert the API’s 'item' dicts into a GeoDataFrame.
    If items have 'sagoLat'/'sagoLon' or 'geomLat'/'geomLon', use those;
    otherwise return an empty GeoDataFrame or plain DataFrame.
    """
    if not items:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    # Detect geographic fields
    lat_key = next((k for k in ("geomLat", "sagoLat") if k in items[0]), None)
    lon_key = next((k for k in ("geomLon", "sagoLon") if k in items[0]), None)

    df = pd.DataFrame(items)
    if lat_key and lon_key:
        # Build Point geometries
        df["geometry"] = gpd.points_from_xy(df[lon_key], df[lat_key])
        return gpd.GeoDataFrame(df, geometry="geometry", crs=crs)

    # No coords → return plain DataFrame
    return df
# ---------------------------------------------------------------------------
# Public fetcher
# ---------------------------------------------------------------------------


MAX_BYTES = 1024
MIN_INTERVAL = 0.05


def fetch_underground_safety_data(
    api_key: str,
    operation: str,
    page_size: int = 10,
    crs: str = "EPSG:4326",
    **params,
):
    _validate_params(operation, params)
    endpoint = OPERATIONS[operation]
    base_url = f"http://apis.data.go.kr/1611000/undergroundsafetyinfo/{endpoint}?serviceKey={api_key}"
    query = {"type":"json", **params}
    session = requests.Session()

    pages: list[gpd.GeoDataFrame | pd.DataFrame] = []
    last_call = 0.0

    while True:
        elapsed = time.time() - last_call
        if elapsed < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - elapsed)

        r = session.get(base_url, params=query, timeout=(10,60), verify=False)
        last_call = time.time()

        if r.text.strip().startswith("<?xml") and "<soapenv:Fault>" in r.text:
            fault_string = ET.fromstring(r.text).findtext(".//faultstring")
            raise RuntimeError(f"SOAP Fault: {fault_string}")

        try:
            payload = r.json()
        except ValueError:
            raise RuntimeError(f"JSON 파싱 실패, 응답 본문:\n{r.text}")
        items = (payload.get("response", {})
                        .get("body", {})
                        .get("items", {})
                 )
        if not items:
            break

        page_df = _items_to_gdf(items, crs)
        pages.append(page_df)

        # pagination
        if len(items) < page_size:
            break
        query["pageNo"] = int(query.get("pageNo", 1)) + 1

    # concat all pages (or return empty GeoDataFrame if none)
    if not pages:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    combined = pd.concat(pages, ignore_index=True)
    # if any page was GeoDataFrame, result is GeoDataFrame
    if isinstance(pages[0], gpd.GeoDataFrame):
        return gpd.GeoDataFrame(combined, geometry="geometry", crs=crs)
    return combined

# Thin wrapper helpers (typed) (typed)
# ---------------------------------------------------------------------------

def get_impact_evaluation_list(api_key: str, **params):
    """Wrapper for *지하안전평가 리스트 조회* (MOLITJIS‑01). """
    return fetch_underground_safety_data(api_key, "impact_list", **params)

def get_impact_evaluation_info(api_key: str, **params):
    """Wrapper for *지하안전평가 정보 조회* (MOLITJIS‑02)."""
    return fetch_underground_safety_data(api_key, "impact_info", **params)


def get_small_impact_evaluation_list(api_key: str, **params):
    """Wrapper for *소규모 영향평가 리스트 조회*."""
    return fetch_underground_safety_data(api_key, "small_impact_list", **params)


def get_small_impact_evaluation_info(api_key: str, **params):
    """Wrapper for *소규모 영향평가 정보 조회*."""
    return fetch_underground_safety_data(api_key, "small_impact_info", **params)


def get_subsidence_evaluation_list(api_key: str, **params):
    """Wrapper for *지반침하위험도평가 리스트 조회* (MOLITJIS‑10)."""
    return fetch_underground_safety_data(api_key, "subsidence_eval_list", **params)


def get_subsidence_result(api_key: str, **params):
    """Wrapper for *지반침하위험도평가 안전조치내용* (MOLITJIS‑11)."""
    return fetch_underground_safety_data(api_key, "subsidence_action", **params)


def get_subsidence_expediency(api_key: str, **params):
    """Wrapper for *지반침하위험도평가 응급조치내용* (MOLITJIS‑12)."""
    return fetch_underground_safety_data(api_key, "subsidence_emergency", **params)


def get_subsidence_accident_list(api_key: str, **params):
    """Wrapper for *지반침하사고 리스트 조회* (MOLITJIS‑15)."""
    return fetch_underground_safety_data(api_key, "accident_list", **params)


def get_subsidence_accident_info(api_key: str, **params):
    """Wrapper for *지반침하사고 정보 조회* (MOLITJIS‑16)."""
    return fetch_underground_safety_data(api_key, "accident_info", **params)

def get_underground_utility_list(api_key: str, **params):
    """
    Wrapper for *안전점검대상 지하시설물 리스트 조회* (MOLITJIS-07: getUndergroundUtilityList).
    Required params: startYmd, endYmd
    Optional params: numOfRows, pageNo
    """
    return fetch_underground_safety_data(api_key, "utility_list", **params)


def get_underground_utility_info(api_key: str, **params):
    """
    Wrapper for *안전점검대상 지하시설물 정보 조회* (MOLITJIS-08: getUndergroundUtilityInfo).
    Required param: facilNo
    Optional params: numOfRows, pageNo
    """
    return fetch_underground_safety_data(api_key, "utility_info", **params)


# ---------------------------------------------------------------------------
# End of module
# ---------------------------------------------------------------------------
