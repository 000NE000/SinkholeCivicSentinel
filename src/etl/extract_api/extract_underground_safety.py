"""underground_safety_api.py
Utility layer for MOLIT Underground Safety Info API (지하안전정보). 

Design principles
-----------------
1. **Single core fetcher** – handles HTTP, pagination, JSON→GeoDataFrame.
2. **Operation registry** – maps *internal* operation keys → endpoint paths.
3. **Parameter spec validation** – each operation declares `required` / `optional` keys.
4. **Thin wrapper helpers** – Python‑friendly functions per frequently‑used operation.

This keeps low‑level logic DRY while exposing an ergonomic interface.
"""
from __future__ import annotations

import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from src.utils.config import API_CONFIG

# ---------------------------------------------------------------------------
# Operation registry & parameter specifications
# ---------------------------------------------------------------------------

OPERATIONS = {
    # Evaluation (영향평가)
    "impact_list": "getImpatEvalutionList",          # MOLITJIS‑01
    "impact_info": "getImpatEvalutionInfo",          # MOLITJIS‑02
    # Small Evaluation (소규모 영향평가)
    "small_impact_list": "getSmallImpactEvaluationList",  # "소규모지하안전평가 리스트"
    "small_impact_info": "getSmallImpactEvaluationInfo",
    # Subsidence risk evaluation (지반침하위험도평가)
    "subsidence_eval_list": "getSubsidenceEvalutionList",   # MOLITJIS‑10
    "subsidence_action": "getSubsidenceResult",            # MOLITJIS‑11
    "subsidence_emergency": "getSubsidenceExpediency",      # MOLITJIS‑12
    # Subsidence accidents (지반침하사고)
    "accident_list": "getSubsidenceList",           # MOLITJIS‑15
    "accident_info": "getSubsidenceInfo",           # MOLITJIS‑16
}

PARAM_SPECS = {
    "impact_list": {
        "required": ["sysRegDateFrom", "sysRegDateTo"],
        "optional": [
            "geomLon", "geomLat", "buffer", "numOfRows", "pageNo",
        ],
    },
    "impact_info": {
        "required": ["evalNo"],
        "optional": ["numOfRows", "pageNo"],
    },
    "small_impact_list": {
        "required": ["sysRegDateFrom", "sysRegDateTo"],
        "optional": [
            "geomLon", "geomLat", "buffer", "numOfRows", "pageNo",
        ],
    },
    "small_impact_info": {
        "required": ["evalNo"],
        "optional": ["numOfRows", "pageNo"],
    },
    "subsidence_eval_list": {
        "required": ["startYmd", "endYmd"],
        "optional": ["numOfRows", "pageNo"],
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
            "geomLon", "geomLat", "buffer", "numOfRows", "pageNo",
        ],
    },
    "accident_info": {
        "required": ["sagoNo"],
        "optional": ["numOfRows", "pageNo"],
    },
}

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _validate_params(operation: str, params: dict) -> None:
    """Raise ``ValueError`` if required parameters are missing."""
    spec = PARAM_SPECS.get(operation)
    if spec is None:
        raise ValueError(f"Unknown operation: {operation}")

    missing = [key for key in spec["required"] if key not in params or params[key] is None]
    if missing:
        raise ValueError(
            f"Missing required parameters for '{operation}': {', '.join(missing)}"
        )

    # Warn for unexpected params (helps catch typos)
    unexpected = [k for k in params if k not in spec["required"] + spec["optional"]]
    if unexpected:
        raise ValueError(
            f"Unexpected parameter(s) for '{operation}': {', '.join(unexpected)}"
        )


def _json_to_gdf(features: list[dict], crs: str) -> gpd.GeoDataFrame:
    """Convert GeoJSON‑like feature list to GeoDataFrame or DataFrame."""
    if not features:
        return gpd.GeoDataFrame(crs=crs)

    # Some endpoints include geometry, others don't.
    if features[0].get("geometry"):
        rows = []
        for feat in features:
            geom = shape(feat["geometry"])
            props = feat["properties"].copy()
            props["geometry"] = geom
            rows.append(props)
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    else:
        # Flatten properties + (optionally) geometry bbox
        rows = [feat.get("properties", feat) for feat in features]
        return pd.DataFrame(rows)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public: universal fetcher
# ---------------------------------------------------------------------------

def fetch_underground_safety_data(
    api_key: str,
    operation: str,
    page_size: int = 1000,
    crs: str = "EPSG:4326",
    **params,
):
    """Generic downloader for all Underground Safety Info endpoints.

    Parameters
    ----------
    api_key : str
        MOLIT OpenAPI key.
    operation : str
        Internal operation key (see ``OPERATIONS``).
    page_size : int, default 1000
        Items per page. Some endpoints cap at 1000.
    crs : str, default "EPSG:4326"
        Target CRS for geometry; ignored if endpoint has no geometries.
    **params
        Query parameters specific to the operation.

    Returns
    -------
    pandas.DataFrame | geopandas.GeoDataFrame
    """
    _validate_params(operation, params)

    endpoint = OPERATIONS[operation]
    base_url = "https://apis.data.go.kr/1611000/undergroundsafetyinfo"

    query = {
        "service": "data",
        "version": "2.0",
        "request": "GetFeature",
        "key": api_key,
        "format": "json",
        "size": page_size,
        "page": 1,
        # geometry / attribute flags make no harm even if geometry absent.
        "geometry": "true",
        "attribute": "true",
        **params,
    }

    all_features: list[dict] = []
    while True:
        resp = requests.get(f"{base_url}/{endpoint}.json", params=query, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        feats = payload.get("features", [])
        if not feats:
            break
        all_features.extend(feats)

        # Pagination termination condition
        if len(feats) < page_size:
            break
        query["page"] += 1

    return _json_to_gdf(all_features, crs)


# ---------------------------------------------------------------------------
# Thin wrapper helpers (typed)
# ---------------------------------------------------------------------------

def get_impact_evaluation_list(api_key: str, **params):
    """Wrapper for *지하안전평가 리스트 조회* (MOLITJIS‑01)."""
    return fetch_underground_safety_data(api_key, "impact_list", **params)


def get_impact_evaluation_info(api_key: str, **params):
    """Wrapper for *지하안전평가 정보 조회* (MOLITJIS‑02)."""
    return fetch_underground_safety_data(api_key, "impact_info", **params)

def get_small_impact_evaluation_list(api_key: str, **params):
    """Wrapper for *지하안전평가 리스트 조회* (MOLITJIS‑01)."""
    return fetch_underground_safety_data(api_key, "impact_list", **params)


def get_small_impact_evaluation_info(api_key: str, **params):
    """Wrapper for *지하안전평가 정보 조회* (MOLITJIS‑02)."""
    return fetch_underground_safety_data(api_key, "impact_info", **params)


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

# ---------------------------------------------------------------------------
# End of module
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    api_key = API_CONFIG["DATAGOKR_ENCODING_KEY"]
    get_impact_evaluation_list(api_key, sysRegDateFrom = 20250101 , sysRegDateTo = 20250403)
