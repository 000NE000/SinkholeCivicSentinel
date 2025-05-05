#자갈함량
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from src.utils.config import API_CONFIG
from ydata_profiling import ProfileReport
import os
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['AppleGothic', 'sans-serif']

class VWorldAPIError(Exception):
    """Raised when the vworld API returns an error status."""
    pass

# VWorld가 지원하는 EPSG 코드 목록
SUPPORTED_CRS = {
    "4326": "WGS84",
    "5179": "Korean_2000_TM"
}

def normalize_crs(crs: str) -> str:
    code = crs.split(":")[-1]
    return code if code in SUPPORTED_CRS else "4326"

def fetch_stone_content(
    api_key: str,
    bbox: tuple = None,
    emdCd: str = None,
    page_size: int = 1000,
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    base_url = "https://api.vworld.kr/req/data"
    crs_code = normalize_crs(crs)
    params = {
        "service": "data",
        "version": "2.0",
        "request": "GetFeature",
        "key": api_key,
        "format": "json",
        "data": "LT_C_ASITSURSTON",
        "size": page_size,
        "page": 1,
        "geometry": "true",
        "attribute": "true",
        "crs": f"EPSG:{crs_code}",
    }

    if emdCd:
        params["attrFilter"] = f"emdCd:=:{emdCd}"
    elif bbox:
        minx, miny, maxx, maxy = bbox
        params["geomFilter"] = f"BOX({minx},{miny},{maxx},{maxy})"
    else:
        raise ValueError("Either bbox or emdCd must be provided.")

    records = []
    while True:
        resp = requests.get(base_url, params=params)
        data = resp.json().get("response", resp.json())

        status = data.get("status")
        if status == "NOT_FOUND":
            break
        if status != "OK":
            err = data.get("error", {})
            raise VWorldAPIError(
                f"API error {err.get('code', '<no code>')}: {err.get('text', '<no text>')}"
            )

        features = data["result"]["featureCollection"]["features"]
        for feat in features:
            props = feat["properties"]
            geom = shape(feat["geometry"])
            records.append({
                "emd_cd": props.get("emdCd"),
                "stone_code": props.get("code_sg"),
                "stone_label": props.get("label"),
                "geometry": geom
            })

        page = data["page"]["current"]
        total = data["page"]["total"]
        if page >= total:
            break
        params["page"] += 1

    # GeoDataFrame 생성 및 CRS 변환
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=f"EPSG:{crs_code}")
    if crs_code != "5186":
        gdf = gdf.to_crs("EPSG:5186")

    # # 프로파일링: geometry 제외하고 데이터프레임 생성
    # prof_df = gdf.drop(columns="geometry")
    # profile = ProfileReport(
    #     prof_df,
    #     title=f"Stone Content Profile ({emdCd or 'bbox'})",
    #     minimal=True
    # )
    #
    # # 프로필 저장 디렉터리 및 파일 생성
    # os.makedirs("../../../docs/profiles", exist_ok=True)
    # name_tag = emdCd if emdCd else "bbox"
    # profile_path = f"../../../docs/profiles/stone_content_{name_tag}.html"
    # profile.to_file(profile_path)
    # print(f"[INFO] Profile report saved to {profile_path}")

    return gdf

if __name__ == "__main__":
    bbox_wgs84 = (126.5, 37.2, 127.5, 37.8)
    API_KEY = API_CONFIG["VWORLD_API_KEY"]
    try:
        df = fetch_stone_content(API_KEY, bbox=bbox_wgs84, crs="EPSG:4326")
        print("Rows:", len(df))
        print(df.shape)

        # plotting 예시
        df4326 = df.to_crs("EPSG:4326")
        fig, ax = plt.subplots(figsize=(10, 10))
        df4326.plot(
            column='stone_code',
            categorical=True,
            legend=True,
            ax=ax
        )
        ax.set_title("Stone Content Distribution (자갈 함량 분포)")
        ax.set_axis_off()
        plt.show()
    except VWorldAPIError as e:
        print("Caught VWorldAPIError:", e)
    except Exception as e:
        print(f"[WARN] {e}")