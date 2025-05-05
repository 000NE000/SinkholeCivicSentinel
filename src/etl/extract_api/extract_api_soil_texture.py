#토양도
import requests
import geopandas as gpd
from shapely.geometry import shape
from src.utils.config import API_CONFIG
from ydata_profiling import ProfileReport
import os

class VWorldAPIError(Exception):
    """Raised when the vworld API returns an error status."""
    pass

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['AppleGothic', 'sans-serif']


# VWorld가 지원하는 EPSG 코드 목록
SUPPORTED_CRS = {
    "4326": "WGS84",
    "5179": "Korean_2000_TM"   # 중앙원점 38° … 토성도에서 지원
}


def normalize_crs(crs: str) -> str:
    code = crs.split(":")[-1]
    return code if code in SUPPORTED_CRS else "4326"


def fetch_soil_texture(
    api_key: str,
    bbox: tuple = None,
    emdCd: str = None,
    page_size: int = 1000,
    crs: str = "EPSG:5186",
) -> gpd.GeoDataFrame:
    base_url = "https://api.vworld.kr/req/data"
    crs_code = normalize_crs(crs)
    params = {
        "service": "data",
        "version": "2.0",
        "request": "GetFeature",
        "key": api_key,
        "format": "json",
        "data": "LT_C_GIMSSCS",
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

    all_records = []
    while True:
        resp = requests.get(base_url, params=params)
        raw = resp.json()
        # 1) "response" 래퍼가 있으면 꺼내고, 없으면 원본(raw) 그대로 사용
        resp_data = raw.get("response", raw)

        status = resp_data.get("status")
        if status == "NOT_FOUND":
            # 빈 결과를 정상 처리
            break
        elif status != "OK":
            err = resp_data.get("error", {})
            raise VWorldAPIError(
                f"API error {err.get('code', '<no code>')} "
                f"(level {err.get('level', '<no level>')}): "
                f"{err.get('text', '<no error text>')}"
            )

        # 2) result 키가 없으면 빈 dict, features 없으면 빈 리스트
        result = resp_data.get("result", {})
        features = result.get("featureCollection").get("features", [])
        print(f"[DEBUG] Page {params['page']} returned {len(features)} features")
        for feat in features:
            print(f"[DEBUG] Processing feature id: {feat.get('id')}")
            props = feat.get("properties", {})
            geom = shape(feat.get("geometry", {}))
            all_records.append({
                "emd_cd": props.get("emdCd"),
                "soil_group": props.get("legend"),
                "geometry": geom
            })

        # page_info = raw.get("page", {})
        if resp_data.get("page", {}).get("current", 0) >= resp_data["page"].get("total", 0):
            break
        params["page"] += 1

    if not all_records:  # nothing collected in any page
        return gpd.GeoDataFrame(
            columns=["emd_cd", "soil_group", "geometry"],
            geometry="geometry",
            crs=f"EPSG:{crs_code}"
        )

    gdf = gpd.GeoDataFrame(all_records, geometry="geometry", crs=f"EPSG:{crs_code}")
    if crs_code != "5186":
        gdf = gdf.to_crs("EPSG:5186")
    #
    # prof_df = gdf.drop(columns="geometry")
    # profile = ProfileReport(
    #     prof_df,
    #     title=f"SoilTexture Profile ({emdCd or 'bbox'})",
    #     minimal=True
    # )
    # # 디렉터리 없으면 생성
    # os.makedirs("../../../docs/profiles", exist_ok=True)
    # # 파일명: soil_texture_emdCd.html 또는 soil_texture_bbox.html
    # name_tag = emdCd if emdCd else "bbox"
    # profile.to_file(f"../../../docs/profiles/soil_texture_{name_tag}.html")
    return gdf

# Example to see the real error message:
if __name__ == "__main__":
    bbox_wgs84 = (126.5, 37.2, 127.5, 37.8)
    API_KEY = API_CONFIG["VWORLD_API_KEY"]
    try:
        df = fetch_soil_texture(API_KEY, bbox=bbox_wgs84, crs="EPSG:4326")
        big_bbox = (126.4, 37.0, 127.6, 38.0)
        print("Rows:", len(df))

        print(df)  # prints all rows (or a truncated view)
        print(df.shape)  # shows (n_rows, n_columns)
        df4326 = df.to_crs("EPSG:4326")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))
        df4326.plot(
            column='soil_group',  # uses the 'soil_group' column for categories
            categorical=True,  # treat values as discrete categories
            legend=True,  # show a legend
            ax=ax
        )
        ax.set_title("Soil Group Distribution")
        ax.set_axis_off()  # hide axes for a cleaner map
        ax.set_aspect('equal')  # ensure correct map proportions
        plt.show()
    except VWorldAPIError as e:
        print("Caught VWorldAPIError:", e)
    except Exception as e:
        print(f"[WARN] Profiling failed: {e}")