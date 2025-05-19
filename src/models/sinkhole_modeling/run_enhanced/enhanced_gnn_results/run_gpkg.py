import geopandas as gpd
import googlemaps
from tqdm import tqdm
from dotenv import load_dotenv
from src.utils.config import GOOGLE_CLOUD_PLATFORM_API

# 0) API 키 로드
load_dotenv()
GOOGLE_KEY = GOOGLE_CLOUD_PLATFORM_API["API_KEY"]
if not GOOGLE_KEY:
    raise RuntimeError("Set GOOGLE_MAPS_API_KEY in your environment")
gmaps = googlemaps.Client(key=GOOGLE_KEY)

# 1) GeoDataFrame 로드
file = "/Users/macforhsj/Desktop/SinkholeCivicSentinel/src/models/sinkhole_modeling/run_enhanced/enhanced_gnn_results/hotspot_analysis.gpkg"
gdf = gpd.read_file(file)

# 2) WGS84(위경도) 좌표계로 변환 — Google 역지오코딩은 EPSG:4326 사용
gdf = gdf[gdf['is_super_hotspot'] == 1].copy()
gdf = gdf.to_crs(epsg=4326)


# 3) 대표점(centroid) 기준 lon/lat 계산
gdf['lon'] = gdf.geometry.centroid.x
gdf['lat'] = gdf.geometry.centroid.y

# 4) 역지오코딩 함수 정의 (row 단위)
def reverse_geocode(lat, lon):
    try:
        res = gmaps.reverse_geocode((lat, lon), language='ko')
        if res:
            return res[0]['formatted_address']
    except Exception:
        return None
    return None

# 5) tqdm 으로 진행률 표시하며 ‘address’ 컬럼 채우기
tqdm.pandas()
gdf['address'] = gdf.progress_apply(
    lambda row: reverse_geocode(row.lat, row.lon)
                if row.geometry.geom_type in ('Point','Polygon','MultiPolygon')
                else None,
    axis=1
)

# 6) 결과 확인

gdf[['grid_id', 'risk_score', 'address']].to_csv("hotspot_analysis_full.csv", index=False)
