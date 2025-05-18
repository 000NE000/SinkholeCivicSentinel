import os
import webbrowser
import geopandas as gpd
import folium
from sqlalchemy import create_engine
from dotenv import load_dotenv
# 1) DB 연결
load_dotenv()
db_url = os.getenv("DB_DSN")
engine = create_engine(db_url)

# 2) grid_id로 geom을 조인한 쿼리
sql = """
SELECT 
  b.grid_id,
  b.year,
  b.area,
  g.geom
FROM bld_in_grid AS b
JOIN grid_100m AS g
  ON b.grid_id = g.grid_id
"""

# 3) GeoDataFrame으로 불러오기 (EPSG:5179)
gdf = gpd.read_postgis(
    sql,
    con=engine,
    geom_col="geom",
    crs="EPSG:5179"
)

# 4) Folium용 좌표계(WGS84)로 변환
gdf = gdf.to_crs("EPSG:4326")

# 5) Folium 지도 생성
m = folium.Map(
    location=[37.5665, 126.9780],
    zoom_start=11,
    tiles="CartoDB.Positron"
)

# 6) GeoJson 레이어 추가
folium.GeoJson(
    gdf,
    name="bld_in_grid",
    tooltip=folium.GeoJsonTooltip(
        fields=["grid_id", "year", "area"],
        aliases=["Grid ID", "Year", "Area"],
        localize=True
    )
).add_to(m)

# 7) HTML로 저장 & 브라우저 열기
output_path = "bld_in_grid_map.html"
m.save(output_path)
webbrowser.open(f"file://{os.path.abspath(output_path)}")