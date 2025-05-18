import os
import geopandas as gpd
import osmnx as ox
from sqlalchemy import create_engine
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm

load_dotenv()
DB_DSN = os.getenv("DB_DSN")



# 2. Database connection (adjust credentials)
engine = create_engine(DB_DSN)

# 1. Prepare both Seoul masks
seoul = ox.geocode_to_gdf("Seoul, South Korea")
seoul_5174 = seoul.to_crs(epsg=5174).geometry.iloc[0]
seoul_5179 = seoul.to_crs(epsg=5179).geometry.iloc[0]

sources = [
    (
        "/Users/macforhsj/Desktop/SinkholeCivicSentinel/data/raw/자연 요인/단층",
        "W_HG_FAULT_WGS_L.shp",
        "fault",
        5174,
    ),
    (
        "/Users/macforhsj/Desktop/SinkholeCivicSentinel/data/raw/자연 요인/배수등급",
        "ASIT_SOILDRA_AREA.shp",
        "drainage_grade",
        5174,
    ),
    (
        "/Users/macforhsj/Desktop/SinkholeCivicSentinel/data/raw/자연 요인/수문지질단위",
        "W_HG_HYDROGEOLOGICUINT_WGS_P.shp",
        "hydrogeological_unit",
        5174,
    ),
    (
        "/Users/macforhsj/Desktop/SinkholeCivicSentinel/data/raw/자연 요인/지질구조밀도",
        "W_ETC_LINEAMENTDENSITY_WGS_P.shp",
        "fracture_density",
        5174,
    ),
]

effective_soil_depth = ("/Users/macforhsj/Desktop/SinkholeCivicSentinel/data/raw/자연 요인/유효토심","/ASIT_VLDSOILDEP_AREA.shp", "effective_soil_depth", 5174 )

for folder, shp, table, src_epsg in tqdm(sources, total=len(sources)):
    path = os.path.join(folder, shp)
    gdf = gpd.read_file(path)

    # a) If .prj was missing, assign—but for 유효토심 it already has EPSG:5174
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=src_epsg)

    # b) Clip in the *source* CRS
    mask = seoul_5174
    clipped = gpd.clip(gdf, mask)
    print(f"{table}: raw={len(gdf)}, clipped={len(clipped)}")

    if clipped.empty:
        print(f"‼️  {table} empty after clip—check src_epsg={src_epsg}")
        continue

    # c) Reproject the clipped result into your *storage* CRS (5179)
    clipped_5179 = clipped.to_crs(epsg=5179)

    # d) Push into PostGIS
    geom_type = clipped_5179.geom_type.unique()[0].upper()
    clipped_5179.to_postgis(
        name=table,
        con=engine,
        if_exists="replace",
        index=False,
        dtype={"geometry": f"Geometry({geom_type},5179)"}
    )
    print(f"✅ {table}: loaded {len(clipped_5179)} rows into PostGIS (EPSG:5179).")
