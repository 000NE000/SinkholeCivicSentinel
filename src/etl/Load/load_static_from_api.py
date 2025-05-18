# Fetch and upload VWorld soil texture and stone content to PostGIS (with grid-based bbox splitting)

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import geopandas as gpd
import osmnx as ox
import pandas as pd
import numpy as np
from shapely.geometry import box
from src.utils.config import API_CONFIG
# Import your fetch functions
from src.etl.extract_api.extract_api_soil_texture import fetch_soil_texture
from src.etl.extract_api.extract_api_stone_content import fetch_stone_content
from shapely.validation import make_valid, explain_validity


def clean_gdf(gdf):
    # 1) Drop null geometries
    gdf = gdf.dropna(subset=["geometry"]).copy()
    # 2) Repair issues and buffer zero-width
    gdf["geometry"] = gdf.geometry.apply(lambda geom: make_valid(geom)).buffer(0)
    # 3) Remove invalid
    invalid = gdf[~gdf.is_valid]
    if not invalid.empty:
        print(f"Still {len(invalid)} invalid geometries (dropping):")
        for idx, row in invalid.iterrows():
            print(f" • index={idx}, reason={explain_validity(row.geometry)}")
        gdf = gdf[gdf.is_valid]
    return gdf


def split_bbox(bbox, step_deg=0.1):
    """
    Split bbox into a grid of shapely Polygons (boxes).
    Smaller step_deg ensures finer coverage.
    """
    minx, miny, maxx, maxy = bbox
    xs = np.arange(minx, maxx + step_deg, step_deg)
    ys = np.arange(miny, maxy + step_deg, step_deg)
    cells = []
    for x0 in xs[:-1]:
        for y0 in ys[:-1]:
            cells.append(box(x0, y0, x0 + step_deg, y0 + step_deg))
    return cells


def main():
    load_dotenv()
    engine = create_engine(os.getenv("DB_DSN"), echo=False)
    API_KEY = API_CONFIG["VWORLD_API_KEY"]

    # Overall bounding box (lon_min, lat_min, lon_max, lat_max)
    bbox = (126.5, 37.2, 127.5, 37.8)
    # Create finer grid (0.1° steps)
    grid_polys = split_bbox(bbox, step_deg=0.1)

    # Load Seoul boundary and project to EPSG:5179
    seoul_gdf = ox.geocode_to_gdf("Seoul, South Korea").to_crs(epsg=5179)
    seoul_poly = seoul_gdf.geometry.union_all().buffer(0)

    soil_parts = []
    # Iterate tiles, fetch and clean
    for idx, poly in enumerate(grid_polys):
        bounds = poly.bounds
        try:
            tile = fetch_soil_texture(API_KEY, bbox=bounds, crs="EPSG:4326")
            print("raw tile count:", len(tile))
            tile = tile.to_crs(epsg=5179)
            cleaned = clean_gdf(tile)
            if not cleaned.empty:
                soil_parts.append(cleaned)
                print(f"Tile {idx}: {len(cleaned)} geometries added.")
        except Exception as e:
            print(f"Error fetching soil for tile {idx}, bounds={bounds}: {e}")

    # Concatenate and final clean
    soil_all = gpd.GeoDataFrame(pd.concat(soil_parts, ignore_index=True), crs="EPSG:5179")
    soil_all = clean_gdf(soil_all)

    # Clip to Seoul polygon
    soil_seoul = gpd.clip(soil_all, seoul_poly)
    print(f"Total soil features after clip: {len(soil_seoul)}")

    # Stone content
    stone = fetch_stone_content(API_KEY, bbox=bbox, crs="EPSG:4326").to_crs(epsg=5179)
    stone_clean = clean_gdf(stone)
    stone_seoul = gpd.clip(stone_clean, seoul_poly)
    print(f"Total stone features after clip: {len(stone_seoul)}")

    # Upload to PostGIS
    soil_seoul.to_postgis("soil_texture", engine, if_exists="replace", index=False)
    stone_seoul.to_postgis("stone_content", engine, if_exists="replace", index=False)
    print("Upload complete.")


if __name__ == "__main__":
    main()
