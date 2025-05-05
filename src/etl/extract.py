from src.utils.visualization import profile_df

def load_faults() -> gpd.GeoDataFrame:
    gdf = gpd.read_file('data/raw/static/lt_l_gimsfault.shp')
    profile_df(gdf, "Faults (Shapefile)")
    return gdf

def load_soil_texture(api_key, emdCd=None, bbox=None):
    gdf = fetch_soil_texture(api_key, emdCd=emdCd, bbox=bbox, crs="EPSG:5181")
    profile_df(gdf, "Soil Texture (API)")
    return gdf

def load_inspections_table(engine) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM inspections_raw", con=engine)
    profile_df(df, "Inspections (Table)")
    return df