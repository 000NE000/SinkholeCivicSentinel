import pandas as pd
from IPython.display import display

def profile_df(df: pd.DataFrame, name: str):
    """
    Prints basic shape, dtypes, null counts, describe() for any DataFrame or GeoDataFrame.
    """
    print(f"\n=== PROFILE: {name} ===")
    print("Shape:", df.shape)
    print("Dtypes:\n", df.dtypes, "\n")
    print("Null counts:\n", df.isna().sum(), "\n")
    try:
        display(df.describe(include='all').T)
    except Exception:
        pass
    # For geodataframe, show CRS and bounds
    if hasattr(df, 'crs'):
        print("CRS:", df.crs)
        print("Bounds:", df.total_bounds)  # [minx, miny, maxx, maxy]
    print("="*30)