import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from src.utils.config import STADIA_API
def visualize_postGIS(table_name, code_column, geom_column=None):
    # 1) connect
    load_dotenv()
    engine = create_engine(os.getenv("DB_DSN"))

    # 2) discover geometry column if not provided
    if geom_column is None:
        meta_sql = """
            SELECT f_geometry_column
            FROM geometry_columns
            WHERE f_table_name = :table
        """
        with engine.connect() as conn:
            result = conn.execute(text(meta_sql), {"table": table_name}).fetchone()
        if result is None:
            raise ValueError(f"No geometry column found for table '{table_name}'")
        geom_column = result[0]

    # 3) read the table without aliasing
    sql = f'SELECT * FROM "{table_name}"'
    gdf = gpd.read_postgis(sql, con=engine, geom_col=geom_column, crs="EPSG:5179")

    # 4) transform to WGS84 for display
    gdf = gdf.to_crs(epsg=4326)

    # 5) plot
    fig, ax = plt.subplots(figsize=(8, 8))
    gdf.plot(
        ax=ax,
        column=code_column,
        categorical=True,
        legend=True,
        legend_kwds={"title": code_column},
    )
    ax.set_title(f"{table_name} – {code_column}")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def visualize_effective_soil_depth():
    visualize_postGIS("effective_soil_depth", "CODE_AD")

def visualize_fracture_density():
    visualize_postGIS("fracture_density", "LEGEND")

def visualize_fault():
    visualize_postGIS("fault", "LEGEND")

def visualize_hydrogeological_unit():
    visualize_postGIS("hydrogeological_unit", "LEGEND")

def visualize_drainage_grade():
    visualize_postGIS("drainage_grade", "CODE_DC")

STADIA_API_KEY = STADIA_API["API_KEY"]
STADIA_ALIDADE_URL = (
    f"https://tiles.stadiamaps.com/tiles/alidade_smooth/{{z}}/{{x}}/{{y}}{{r}}.png"
    f"?api_key={STADIA_API_KEY}"
)
ATTRIBUTION = "© Stadia Maps, © OpenMapTiles, © OpenStreetMap contributors"

def visualize_soil_texture(
    table_name: str = "soil_texture",
    category_column: str = "soil_group",
    crs_source: str = "EPSG:5179",
    crs_plot: str = "EPSG:3857",
    zoom_level: int = 12,
):
    load_dotenv()
    engine = create_engine(os.getenv("DB_DSN"))

    # discover geometry column
    geom_col = engine.connect().execute(text(
        "SELECT f_geometry_column FROM geometry_columns WHERE f_table_name = :t"
    ), {"t": table_name}).scalar()
    if not geom_col:
        raise ValueError(f"No geometry column for '{table_name}'")

    # load & reproject to WebMercator
    gdf = (
        gpd.read_postgis(
            f'SELECT * FROM "{table_name}"',
            con=engine,
            geom_col=geom_col,
            crs=crs_source
        )
        .to_crs(epsg=int(crs_plot.split(":")[1]))
    )

    print(gdf.columns)
    print(gdf[category_column].value_counts())

    if gdf.empty:
        raise ValueError("Loaded GeoDataFrame is empty. Check your table and CRS.")

    # get bounds
    xmin, ymin, xmax, ymax = gdf.total_bounds

    # plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # use your Stadia URL as basemap
    # ctx.add_basemap(
    #     ax,
    #     source=STADIA_ALIDADE_URL,
    #     attribution=ATTRIBUTION,
    #     zoom=zoom_level,
    #     crs=gdf.crs.to_string(),
    #     zorder=1
    # )

    gdf[category_column] = gdf[category_column].astype(str)
    # overlay soil texture with some transparency
    gdf.plot(
        column=category_column,
        categorical=True,
        cmap="tab20",  # a qualitative palette with up to 20 distinct colors
        legend=True,
        legend_kwds={"title": category_column, "loc": "lower left"},
        edgecolor="black",
        linewidth=0.3,
        alpha=0.6,
        ax=ax,
        zorder=2
    )

    ax.set_axis_off()
    ax.set_title(f"{table_name} — {category_column}", fontsize=14)
    plt.tight_layout()
    plt.show()
def visualize_stone_content(
    table_name: str = "stone_content",
    category_column: str = "stone_code",
    crs_source: str = "EPSG:5179",
    crs_plot: str = "EPSG:3857",
    zoom_level: int = 12,
):
    """Visualize stone content on a tiled basemap."""
    load_dotenv()
    engine = create_engine(os.getenv("DB_DSN"))

    # discover geom column
    geom_col = engine.connect().execute(
        text(
            "SELECT f_geometry_column FROM geometry_columns WHERE f_table_name = :t"
        ), {"t": table_name}
    ).scalar()
    if not geom_col:
        raise ValueError(f"No geometry column for '{table_name}'")

    # load & reproject
    gdf = (
        gpd.read_postgis(
            f'SELECT * FROM "{table_name}"',
            con=engine,
            geom_col=geom_col,
            crs=crs_source
        )
        .to_crs(epsg=int(crs_plot.split(":")[1]))
    )

    # 1) WebMercator로 변환
    gdf = gdf.to_crs(epsg=3857)

    # 2) plt 초기화
    fig, ax = plt.subplots(figsize=(10, 10))

    # 3) GeoDataFrame 플롯 (이때 축 범위가 gdf로 설정됨)
    gdf.plot(
        ax=ax,
        column=category_column,
        categorical=True,
        cmap="tab20",
        legend=True,
        legend_kwds={"title": category_column, "loc": "lower left"},
        edgecolor="black",
        linewidth=0.3,
        alpha=0.6,
        zorder=2
    )

    # 4) 축 범위를 지오메트리 범위로 고정 (선택)
    xmin, ymin, xmax, ymax = gdf.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # 5) basemap 추가
    ctx.add_basemap(
        ax,
        source=STADIA_ALIDADE_URL,
        attribution=ATTRIBUTION,
        zoom=zoom_level,
        crs=gdf.crs.to_string(),
        zorder=1
    )

    ax.set_axis_off()
    ax.set_title(f"{table_name} — {category_column}", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # visualize_effective_soil_depth()
    visualize_fracture_density()
    # visualize_fault()
    # visualize_hydrogeological_unit()
    # visualize_drainage_grade()

    # Soil texture
    # visualize_soil_texture()
    # visualize_stone_content()