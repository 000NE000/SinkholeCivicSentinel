import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.family'] = ['AppleGothic', 'sans-serif']
# gdf = gpd.read_file(
#     "../../../data/raw/static/수문지질단위/W_HG_HYDROGEOLOGICUINT_WGS_P.shp",
#     encoding='cp949'  # 또는 encoding='euc-kr'
# )

# gdf = gpd.read_file("../../../data/raw/static/유효토심/ASIT_VLDSOILDEP_AREA.shp")
gdf = gpd.read_file("../../../data/raw/static/지질구조밀도/W_ETC_LINEAMENTDENSITY_WGS_P.shp")
# gdf = gpd.read_file("../../../data/raw/static/단층/W_HG_FAULT_WGS_L.shp")
# gdf = gpd.read_file("../../../data/raw/static/배수등급/ASIT_SOILDRA_AREA.shp")

# gdf = gdf.set_crs("EPSG:5186")
print(gdf.columns)
print(gdf.dtypes)
print(gdf.shape)
print(gdf.head())
print("CRS:", gdf.crs)

# print(gdf[''].unique())

fig, ax = plt.subplots(figsize=(12, 12))
gdf.plot(
    column='LEGEND',
    cmap='tab20',             # 컬러맵 (20종 범주형 색)
    linewidth=1.5,            # 선 굵기
    legend=True,              # 범례 표시
    ax=ax
)
ax.set_title("칼럼 파악", fontsize=16)
ax.set_axis_off()             # 축 제거
plt.tight_layout()
plt.show()



# mapping_df = gdf[['LEGEND', 'INFO']].drop_duplicates()
# # None 또는 NaN 제거 + 정렬
# valid_legend_codes = sorted([code for code in gdf['LEGEND'].unique() if pd.notnull(code)])
#
# # 매핑 출력
# for code in valid_legend_codes:
#     infos = mapping_df[mapping_df['LEGEND'] == code]['INFO'].tolist()
#     print(f"LEGEND {code}: {infos}")






