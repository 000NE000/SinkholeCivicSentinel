import os
import pickle
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import sys

# ─── 1) Setup & Data Load ─────────────────────────────────────────────────────────

sys.path.append("/Users/macforhsj/Desktop/SinkholeCivicSentinel/src/models/sinkhole_modeling/run_enhanced")
from src.models.sinkhole_modeling.data_loader import load_dataset

# PostGIS 연결
load_dotenv()
engine = create_engine(os.getenv("DB_DSN"))

# Grid GeoDataFrame 불러오기
gdf = load_dataset(table="feature_matrix_100_geo")
if gdf.crs is None:
    gdf.set_crs(epsg=5179, inplace=True)
gdf = gdf.to_crs(epsg=3857)

# 포트홀 KDE 병합
pothole_df = pd.read_sql("SELECT grid_id, pothole_kde_density FROM feature_pothole_all", engine)
gdf = gdf.merge(pothole_df, on="grid_id", how="left")
gdf["pothole_kde_density"] = gdf["pothole_kde_density"].fillna(0)

# 서울시 경계로 클리핑
seoul = gpd.read_file("/Users/macforhsj/Desktop/SinkholeCivicSentinel/data/raw/seoul_district/LARD_ADM_SECT_SGG_11_202505.shp") \
            .to_crs(epsg=3857)
gdf = gpd.clip(gdf, seoul)

# ─── 2) Risk Score Computation ────────────────────────────────────────────────────

# Enhanced GNN 모델 로드
with open("/Users/macforhsj/Desktop/SinkholeCivicSentinel/src/models/sinkhole_modeling/run_enhanced/enhanced_gnn_results/baseline_model.pkl", "rb") as f:
    enhanced_model = pickle.load(f)

# 예측용 데이터 준비
X = gdf.drop(columns=["subsidence_occurrence","subsidence_count","geometry"], errors="ignore")

# risk_score 예측
if hasattr(enhanced_model, "predict_proba"):
    gdf["risk_score"] = enhanced_model.predict_proba(X)[:,1]
else:
    feat = enhanced_model.proximity_feat
    gdf["risk_score"] = enhanced_model.stage1_model.predict_proba(X[[feat]])[:,1]

# ─── 3) Super-Hotspots & Top-500 ─────────────────────────────────────────────────

# 파이프 리스크 컬럼
pipe_col = [c for c in X.columns if c.endswith("_pipe_risk")][0]

# 임계값
r_th  = gdf["risk_score"].quantile(0.975)
p_th  = gdf["pothole_kde_density"].quantile(0.90)
pi_th = gdf[pipe_col].quantile(0.90)

# super-hotspot 플래그
gdf["is_super_hotspot"] = (
    (gdf["risk_score"] >= r_th)
    & (gdf["pothole_kde_density"] >= p_th) &
    (gdf[pipe_col] >= pi_th)
)

# Top-500 셀 추출 및 centroid 변환
top500 = gdf.nlargest(1000, "risk_score").copy()
top500["geometry"] = top500.geometry.centroid

# 그중 super-hotspot
hot500 = top500[top500["is_super_hotspot"]]

# 관측 싱크홀 지점 (centroid)
subs_pts = gdf[gdf["subsidence_occurrence"] == 1].copy()
subs_pts["geometry"] = subs_pts.geometry.centroid

# ─── 4) Plot Everything ──────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12,12))

# 4.1) Top-500 risk centroids heatmap
vmin, vmax = top500["risk_score"].min(), top500["risk_score"].max()
# top500.plot(
#     ax=ax,
#     column="risk_score",
#     cmap="hot",
#     markersize=30,
#     alpha=0.8,
#     legend=False,
#     linewidth=0
# )

# 4.2) Super-hotspot cyan outline + halos
hot500.plot(
    ax=ax,
    facecolor="none",
    edgecolor="deepskyblue",
    linewidth=3,
    alpha=0.9,
    zorder=4
)
# for geom in hot500.geometry:
#     x, y = geom.x, geom.y
#     for radius, lw, alpha in [(150, 3, 1.0), (300, 2, 0.7), (450, 1.5, 0.4)]:
#         ax.add_patch(mpatches.Circle(
#             (x,y), radius=radius,
#             edgecolor="cyan", facecolor="none",
#             linewidth=lw, alpha=alpha, zorder=3
#         ))

# 4.3) Observed sinkholes as white stars
# subs_pts.plot(
#     ax=ax,
#     markersize=100,
#     marker="*",
#     facecolor="white",
#     edgecolor="black",
#     linewidth=1.5,
#     zorder=5
# )

# 4.4) 서울시 외곽 경계 (light)
seoul.boundary.plot(ax=ax, edgecolor="gray", linewidth=0.5, alpha=0.3)

# 4.5) Basemap
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

# 4.6) Legend
legend_elems = [
    Line2D([0],[0], marker="o", color="w", label="Top-500 Risk Cells",
           markerfacecolor="grey", markersize=10, alpha=0.8),
    Line2D([0],[0], color="cyan", lw=3, label="Super-hotspots"),
    Line2D([0],[0], marker="*", color="w", label="Observed Sinkhole",
           markerfacecolor="white", markeredgecolor="black", markersize=15)
]
ax.legend(handles=legend_elems, loc="lower left", fontsize=12)

ax.set_title(
    "Seoul Sinkhole Top-500 Risk Centroids\nSuper-hotspots & Observed Events",
    fontsize=16
)
ax.set_axis_off()
plt.tight_layout()
# plt.savefig("enhanced_model_hotspot_vs_sinkhole.png", dpi=1200)
plt.show()