import pickle
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from src.models.sinkhole_modeling.data_loader import load_dataset
from src.models.sinkhole_modeling.run_enhanced.feature_importance import analyze_false_negatives
import sys
sys.path.append("/Users/macforhsj/Desktop/SinkholeCivicSentinel/src/models/sinkhole_modeling/run_enhanced")

baseline_path = "/Users/macforhsj/Desktop/SinkholeCivicSentinel/src/models/sinkhole_modeling/run_enhanced/enhanced_gnn_results/baseline_model.pkl"
enhanced_path = "/Users/macforhsj/Desktop/SinkholeCivicSentinel/src/models/sinkhole_modeling/run_enhanced/enhanced_gnn_results/enhanced_model.pkl"




with open(baseline_path, 'rb') as f:
    baseline_model = pickle.load(f)

with open(enhanced_path, 'rb') as f:
    enhanced_model = pickle.load(f)

# === Step 1: Prepare data (assumes X, y already loaded)
gdf = load_dataset(table="feature_matrix_100_geo")
if 'pothole_kde_density' not in gdf.columns:
    gdf['pothole_kde_density'] = np.random.random(len(gdf)) * 0.5

y = gdf["subsidence_occurrence"].astype(int)
X = gdf.drop(columns=["subsidence_occurrence", "subsidence_count"], errors="ignore")

# === Step 2: Run FN analysis
fn_base = analyze_false_negatives(baseline_model, X, y, top_k=100)
fn_enh = analyze_false_negatives(enhanced_model, X, y, top_k=100)

print("🔹 Baseline FN 수:", len(fn_base))
print("🔹 Enhanced FN 수:", len(fn_enh))

recovered_ids = set(fn_base['grid_id']) - set(fn_enh['grid_id'])
print(f"🔹 Enhanced 모델이 복구한 FN 수: {len(recovered_ids)}")

# === Step 3: Plot comparison
gdf_fn_base = gpd.GeoDataFrame(fn_base, geometry='geometry', crs="EPSG:5179")
gdf_fn_enh = gpd.GeoDataFrame(fn_enh, geometry='geometry', crs="EPSG:5179")

fig, ax = plt.subplots(figsize=(10, 10))
gdf_fn_base.plot(ax=ax, color='red', label='Baseline FN', alpha=0.5, markersize=10)
gdf_fn_enh.plot(ax=ax, color='green', label='Enhanced FN', alpha=0.5, markersize=10)
plt.legend()
plt.title("False Negatives: Baseline vs Enhanced GNN")
plt.tight_layout()
plt.show()