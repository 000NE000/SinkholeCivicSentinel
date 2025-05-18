import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier
import shap
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")
from sqlalchemy import create_engine
from dotenv import load_dotenv

# === 0. Load Data from DB (one-shot join) ===
load_dotenv()
engine = create_engine(os.getenv("DB_DSN"))

sql = """
-- final_score 제거한 버전
SELECT 
    p.grid_id,
    p.pothole_report_count,
    p.pothole_report_density,
    p.pothole_kde_density,
    p.pothole_nearest_3_dist,
    COALESCE(s.subsidence_occurrence, 0) AS subsidence_occurrence,
    f.weighted_sinkhole_density,
    f.sinkhole_area_pipe_risk,
    f.road_type_score,
    f.risk_road_load,
    f.soil_road_stress,
    f.risk_fault,
    f.sinkhole_spatial_stddev,
    f.drainage_road_risk,
    f.neighboring_pipe_risk_avg,
    f.min_distance_to_sinkhole,
    f.total_road_length,
    f.neighboring_drainage_risk_avg,
    f.fault_pipe_risk,
    f.risk_stone_content,
    f.neighboring_pipe_risk_min,
    f.risk_hydro,
    f.road_network_complexity,
    f.soil_pipe_risk,
    f.hydro_pipe_risk,
    f.fault_road_stress,
    f.nearby_sinkhole_grid_count,
    f.risk_drainage,
    f.road_area_ratio,
    f.infrastructure_vulnerability,
    f.drainage_pipe_risk,
    f.nearby_sinkhole_total_count,
    f.fracture_pipe_risk,
    f.pothole_pipe_risk,
    f.risk_soil_depth,
    f.risk_fracture_density
FROM pothole_features p
LEFT JOIN feature_matrix_100 f ON p.grid_id = f.grid_id
LEFT JOIN (
    SELECT grid_id, 1 AS subsidence_occurrence
    FROM grid_sinkhole_table
) s ON p.grid_id = s.grid_id;
"""
df = pd.read_sql(sql, engine)

# === 1. 분포 차이 시각화 ===
features = ['pothole_report_count', 'pothole_report_density',
            'pothole_kde_density', 'pothole_nearest_3_dist']

df_pos = df[df.subsidence_occurrence == 1]
df_neg = df[df.subsidence_occurrence == 0]

for feat in features:
    plt.figure()
    sns.kdeplot(df_pos[feat], label='Positive')
    sns.kdeplot(df_neg[feat], label='Negative')
    plt.title(f"[분포 비교] {feat}")
    plt.legend()
    plt.savefig(f"distribution_{feat}.png")
    plt.close()

# === 2. 단일변수 성능 평가 ===
print("\n[단일 피처 성능 비교]")
y_true = df.subsidence_occurrence
for feat in features:
    y_score = df[feat]
    print(f"{feat:<30} | ROC-AUC: {roc_auc_score(y_true, y_score):.4f} | PR-AUC: {average_precision_score(y_true, y_score):.4f}")

# === 3. SHAP 기여도 비교 ===
print("\n[SHAP 기여도 비교]")

# 자동으로 baseline_feats 뽑기
baseline_feats = [c for c in df.columns
                  if c not in features
                     and c not in ('grid_id','final_score','subsidence_occurrence')]

X_base = df[baseline_feats]
X_enh  = df[baseline_feats + features]

model_base = LGBMClassifier(random_state=42).fit(X_base, y_true)
model_enh  = LGBMClassifier(random_state=42).fit(X_enh, y_true)

explainer_base = shap.TreeExplainer(model_base)
explainer_enh  = shap.TreeExplainer(model_enh)

shap_base = explainer_base.shap_values(X_base)[1]
shap_enh  = explainer_enh.shap_values(X_enh)[1]

for data, title, fname in [
    (shap_base, "Baseline",   "shap_baseline.png"),
    (shap_enh,  "With Pothole Features", "shap_enhanced.png")
]:
    shap.summary_plot(data, X_base if title=="Baseline" else X_enh,
                      show=False)
    plt.title(title)
    plt.savefig(fname)
    plt.close()

# === 4. False-Negative 보완율 ===
print("\n[False-Negative 보완율]")
pred_base = model_base.predict(X_base)
pred_enh  = model_enh.predict(X_enh)

fn_idx  = df[(y_true==1)&(pred_base==0)].index
recover = fn_idx.intersection(df[pred_enh==1].index)
if len(fn_idx):
    rate = len(recover)/len(fn_idx)
    print(f"FN recover rate: {len(recover)}/{len(fn_idx)} = {rate:.2%}")
else:
    print("No false negatives to recover")

# === 5. Crowd-Sensing 커버리지 비교 ===
print("\n[Crowd-Sensing 커버리지 비교]")

df['density_q'] = pd.qcut(df.pothole_report_density, 4, labels=False, duplicates='drop')

for grp in [0, 3]:
    sub = df[df.density_q == grp]
    # 밀도 기준으로 정렬 → 가장 "제보가 많았던 지역" 100개 선택
    topk = sub.sort_values('pothole_report_density', ascending=False).head(100)

    recall = topk.subsidence_occurrence.sum() / df.subsidence_occurrence.sum()
    print(f"group {grp} recall@100: {recall:.2%}")