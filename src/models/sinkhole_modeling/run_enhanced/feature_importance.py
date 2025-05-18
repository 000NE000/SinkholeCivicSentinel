# feature_importance.py

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from src.models.sinkhole_modeling.evaluation import compute_comprehensive_metrics


def calculate_feature_importance(model, X, y):
    import pandas as pd, numpy as np
    print("Calculating feature importance...")

    # 1) 우선 stage2_model이 있으면 shap 사용
    try:
        if hasattr(model, 'stage2_model'):
            import shap
            # LightGBM도, GNN도 일단 트리/SkLearn wrapper로 shap 값 계산
            explainer = shap.TreeExplainer(model.stage2_model)
            # Stage2에 쓰는 모든 피처
            feat_names = model.stage2_feature_names
            shap_vals = explainer.shap_values(X[feat_names])
            # positive 클래스 기여도 절댓값 평균
            imp = np.abs(shap_vals).mean(0)
            importance_df = pd.DataFrame({
                'feature': feat_names,
                'importance': imp
            }).sort_values('importance', ascending=False).reset_index(drop=True)
            return importance_df
    except Exception:
        pass

    # 2) fallback: 랜덤포레스트 surrogate
    from sklearn.ensemble import RandomForestClassifier
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and c not in ['geometry','grid_id']]
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X[numeric], y)
    imp = rf.feature_importances_
    importance_df = pd.DataFrame({'feature': numeric, 'importance': imp}) \
                      .sort_values('importance', ascending=False).reset_index(drop=True)
    return importance_df

def plot_feature_importance(importance_df, title="Feature Importance", top_n=15, figsize=(10, 6)):
    """
    Plot feature importance

    Args:
        importance_df: DataFrame with feature importance
        title: Plot title
        top_n: Number of top features to show
        figsize: Figure size as (width, height)

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Use top N features for plotting
    plot_df = importance_df.head(top_n).copy()

    # Create the plot
    plt.figure(figsize=figsize)
    sns.barplot(x='importance', y='feature', data=plot_df)
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()

    return plt.gcf()


def run_ablation_study(X, y, spatial_blocks, important_features=None):
    """
    Run ablation study to measure impact of removing each feature

    Args:
        X: Feature dataframe
        y: Target variable
        spatial_blocks: Spatial blocks for cross-validation
        important_features: List of important features to test (if None, uses top 10)

    Returns:
        DataFrame with ablation results
    """
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

    print("Running ablation study...")

    # Step 1: Get numeric features only (exclude non-numeric and irrelevant columns)
    numeric_features = [
        col for col in X.columns
        if pd.api.types.is_numeric_dtype(X[col]) and col not in ['geometry', 'grid_id']
    ]

    # Step 2: Train RandomForest to get importances
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X[numeric_features], y)

    importances = rf.feature_importances_
    features = numeric_features  # ensure same length

    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Step 3: Select top 10 important features
    if important_features is None:
        important_features = importance_df.head(10)['feature'].tolist()

    # Step 4: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train baseline model with all features
    baseline_model = GradientBoostingClassifier(random_state=42)
    baseline_model.fit(X_train[numeric_features], y_train)
    baseline_preds = baseline_model.predict_proba(X_test[numeric_features])[:, 1]

    # Baseline metrics
    precision, recall, _ = precision_recall_curve(y_test, baseline_preds)
    baseline_pr_auc = auc(recall, precision)
    baseline_roc_auc = roc_auc_score(y_test, baseline_preds)

    # Step 5: Run ablation loop
    results = []
    for feature in important_features:
        print(f"Testing without feature: {feature}")

        drop_features = [feature]
        X_train_ablation = X_train.drop(columns=drop_features, errors='ignore')
        X_test_ablation = X_test.drop(columns=drop_features, errors='ignore')

        # Ensure only numeric columns
        ablation_features = [
            col for col in X_train_ablation.columns
            if pd.api.types.is_numeric_dtype(X_train_ablation[col])
        ]

        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train_ablation[ablation_features], y_train)
        preds = model.predict_proba(X_test_ablation[ablation_features])[:, 1]

        precision, recall, _ = precision_recall_curve(y_test, preds)
        pr_auc = auc(recall, precision)
        roc_auc = roc_auc_score(y_test, preds)

        results.append({
            'feature': feature,
            'baseline_pr_auc': baseline_pr_auc,
            'ablation_pr_auc': pr_auc,
            'pr_auc_change': baseline_pr_auc - pr_auc,
            'baseline_roc_auc': baseline_roc_auc,
            'ablation_roc_auc': roc_auc,
            'roc_auc_change': baseline_roc_auc - roc_auc
        })

    return pd.DataFrame(results).sort_values('pr_auc_change', ascending=False)

# 2) analyze_false_negatives: Top-K 기반으로 FN 찾아내기
def analyze_false_negatives(model, X, y, top_k=100):
    import pandas as pd, numpy as np
    print("Analyzing false negatives (Top-K)...")

    # Stage2까지 합친 final score 계산
    try:
        # predict_proba가 있으면 확률, 아니면 decision_function
        if hasattr(model, 'predict_proba'):
            scores = model.predict_proba(X)[:,1]
        else:
            scores = model.decision_function(X)
    except:
        # fallback to stage1
        scores = model.stage1_model.predict_proba(X[[model.proximity_feat]])[:,1]

    # Top-K 양성 예측 지점 index
    pred_idx = np.argsort(scores)[::-1][:top_k]
    # 실제 침하(1)인데 top-K에 못 든 것들이 false negative
    fn_mask = (y==1) & (~np.isin(np.arange(len(y)), pred_idx))
    false_neg = X[fn_mask].copy()
    if false_neg.empty:
        print("No false negatives in Top-K")
        return pd.DataFrame()

    false_neg['score'] = scores[fn_mask]
    false_neg['distance_to_cutoff'] = scores[pred_idx[-1]] - false_neg['score']
    return false_neg


import geopandas as gpd
import numpy as np
import pandas as pd


def identify_pothole_hotspots(X: pd.DataFrame, y: pd.Series, model) -> gpd.GeoDataFrame:
    # 1) Ensure GeoDataFrame with CRS
    if not isinstance(X, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(X.copy(), geometry='geometry', crs="EPSG:5179")
    else:
        gdf = X.copy()
        gdf.crs = "EPSG:5179"

    # 2) Pick your pipe-risk column dynamically
    pipe_cols = [c for c in gdf.columns if c.endswith('_pipe_risk')]
    if not pipe_cols:
        print("No pipe_risk column found; skipping hotspot analysis.")
        return gdf
    pipe_col = pipe_cols[0]

    # 3) Compute final risk scores
    try:
        scores = model.predict_proba(gdf)[:, 1]
    except:
        # fallback to stage1
        scores = model.stage1_model.predict_proba(gdf[[model.proximity_feat]])[:, 1]
    gdf['risk_score'] = scores
    gdf['actual'] = y.values

    # 4) Define strict “top 10%” as > (not >=) 90th percentile
    r_th = np.percentile(scores, 90)
    p_th = np.percentile(gdf['pothole_kde_density'], 75)
    pi_th = np.percentile(gdf[pipe_col], 75)

    gdf['is_high_risk'] = gdf['risk_score'] > r_th
    gdf['is_pothole_hotspot'] = gdf['is_high_risk'] & (gdf['pothole_kde_density'] > p_th)
    gdf['is_pipe_hotspot'] = gdf['is_high_risk'] & (gdf[pipe_col] > pi_th)
    gdf['is_super_hotspot'] = gdf['is_pothole_hotspot'] & gdf['is_pipe_hotspot']

    stats = {
        'total': len(gdf),
        'high_risk': gdf['is_high_risk'].sum(),
        'pothole_hotspots': gdf['is_pothole_hotspot'].sum(),
        'pipe_hotspots': gdf['is_pipe_hotspot'].sum(),
        'super_hotspots': gdf['is_super_hotspot'].sum()
    }
    print("Hotspot stats:", stats)
    if stats['super_hotspots'] > 0:
        prec = gdf.loc[gdf['is_super_hotspot'], 'actual'].mean()
        print(f"Super-hotspot precision: {prec:.4f}")

    return gdf
