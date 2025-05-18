from __future__ import annotations

import os
import time
from datetime import datetime
from typing import List, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.wkb
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold
import lightgbm as lgb
import shap


# ────────────────────────────────────────────────────────────────────────────────
# Utility logging
# ────────────────────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}")


# ────────────────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────────────────
def load_dataset(table: str = "feature_matrix_25_geo", chunksize: int = 100_000) -> gpd.GeoDataFrame:
    load_dotenv()
    dsn = os.getenv("DB_DSN")
    if not dsn:
        raise RuntimeError("DB_DSN not set.")
    engine = create_engine(dsn)
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_sql_query(f"SELECT * FROM {table}", engine, chunksize=chunksize):
        if "geom" not in chunk:
            raise ValueError("Missing 'geom' column.")
        chunk["geometry"] = chunk["geom"].apply(lambda h: shapely.wkb.loads(bytes.fromhex(h)))
        parts.append(chunk.drop(columns=["geom"]))
    gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry")
    if "subsidence_occurrence" not in gdf:
        raise ValueError("Missing 'subsidence_occurrence' column.")
    pos = int(gdf["subsidence_occurrence"].sum())
    log(f"Loaded {len(gdf):,} rows; positives {pos} ({pos / len(gdf):.4%})")
    return gdf


# ────────────────────────────────────────────────────────────────────────────────
# Feature engineering utilities
# ────────────────────────────────────────────────────────────────────────────────
def get_feature_importance(model: lgb.LGBMClassifier, X: pd.DataFrame, top_n: int = 30) -> Dict[str, float]:
    """Get feature importance from model and return as dictionary."""
    importances = model.feature_importances_
    names = X.columns
    imp_dict = {names[i]: importances[i] for i in range(len(names))}
    # Sort by importance in descending order
    sorted_imp = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))

    # Print top N features
    log(f"Top {top_n} features by model importance:")
    for i, (feat, imp) in enumerate(list(sorted_imp.items())[:top_n]):
        log(f"  {i + 1:2d}. {feat:30s}: {imp:.6f}")

    return sorted_imp


def perform_shap_analysis(model: lgb.LGBMClassifier, X: pd.DataFrame, sample_size: int = 1000) -> Dict[str, float]:
    """Perform SHAP analysis and return feature importance dictionary."""
    log("Performing SHAP analysis...")
    # Sample data for SHAP analysis
    X_sample = X.sample(min(sample_size, len(X)), random_state=42)

    # Create explainer and get SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # If binary classification, shap_values is a list with one element (positive class)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = {X_sample.columns[i]: np.abs(shap_values[:, i]).mean()
                     for i in range(X_sample.shape[1])}

    # Sort features by importance (descending)
    shap_sorted = dict(sorted(mean_abs_shap.items(), key=lambda item: item[1], reverse=True))

    # Log top features by SHAP importance
    log("Top 30 features by SHAP importance:")
    for i, (feat, value) in enumerate(list(shap_sorted.items())[:30]):
        log(f"  {i + 1:2d}. {feat:30s}: {value:.6f}")

    return shap_sorted


def filter_correlated(X: pd.DataFrame, thresh: float = 0.95) -> pd.DataFrame:
    """Remove highly correlated features."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [col for col in upper.columns if any(upper[col] > thresh)]
    log(f"Dropping {len(drop)} correlated features")
    for col in drop:
        correlated_with = [idx for idx in upper.index if upper.loc[idx, col] > thresh]
        if correlated_with:
            log(f"  {col} correlated with {correlated_with}")
    return X.drop(columns=drop)


# ────────────────────────────────────────────────────────────────────────────────
# Ranking metrics
# ────────────────────────────────────────────────────────────────────────────────
def topk_metrics(y: Sequence[int], scores: Sequence[float], ks: List[int]) -> Dict[str, float]:
    arr_y = np.array(y)
    arr_s = np.array(scores)
    order = np.argsort(arr_s)[::-1]
    arr_y = arr_y[order]
    total = arr_y.sum()
    rate = total / len(arr_y)
    out: dict[str, float] = {}
    for k in ks:
        if k > len(arr_y): continue
        tp = arr_y[:k].sum()
        out[f"recall@{k}"] = tp / total
        out[f"precision@{k}"] = tp / k
        out[f"lift@{k}"] = (tp / k) / rate
    return out


# ────────────────────────────────────────────────────────────────────────────────
# Model training
# ────────────────────────────────────────────────────────────────────────────────
LGB_BASE = dict(
    objective="binary", metric="auc", boosting_type="gbdt",
    learning_rate=0.05, num_leaves=31, feature_fraction=0.9,
    bagging_fraction=0.8, bagging_freq=5, min_data_in_leaf=20,
    reg_alpha=0.1, reg_lambda=1.0, max_depth=7, verbose=-1)


def train_lgb(X_tr, y_tr, X_va, y_va, spw: int) -> lgb.LGBMClassifier:
    p = LGB_BASE.copy()
    p["scale_pos_weight"] = spw
    m = lgb.LGBMClassifier(**p)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def isolation_risk(X: pd.DataFrame, cont: float = 0.02) -> np.ndarray:
    iso = IsolationForest(n_estimators=200, contamination=cont,
                          random_state=42, n_jobs=-1)
    iso.fit(X)
    return -iso.decision_function(X)


# ────────────────────────────────────────────────────────────────────────────────
# Feature selection pipeline
# ────────────────────────────────────────────────────────────────────────────────
def evaluate_feature_impact(X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series,
                            features_to_test: List[str], fixed_spw: int) -> Dict[str, float]:
    """Evaluate impact of removing individual features on Recall@100."""
    results = {}

    # First get baseline performance with all features
    baseline_model = train_lgb(X_train, y_train, X_val, y_val, spw=fixed_spw)
    baseline_scores = baseline_model.predict_proba(X_val)[:, 1]
    baseline_recall = topk_metrics(y_val, baseline_scores, [100])["recall@100"]
    log(f"Baseline Recall@100: {baseline_recall:.4f}")
    results["baseline"] = baseline_recall

    # Test removal of each feature
    for feat in features_to_test:
        reduced_features = [f for f in X_train.columns if f != feat]

        # Retrain model without this feature
        model = train_lgb(X_train[reduced_features], y_train, X_val[reduced_features], y_val, spw=fixed_spw)

        # Evaluate performance
        scores = model.predict_proba(X_val[reduced_features])[:, 1]
        recall_at_100 = topk_metrics(y_val, scores, [100])["recall@100"]

        # Store result
        results[feat] = recall_at_100

        # Log result
        delta = recall_at_100 - baseline_recall
        direction = "↑" if delta > 0 else "↓"
        log(f"Removed {feat:30s} → Recall@100: {recall_at_100:.4f} ({direction}{abs(delta):.4f})")

    return results


def select_from_feature_groups(feature_groups: Dict[str, List[str]],
                               shap_importances: Dict[str, float],
                               features_per_group: int = 3) -> List[str]:
    """Select top N features from each group based on SHAP values."""
    final_features = []

    log("Selecting features from groups:")
    for group_name, features in feature_groups.items():
        # Filter to features that exist in our SHAP values
        available_features = [f for f in features if f in shap_importances]

        if not available_features:
            log(f"  Group '{group_name}': No features available")
            continue

        # Rank by SHAP importance
        ranked_features = sorted(available_features,
                                 key=lambda f: shap_importances.get(f, 0),
                                 reverse=True)

        # Select top N
        selected = ranked_features[:features_per_group]
        final_features.extend(selected)

        log(f"  Group '{group_name}': Selected {len(selected)} features")
        for f in selected:
            log(f"    - {f:30s}: {shap_importances.get(f, 0):.6f}")

    # Ensure unique
    final_features = list(dict.fromkeys(final_features))
    return final_features


# ────────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────────────────────────
def main():
    # Load dataset
    gdf = load_dataset()
    y = gdf["subsidence_occurrence"].astype(int)
    X = gdf.drop(columns=[c for c in ["grid_id", "subsidence_occurrence", "subsidence_count", "geometry"] if c in gdf])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 1. Lock in fixed scale_pos_weight
    fixed_spw = 300
    log(f"Using fixed scale_pos_weight={fixed_spw} for all training")

    # 2. Train baseline model
    log("Training baseline model")
    base_model = train_lgb(X_train, y_train, X_val, y_val, spw=fixed_spw)
    base_scores = base_model.predict_proba(X_val)[:, 1]
    base_metrics = topk_metrics(y_val, base_scores, [100, 200, 500])
    log(f"Baseline metrics: {base_metrics}")

    # 3. Get model feature importance
    model_importances = get_feature_importance(base_model, X_train)

    # 4. Compute SHAP feature importances
    log("Computing SHAP feature importances")
    shap_importances = perform_shap_analysis(base_model, X_train)

    # 5. Identify lowest SHAP-importance features for pruning
    all_features = list(shap_importances.keys())
    lowest_shap_features = all_features[-3:]
    log(f"Lowest SHAP features to consider pruning: {lowest_shap_features}")

    # 6. Evaluate Recall@100 impact of removing low-importance features
    impact_results = evaluate_feature_impact(
        X_train, y_train, X_val, y_val, lowest_shap_features, fixed_spw
    )

    # 7. Define feature groups
    feature_groups = {
        "GeoSoil": [
            'risk_soil_depth', 'risk_stone_content', 'risk_fault', 'risk_hydro',
            'risk_fracture_density', 'risk_drainage', 'infrastructure_vulnerability', 'dynamic_signal'
        ],
        "RoadNetwork": [
            'risk_road_load', 'total_road_length', 'road_area_ratio', 'road_type_score',
            'major_road_presence', 'road_density', 'road_network_complexity'
        ],
        "PipeRisk": [
            'soil_pipe_risk', 'fault_pipe_risk', 'hydro_pipe_risk',
            'fracture_pipe_risk', 'drainage_pipe_risk', 'pothole_pipe_risk'
        ],
        "CrossStress": [
            'soil_road_stress', 'fault_road_stress', 'drainage_road_risk', 'pothole_drainage_risk'
        ],
        "NeighborAggregates": [
            'neighboring_pipe_risk_avg', 'neighboring_pipe_risk_min',
            'neighboring_pipe_risk_max', 'neighboring_drainage_risk_avg'
        ],
        "ObservedSinkhole": [
            'min_distance_to_sinkhole', 'nearby_sinkhole_grid_count',
            'nearby_sinkhole_total_count', 'sinkhole_spatial_stddev',
            'weighted_sinkhole_density', 'sinkhole_area_pipe_risk'
        ],
        "Meta": [
            'grid_id', 'subsidence_occurrence', 'subsidence_count'
        ]
    }

    # For actual running, we'd replace the group contents with columns from our dataset
    # This helps handle cases where some features might not be in our dataset
    actual_groups = {}
    for group, features in feature_groups.items():
        # Keep only features that are in our dataset
        available = [f for f in features if f in X.columns]
        if available:
            actual_groups[group] = available

    # 8. Select top features from each group based on SHAP values
    final_feature_set = select_from_feature_groups(actual_groups, shap_importances, features_per_group=3)

    # 9. Log final feature set
    log(f"Final selected features: {final_feature_set}")
    log(f"Total features: {len(final_feature_set)}")

    # 10. Retrain model with final feature set for validation
    final_model = train_lgb(X_train[final_feature_set], y_train,
                            X_val[final_feature_set], y_val, spw=fixed_spw)
    final_scores = final_model.predict_proba(X_val[final_feature_set])[:, 1]
    final_metrics = topk_metrics(y_val, final_scores, [100, 200, 500])

    # 11. Compare baseline vs final metrics
    log(f"Final metrics with selected features: {final_metrics}")
    for k in [100, 200, 500]:
        baseline = base_metrics[f"recall@{k}"]
        final = final_metrics[f"recall@{k}"]
        diff = final - baseline
        direction = "↑" if diff > 0 else "↓"
        log(f"Recall@{k}: {baseline:.4f} → {final:.4f} ({direction}{abs(diff):.4f})")

    # 12. Train isolation forest on selected features
    iso_scores = isolation_risk(X_val[final_feature_set])
    iso_metrics = topk_metrics(y_val, iso_scores, [100, 200, 500])
    log(f"Isolation Forest metrics with selected features: {iso_metrics}")

    # 13. Try simple ensemble
    ensemble_scores = 0.7 * final_scores + 0.3 * iso_scores
    ensemble_metrics = topk_metrics(y_val, ensemble_scores, [100, 200, 500])
    log(f"Ensemble metrics with selected features: {ensemble_metrics}")


if __name__ == "__main__":
    t0 = time.time();
    main();
    log(f"Done in {time.time() - t0:.1f}s")