from __future__ import annotations

import os
import time
from datetime import datetime
from typing import List, Dict, Sequence, Tuple, Set

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


def get_correlation_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix for features."""
    return X.corr().abs()


def are_features_correlated(corr_matrix: pd.DataFrame, feat1: str, feat2: str, threshold: float = 0.8) -> bool:
    """Check if two features are highly correlated."""
    if feat1 not in corr_matrix.index or feat2 not in corr_matrix.columns:
        return False
    return corr_matrix.loc[feat1, feat2] > threshold


def find_correlated_features(corr_matrix: pd.DataFrame, feature_set: List[str],
                             threshold: float = 0.8) -> List[Tuple[str, str, float]]:
    """Find pairs of highly correlated features within a feature set."""
    correlated_pairs = []
    for i, feat1 in enumerate(feature_set):
        for feat2 in feature_set[i + 1:]:
            if feat1 in corr_matrix.index and feat2 in corr_matrix.columns:
                corr_value = corr_matrix.loc[feat1, feat2]
                if corr_value > threshold:
                    correlated_pairs.append((feat1, feat2, corr_value))
    return correlated_pairs


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
# Advanced Feature Selection Pipeline
# ────────────────────────────────────────────────────────────────────────────────

def select_features_by_cumulative_shap(group_features: List[str],
                                       shap_dict: Dict[str, float],
                                       threshold: float = 0.8) -> List[str]:
    """
    Select features based on cumulative SHAP contribution.
    Selects minimum number of features to reach threshold of total group SHAP.
    """
    # Filter to features that exist in SHAP dictionary
    available_features = [f for f in group_features if f in shap_dict]

    if not available_features:
        return []

    # Sort by SHAP importance
    sorted_features = sorted(available_features, key=lambda f: shap_dict[f], reverse=True)

    # Calculate total SHAP for this group
    total_shap = sum(shap_dict[f] for f in available_features)

    # Select features until we reach the cumulative threshold
    selected_features = []
    cumulative_shap = 0

    for feat in sorted_features:
        selected_features.append(feat)
        cumulative_shap += shap_dict[feat]

        # Check if we've reached the threshold
        if cumulative_shap / total_shap >= threshold:
            break

    return selected_features


def filter_correlated_features(selected_features: List[str],
                               corr_matrix: pd.DataFrame,
                               shap_dict: Dict[str, float],
                               threshold: float = 0.8) -> List[str]:
    """
    Remove highly correlated features from the selection,
    keeping the one with higher SHAP importance.
    """
    # Find correlated pairs
    correlated_pairs = find_correlated_features(corr_matrix, selected_features, threshold)

    # If no correlated features, return the original selection
    if not correlated_pairs:
        return selected_features

    # Keep track of features to remove
    to_remove = set()

    # Process each correlated pair
    for feat1, feat2, corr_value in correlated_pairs:
        log(f"High correlation ({corr_value:.4f}) between {feat1} and {feat2}")

        # Keep the feature with higher SHAP value
        if shap_dict.get(feat1, 0) >= shap_dict.get(feat2, 0):
            log(f"  Keeping {feat1} (SHAP: {shap_dict.get(feat1, 0):.6f})")
            log(f"  Removing {feat2} (SHAP: {shap_dict.get(feat2, 0):.6f})")
            to_remove.add(feat2)
        else:
            log(f"  Keeping {feat2} (SHAP: {shap_dict.get(feat2, 0):.6f})")
            log(f"  Removing {feat1} (SHAP: {shap_dict.get(feat1, 0):.6f})")
            to_remove.add(feat1)

    # Filter out the features to remove
    return [f for f in selected_features if f not in to_remove]


def greedy_feature_selection(X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame, y_val: pd.Series,
                             candidate_features: List[str],
                             current_features: List[str],
                             spw: int,
                             min_improvement: float = 0.005,
                             max_features: int = 20) -> List[str]:
    """
    Greedily select features based on performance improvement.
    Adds features one by one, stopping when improvements fall below threshold.
    """
    # Start with current features and calculate baseline performance
    selected = current_features.copy()

    if not selected:  # If no features selected yet, start with the first candidate
        selected = [candidate_features[0]]
        candidate_features = candidate_features[1:]

    # Calculate baseline performance
    X_train_sel = X_train[selected]
    X_val_sel = X_val[selected]
    model = train_lgb(X_train_sel, y_train, X_val_sel, y_val, spw=spw)
    scores = model.predict_proba(X_val_sel)[:, 1]
    best_recall = topk_metrics(y_val, scores, [100])["recall@100"]

    log(f"Starting greedy selection with {len(selected)} features. Baseline Recall@100: {best_recall:.4f}")

    # Iterate until we reach max features or improvements are too small
    while len(selected) < max_features and candidate_features:
        best_improvement = 0
        best_feature = None

        # Try adding each candidate feature
        for feature in candidate_features:
            trial_features = selected + [feature]
            X_train_trial = X_train[trial_features]
            X_val_trial = X_val[trial_features]

            # Train model and evaluate
            trial_model = train_lgb(X_train_trial, y_train, X_val_trial, y_val, spw=spw)
            trial_scores = trial_model.predict_proba(X_val_trial)[:, 1]
            trial_recall = topk_metrics(y_val, trial_scores, [100])["recall@100"]

            # Calculate improvement
            improvement = trial_recall - best_recall

            if improvement > best_improvement:
                best_improvement = improvement
                best_feature = feature

        # If best improvement is above threshold, add the feature
        if best_improvement >= min_improvement and best_feature:
            selected.append(best_feature)
            candidate_features.remove(best_feature)
            best_recall += best_improvement
            log(f"Added {best_feature}: Recall@100 improved by {best_improvement:.4f} to {best_recall:.4f}")
        else:
            if best_feature:
                log(f"Stopping: Best improvement ({best_improvement:.4f}) below threshold ({min_improvement:.4f})")
            else:
                log("Stopping: No improvement found")
            break

    return selected


def advanced_feature_selection(X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series,
                               feature_groups: Dict[str, List[str]],
                               shap_dict: Dict[str, float],
                               spw: int) -> List[str]:
    """
    Advanced feature selection combining:
    1. Cumulative SHAP contribution (80% threshold)
    2. Correlation filtering
    3. Greedy performance-based selection
    """
    log("Starting advanced feature selection...")

    # Calculate correlation matrix once
    corr_matrix = get_correlation_matrix(X_train)

    # Step 1: Select features by cumulative SHAP contribution for each group
    group_selections = {}
    combined_selection = []

    log("Step 1: Selecting features by cumulative SHAP contribution (80% threshold)")
    for group_name, features in feature_groups.items():
        # Filter to features that exist in our dataset
        available_features = [f for f in features if f in X_train.columns]

        if not available_features:
            log(f"  Group '{group_name}': No features available")
            continue

        # Get total SHAP for this group
        group_shap_total = sum(shap_dict.get(f, 0) for f in available_features)

        # Select features
        selected = select_features_by_cumulative_shap(available_features, shap_dict, threshold=0.8)
        group_selections[group_name] = selected

        # Calculate what percentage we're capturing
        selected_shap_sum = sum(shap_dict.get(f, 0) for f in selected)
        captured_pct = (selected_shap_sum / group_shap_total) * 100 if group_shap_total > 0 else 0

        log(f"  Group '{group_name}': Selected {len(selected)}/{len(available_features)} features")
        log(f"    Capturing {captured_pct:.1f}% of group SHAP importance")
        for f in selected:
            contribution_pct = (shap_dict.get(f, 0) / group_shap_total) * 100 if group_shap_total > 0 else 0
            log(f"    - {f:30s}: {shap_dict.get(f, 0):.6f} ({contribution_pct:.1f}%)")

        # Add to combined selection
        combined_selection.extend(selected)

    # Step 2: Filter highly correlated features
    log("\nStep 2: Filtering highly correlated features")
    filtered_selection = filter_correlated_features(combined_selection, corr_matrix, shap_dict, threshold=0.8)

    log(f"After correlation filtering: {len(filtered_selection)}/{len(combined_selection)} features remained")
    removed = set(combined_selection) - set(filtered_selection)
    if removed:
        log(f"Removed due to correlation: {sorted(removed)}")

    # Step 3: Greedy feature selection based on performance
    log("\nStep 3: Greedy performance-based selection")

    # Start with empty set and consider all filtered features as candidates
    final_selection = []
    candidates = filtered_selection.copy()

    # Perform greedy selection
    final_selection = greedy_feature_selection(
        X_train, y_train, X_val, y_val,
        candidates, final_selection, spw,
        min_improvement=0.005, max_features=20
    )

    log(f"\nFinal selection: {len(final_selection)} features")
    for i, feat in enumerate(final_selection):
        for group_name, features in feature_groups.items():
            if feat in features:
                log(f"  {i + 1:2d}. {feat:30s} (from {group_name}): SHAP={shap_dict.get(feat, 0):.6f}")
                break
        else:
            log(f"  {i + 1:2d}. {feat:30s} (ungrouped): SHAP={shap_dict.get(feat, 0):.6f}")

    return final_selection


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

    # 5. Define feature groups
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

    # 6. Advanced feature selection
    selected_features = advanced_feature_selection(
        X_train, y_train, X_val, y_val,
        actual_groups, shap_importances, fixed_spw
    )

    # 7. Train final model with selected features
    log("\nTraining final model with selected features")
    final_model = train_lgb(X_train[selected_features], y_train,
                            X_val[selected_features], y_val, spw=fixed_spw)
    final_scores = final_model.predict_proba(X_val[selected_features])[:, 1]
    final_metrics = topk_metrics(y_val, final_scores, [100, 200, 500])

    # 8. Compare baseline vs final metrics
    log("\nPerformance comparison:")
    log(f"Baseline metrics (all features): {base_metrics}")
    log(f"Final metrics (selected features): {final_metrics}")

    for k in [100, 200, 500]:
        baseline = base_metrics[f"recall@{k}"]
        final = final_metrics[f"recall@{k}"]
        diff = final - baseline
        direction = "↑" if diff > 0 else "↓"
        log(f"Recall@{k}: {baseline:.4f} → {final:.4f} ({direction}{abs(diff):.4f})")

    # 9. Train isolation forest on selected features
    log("\nTraining Isolation Forest with selected features")
    iso_scores = isolation_risk(X_val[selected_features])
    iso_metrics = topk_metrics(y_val, iso_scores, [100, 200, 500])
    log(f"Isolation Forest metrics: {iso_metrics}")

    # 10. Try simple ensemble
    log("\nEvaluating ensemble model")
    ensemble_weights = [0.7, 0.3]  # LightGBM, IsoForest
    ensemble_scores = (ensemble_weights[0] * final_scores) + (ensemble_weights[1] * iso_scores)
    ensemble_metrics = topk_metrics(y_val, ensemble_scores, [100, 200, 500])
    log(f"Ensemble metrics (weights={ensemble_weights}): {ensemble_metrics}")

    # 11. Summary
    log(f"\nFinal feature selection summary:")
    log(f"- Started with {X.shape[1]} features")
    log(f"- Selected {len(selected_features)} features ({len(selected_features) / X.shape[1]:.1%} of original)")
    log(f"- Recall@100: {base_metrics['recall@100']:.4f} → {final_metrics['recall@100']:.4f}")
    log(f"- Recall@100 (ensemble): {ensemble_metrics['recall@100']:.4f}")
    log(f"- Selected features: {selected_features}")


if __name__ == "__main__":
    t0 = time.time();
    main();
    log(f"Done in {time.time() - t0:.1f}s")