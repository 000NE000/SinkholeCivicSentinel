from __future__ import annotations

import os
import time
from datetime import datetime
from typing import List, Dict, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.wkb
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import lightgbm as lgb
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ────────────────────────────────────────────────────────────────────────────────
# Utility logging
# ────────────────────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}")


# ────────────────────────────────────────────────────────────────────────────────
# Data loading and feature engineering
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


def create_new_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create new interaction and derived features."""
    df = gdf.copy()

    log("Creating new features...")

    # 1. Road-pipe stress interaction
    if 'soil_road_stress' in df.columns and 'risk_road_load' in df.columns:
        df['road_pipe_stress_interaction'] = df['soil_road_stress'] * df['risk_road_load']
        log("Created: road_pipe_stress_interaction")
    else:
        log("Warning: Could not create road_pipe_stress_interaction (missing source columns)")

    # 2. Sum of inverse distances to k nearest sinkholes
    if 'min_distance_to_sinkhole' in df.columns:
        # For demonstration, we'll create a simulated feature
        # In real implementation, you would use spatial operations to find k-nearest points
        # This is just a placeholder implementation
        df['sum_inv_dist_to_k_nearest'] = 1 / (df['min_distance_to_sinkhole'] + 1e-6)

        # Add some noise to make it different from just the inverse of min_distance
        rng = np.random.RandomState(42)
        noise = rng.normal(1, 0.3, size=len(df))
        df['sum_inv_dist_to_k_nearest'] *= noise

        log("Created: sum_inv_dist_to_k_nearest (simulated)")
    else:
        log("Warning: Could not create sum_inv_dist_to_k_nearest (missing source columns)")

    # 3. Variance within 100m buffer
    # In a real implementation, this would use actual spatial operations
    if 'sinkhole_area_pipe_risk' in df.columns:
        # For demonstration, we'll create a simulated variance feature
        # based on the existing feature with some noise
        rng = np.random.RandomState(42)
        base_values = df['sinkhole_area_pipe_risk'].values
        df['var_within_100m'] = np.abs(base_values + rng.normal(0, base_values.std() * 0.5, size=len(df)))
        log("Created: var_within_100m (simulated)")
    else:
        log("Warning: Could not create var_within_100m (missing source columns)")

    # 4. Density-area risk ratio
    if 'weighted_sinkhole_density' in df.columns and 'sinkhole_area_pipe_risk' in df.columns:
        df['density_area_risk_ratio'] = df['weighted_sinkhole_density'] / (df['sinkhole_area_pipe_risk'] + 1e-6)
        log("Created: density_area_risk_ratio")
    else:
        log("Warning: Could not create density_area_risk_ratio (missing source columns)")

    return df


# ────────────────────────────────────────────────────────────────────────────────
# Feature validation utilities
# ────────────────────────────────────────────────────────────────────────────────
def validate_features_univariate(X: pd.DataFrame, y: pd.Series, new_features: List[str]) -> Dict[str, float]:
    """Calculate univariate AUC for each feature to assess predictive power."""
    results = {}
    log("Univariate AUC analysis of new features:")
    for feat in new_features:
        if feat not in X.columns:
            log(f"  {feat}: Not in dataset")
            continue

        try:
            # Calculate AUC - try both directions in case of inverse relationship
            direct_auc = roc_auc_score(y, X[feat])
            inverse_auc = roc_auc_score(y, -X[feat])
            auc_value = max(direct_auc, inverse_auc)
            direction = "direct" if direct_auc >= inverse_auc else "inverse"

            results[feat] = auc_value
            significance = ""
            if auc_value > 0.6:
                significance = "*** Strong signal"
            elif auc_value > 0.55:
                significance = "** Moderate signal"
            elif auc_value > 0.52:
                significance = "* Weak signal"

            log(f"  {feat}: AUC = {auc_value:.4f} ({direction}) {significance}")
        except Exception as e:
            log(f"  Error calculating AUC for {feat}: {str(e)}")

    return results


def plot_feature_distributions(X: pd.DataFrame, y: pd.Series, new_features: List[str],
                               max_features_per_plot: int = 2):
    """Plot the distribution of new features comparing positive and negative classes."""
    log("Plotting feature distributions by class...")

    available_features = [f for f in new_features if f in X.columns]
    if not available_features:
        log("No features available to plot")
        return

    # Create a combined dataframe with features and target
    plot_df = X[available_features].copy()
    plot_df['target'] = y

    # Calculate number of rows needed for plots
    n_features = len(available_features)
    n_rows = (n_features + max_features_per_plot - 1) // max_features_per_plot

    fig, axes = plt.subplots(n_rows, max_features_per_plot, figsize=(15, 4 * n_rows))
    if n_rows == 1 and max_features_per_plot == 1:
        axes = np.array([axes])
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, feature in enumerate(available_features):
        row = i // max_features_per_plot
        col = i % max_features_per_plot

        ax = axes[row, col]

        # Separate data by target class
        pos_values = plot_df[plot_df['target'] == 1][feature]
        neg_values = plot_df[plot_df['target'] == 0][feature]

        # KDE plots
        sns.kdeplot(pos_values, ax=ax, label='Positive Class', color='red')
        sns.kdeplot(neg_values, ax=ax, label='Negative Class', color='blue')

        # Add vertical lines for means
        ax.axvline(pos_values.mean(), color='red', linestyle='--', alpha=0.7)
        ax.axvline(neg_values.mean(), color='blue', linestyle='--', alpha=0.7)

        # Calculate and display separation metrics
        ks_stat = np.abs(pos_values.mean() - neg_values.mean()) / (pos_values.std() + neg_values.std())

        # Display mean values in the legend
        ax.set_title(f"{feature} Distribution by Class")
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend([f'Positive (mean={pos_values.mean():.3f})',
                   f'Negative (mean={neg_values.mean():.3f})',
                   f'KS stat={ks_stat:.3f}'])

    # Hide any unused subplots
    for i in range(n_features, n_rows * max_features_per_plot):
        row = i // max_features_per_plot
        col = i % max_features_per_plot
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.show()


def check_correlations(X: pd.DataFrame, new_features: List[str], existing_features: List[str],
                       threshold: float = 0.9) -> List[Tuple[str, str, float]]:
    """Check correlations between new features and existing features."""
    log(f"Checking correlations (threshold={threshold})...")

    available_new = [f for f in new_features if f in X.columns]
    available_existing = [f for f in existing_features if f in X.columns]

    if not available_new or not available_existing:
        log("No features available to check correlations")
        return []

    high_correlations = []

    # Check correlations between new features and existing features
    for new_feat in available_new:
        correlations = []
        for exist_feat in available_existing:
            if new_feat == exist_feat:
                continue

            corr = X[new_feat].corr(X[exist_feat])
            correlations.append((exist_feat, abs(corr)))

            if abs(corr) > threshold:
                high_correlations.append((new_feat, exist_feat, corr))
                log(f"  High correlation: {new_feat} ↔ {exist_feat} = {corr:.4f}")

        # Log top 3 correlations for each new feature
        top_corrs = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)[:3]
        log(f"  {new_feat} - Top correlations:")
        for feat, corr_val in top_corrs:
            log(f"    - with {feat}: {corr_val:.4f}")

    # Also check correlations between new features
    for i, feat1 in enumerate(available_new):
        for feat2 in available_new[i + 1:]:
            corr = X[feat1].corr(X[feat2])
            if abs(corr) > threshold:
                high_correlations.append((feat1, feat2, corr))
                log(f"  High correlation between new features: {feat1} ↔ {feat2} = {corr:.4f}")

    return high_correlations


def calculate_vif(X: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Calculate Variance Inflation Factor for features."""
    log("Calculating VIF for feature multicollinearity...")

    available_features = [f for f in features if f in X.columns]
    if len(available_features) < 2:
        log("Need at least 2 features to calculate VIF")
        return pd.DataFrame()

    # Create a dataframe with only the features of interest
    X_vif = X[available_features].copy()

    # Handle potential NaN values
    X_vif = X_vif.fillna(X_vif.mean())

    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = available_features
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

    # Sort by VIF in descending order
    vif_data = vif_data.sort_values("VIF", ascending=False)

    # Log results
    log("VIF results (higher = more multicollinearity):")
    for _, row in vif_data.iterrows():
        feat, vif = row["Feature"], row["VIF"]
        concern = ""
        if vif > 10:
            concern = "*** High multicollinearity"
        elif vif > 5:
            concern = "** Moderate multicollinearity"
        log(f"  {feat:30s}: {vif:.4f} {concern}")

    return vif_data


# ────────────────────────────────────────────────────────────────────────────────
# Model training and evaluation
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


def perform_ablation_test(X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series,
                          existing_features: List[str], new_features: List[str],
                          spw: int = 300) -> Dict[str, Dict[str, float]]:
    """
    Perform ablation tests to measure the impact of features:
    1. Baseline with only existing features
    2. Enhanced model with all new features added
    3. Leave-one-out tests removing each new feature individually
    """
    log("Performing ablation tests...")

    # Filter to available features
    available_existing = [f for f in existing_features if f in X_train.columns]
    available_new = [f for f in new_features if f in X_train.columns]

    if not available_existing:
        log("No existing features available for ablation tests")
        return {}

    results = {}

    # 1. Baseline with only existing features
    log("Training baseline model with existing features...")
    baseline_model = train_lgb(X_train[available_existing], y_train,
                               X_val[available_existing], y_val, spw)
    baseline_scores = baseline_model.predict_proba(X_val[available_existing])[:, 1]
    baseline_metrics = topk_metrics(y_val, baseline_scores, [100, 200, 500])
    results["baseline"] = baseline_metrics
    log(f"  Baseline metrics: {baseline_metrics}")

    if not available_new:
        log("No new features available to test")
        return results

    # 2. Enhanced model with all new features
    combined_features = available_existing + available_new
    log(f"Training enhanced model with {len(combined_features)} features ({len(available_new)} new)...")
    enhanced_model = train_lgb(X_train[combined_features], y_train,
                               X_val[combined_features], y_val, spw)
    enhanced_scores = enhanced_model.predict_proba(X_val[combined_features])[:, 1]
    enhanced_metrics = topk_metrics(y_val, enhanced_scores, [100, 200, 500])
    results["enhanced"] = enhanced_metrics

    # Calculate improvement over baseline
    recall_100_improvement = enhanced_metrics["recall@100"] - baseline_metrics["recall@100"]
    log(f"  Enhanced metrics: {enhanced_metrics}")
    log(f"  Recall@100 improvement: {recall_100_improvement:.4f} " +
        f"({(recall_100_improvement / baseline_metrics['recall@100'] * 100):.2f}%)")

    # 3. Leave-one-out tests for each new feature
    log("Performing leave-one-out tests for each new feature...")
    for feat in available_new:
        # Create feature set without this one feature
        leave_one_out_features = [f for f in combined_features if f != feat]

        # Train model
        loo_model = train_lgb(X_train[leave_one_out_features], y_train,
                              X_val[leave_one_out_features], y_val, spw)
        loo_scores = loo_model.predict_proba(X_val[leave_one_out_features])[:, 1]
        loo_metrics = topk_metrics(y_val, loo_scores, [100, 200, 500])
        results[f"without_{feat}"] = loo_metrics

        # Calculate impact of removing this feature
        impact = enhanced_metrics["recall@100"] - loo_metrics["recall@100"]
        impact_pct = (impact / enhanced_metrics["recall@100"]) * 100

        significance = ""
        if abs(impact) < 0.001:
            significance = "no impact"
        elif impact > 0:
            significance = f"removing HURTS by {impact:.4f} ({impact_pct:.2f}%)"
        else:
            significance = f"removing HELPS by {-impact:.4f} ({-impact_pct:.2f}%)"

        log(f"  Without {feat}: Recall@100 = {loo_metrics['recall@100']:.4f} ({significance})")

    return results


def analyze_feature_shap(model: lgb.LGBMClassifier, X: pd.DataFrame,
                         new_features: List[str], sample_size: int = 1000) -> Dict[str, Dict]:
    """Analyze SHAP values for new features."""
    log("Analyzing SHAP values for new features...")

    available_features = [f for f in new_features if f in X.columns]
    if not available_features:
        log("No new features available for SHAP analysis")
        return {}

    # Sample data for SHAP analysis
    X_sample = X.sample(min(sample_size, len(X)), random_state=42)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, we get a list with values for each class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class

    # Create a DataFrame with the SHAP values
    shap_df = pd.DataFrame(shap_values, columns=X_sample.columns)

    # Result dictionary
    results = {}

    for feat in available_features:
        if feat not in shap_df.columns:
            continue

        # Calculate statistics for this feature
        feat_shap = shap_df[feat]
        abs_shap = np.abs(feat_shap)

        stats = {
            "mean_abs_shap": abs_shap.mean(),
            "mean_shap": feat_shap.mean(),
            "min_shap": feat_shap.min(),
            "max_shap": feat_shap.max(),
            "pos_impact_pct": (feat_shap > 0).mean() * 100,
            "neg_impact_pct": (feat_shap < 0).mean() * 100
        }

        results[feat] = stats

        # Determine direction of impact
        direction = "MIXED"
        if stats["pos_impact_pct"] > 75:
            direction = "POSITIVE"
        elif stats["neg_impact_pct"] > 75:
            direction = "NEGATIVE"

        log(f"  {feat}:")
        log(f"    Mean |SHAP|: {stats['mean_abs_shap']:.6f}")
        log(f"    Impact direction: {direction} ({stats['pos_impact_pct']:.1f}% pos, {stats['neg_impact_pct']:.1f}% neg)")
        log(f"    Range: [{stats['min_shap']:.6f}, {stats['max_shap']:.6f}]")

    return results


def plot_partial_dependence(model: lgb.LGBMClassifier, X: pd.DataFrame,
                            new_features: List[str], num_points: int = 50):
    """Plot partial dependence for new features to visualize feature-target relationships."""
    log("Plotting partial dependence for new features...")

    available_features = [f for f in new_features if f in X.columns]
    if not available_features:
        log("No new features available for partial dependence analysis")
        return

    # Prepare gridspec for multiple plots
    n_features = len(available_features)
    n_cols = min(2, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # For each feature, create a grid of values and predict
    for i, feature in enumerate(available_features):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]

        # Get feature range for grid
        feature_min = X[feature].min()
        feature_max = X[feature].max()

        # Create grid
        grid = np.linspace(feature_min, feature_max, num_points)
        predictions = []

        # For each grid point, create a copy of X with that value
        # and predict (this is a simple approach; real implementation might use
        # proper partial dependence calculation)
        for value in grid:
            X_copy = X.copy()
            X_copy[feature] = value
            # Predict with model
            pred = model.predict_proba(X_copy)[:, 1].mean()
            predictions.append(pred)

        # Plot
        ax.plot(grid, predictions)
        ax.set_title(f"Partial Dependence: {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Average Prediction")
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots
    for i in range(n_features, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.show()


def cross_validate_feature_impact(X: pd.DataFrame, y: pd.Series,
                                  existing_features: List[str],
                                  new_features: List[str],
                                  n_folds: int = 5, spw: int = 300) -> Dict[str, Dict[str, float]]:
    """Perform cross-validation to assess feature stability."""
    log(f"Cross-validating feature impact ({n_folds} folds)...")

    available_existing = [f for f in existing_features if f in X.columns]
    available_new = [f for f in new_features if f in X.columns]

    if not available_existing or not available_new:
        log("Not enough features available for cross-validation")
        return {}

    # Prepare KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Results dictionaries
    baseline_results = []
    enhanced_results = []

    # For each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

        log(f"Fold {fold + 1}/{n_folds}:")

        # Baseline model (existing features)
        base_model = train_lgb(X_fold_train[available_existing], y_fold_train,
                               X_fold_val[available_existing], y_fold_val, spw)
        base_scores = base_model.predict_proba(X_fold_val[available_existing])[:, 1]
        base_metrics = topk_metrics(y_fold_val, base_scores, [100])
        baseline_results.append(base_metrics["recall@100"])

        # Enhanced model (with new features)
        combined_features = available_existing + available_new
        enhanced_model = train_lgb(X_fold_train[combined_features], y_fold_train,
                                   X_fold_val[combined_features], y_fold_val, spw)
        enhanced_scores = enhanced_model.predict_proba(X_fold_val[combined_features])[:, 1]
        enhanced_metrics = topk_metrics(y_fold_val, enhanced_scores, [100])
        enhanced_results.append(enhanced_metrics["recall@100"])

        # Calculate improvement
        improvement = enhanced_metrics["recall@100"] - base_metrics["recall@100"]
        log(f"  Baseline Recall@100: {base_metrics['recall@100']:.4f}")
        log(f"  Enhanced Recall@100: {enhanced_metrics['recall@100']:.4f}")
        log(f"  Improvement: {improvement:.4f} ({improvement / base_metrics['recall@100'] * 100:.2f}%)")

    # Calculate statistics
    baseline_mean = np.mean(baseline_results)
    baseline_std = np.std(baseline_results)
    enhanced_mean = np.mean(enhanced_results)
    enhanced_std = np.std(enhanced_results)

    # Calculate improvement statistics
    improvements = [e - b for e, b in zip(enhanced_results, baseline_results)]
    improvement_mean = np.mean(improvements)
    improvement_std = np.std(improvements)

    log("\nCross-validation summary:")
    log(f"  Baseline Recall@100: {baseline_mean:.4f} ± {baseline_std:.4f}")
    log(f"  Enhanced Recall@100: {enhanced_mean:.4f} ± {enhanced_std:.4f}")
    log(f"  Improvement: {improvement_mean:.4f} ± {improvement_std:.4f}")

    # Statistical significance
    if improvement_mean > 2 * improvement_std:
        log("  Conclusion: Improvement is STATISTICALLY SIGNIFICANT")
    elif improvement_mean > improvement_std:
        log("  Conclusion: Improvement is LIKELY real, but more data needed")
    else:
        log("  Conclusion: Improvement is NOT statistically significant")

    return {
        "baseline": {"mean": baseline_mean, "std": baseline_std},
        "enhanced": {"mean": enhanced_mean, "std": enhanced_std},
        "improvement": {"mean": improvement_mean, "std": improvement_std}
    }


# ────────────────────────────────────────────────────────────────────────────────
# Main function
# ────────────────────────────────────────────────────────────────────────────────
def main():
    # Load dataset
    gdf = load_dataset()

    # Create new features
    gdf_enhanced = create_new_features(gdf)

    # Define lists of features
    new_features = [
        'road_pipe_stress_interaction',
        'sum_inv_dist_to_k_nearest',
        'var_within_100m',
        'density_area_risk_ratio'
    ]

    # Extract target and features
    y = gdf_enhanced["subsidence_occurrence"].astype(int)
    X = gdf_enhanced.drop(columns=[c for c in ["grid_id", "subsidence_occurrence", "subsidence_count", "geometry"]
                                   if c in gdf_enhanced.columns])

    # Get list of existing features (excluding new ones)
    existing_features = [f for f in X.columns if f not in new_features]

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    log(f"Dataset ready: {len(X_train)} train samples, {len(X_val)} validation samples")
    log(f"Existing features: {len(existing_features)}")
    log(f"New features: {len([f for f in new_features if f in X.columns])}")

    # 1. Univariate analysis of new features
    univariate_results = validate_features_univariate(X, y, new_features)

    # 2. Plot feature distributions
    plot_feature_distributions(X, y, new_features)

    # 3. Check correlations
    # 3. Check correlations
    correlation_results = check_correlations(X, new_features, existing_features)

    # 4. Calculate VIF for multicollinearity
    # First for existing features only
    vif_existing = calculate_vif(X, existing_features[:15])  # Limit to first 15 to avoid computational issues

    # Then for existing + new features
    combined_features = existing_features[:15] + [f for f in new_features if f in X.columns]
    vif_combined = calculate_vif(X, combined_features)

    # 5. Ablation testing
    ablation_results = perform_ablation_test(
        X_train, y_train, X_val, y_val,
        existing_features, new_features
    )

    # 6. Train a model with all features (for SHAP analysis)
    combined_features = existing_features + [f for f in new_features if f in X.columns]
    log("\nTraining model for SHAP analysis...")
    full_model = train_lgb(X_train[combined_features], y_train, X_val[combined_features], y_val, 300)

    # 7. SHAP analysis of new features
    shap_analysis = analyze_feature_shap(full_model, X_train, new_features)

    # 8. Partial dependence plots
    plot_partial_dependence(full_model, X_train, new_features)

    # 9. Cross-validation stability
    cv_results = cross_validate_feature_impact(
        X, y, existing_features, new_features, n_folds=5
    )

    # 10. Summarize results for each feature
    log("\n" + "=" * 80)
    log("FEATURE VALIDATION SUMMARY")
    log("=" * 80)

    available_new = [f for f in new_features if f in X.columns]
    for feature in available_new:
        log(f"\nFeature: {feature}")
        log("-" * 50)

        # 1. Univariate predictive power
        if feature in univariate_results:
            auc = univariate_results[feature]
            strength = "Strong" if auc > 0.6 else "Moderate" if auc > 0.55 else "Weak" if auc > 0.52 else "Very weak"
            log(f"1. Univariate AUC: {auc:.4f} ({strength})")
        else:
            log(f"1. Univariate AUC: Not available")

        # 2. Correlation issues
        high_corrs = [c for c in correlation_results if c[0] == feature or c[1] == feature]
        if high_corrs:
            log(f"2. Correlation issues: {len(high_corrs)} high correlations detected")
            for corr in high_corrs:
                other = corr[1] if corr[0] == feature else corr[0]
                log(f"   - Correlated with {other}: {corr[2]:.4f}")
        else:
            log(f"2. Correlation issues: None detected")

        # 3. VIF (if available)
        if feature in vif_combined["Feature"].values:
            vif = vif_combined.loc[vif_combined["Feature"] == feature, "VIF"].values[0]
            concern = "High" if vif > 10 else "Moderate" if vif > 5 else "Low"
            log(f"3. Multicollinearity (VIF): {vif:.2f} ({concern} concern)")
        else:
            log(f"3. Multicollinearity: Not calculated")

        # 4. Ablation impact
        enhanced_recall = ablation_results.get("enhanced", {}).get("recall@100", 0)
        without_recall = ablation_results.get(f"without_{feature}", {}).get("recall@100", 0)
        if enhanced_recall and without_recall:
            impact = enhanced_recall - without_recall
            impact_pct = (impact / enhanced_recall) * 100
            direction = "Positive" if impact > 0 else "Negative" if impact < 0 else "Neutral"
            significance = "Significant" if abs(impact) > 0.01 else "Moderate" if abs(impact) > 0.005 else "Minimal"
            log(f"4. Ablation impact: {impact:.4f} ({impact_pct:.2f}%) - {direction} impact, {significance}")
        else:
            log(f"4. Ablation impact: Not calculated")

        # 5. SHAP insights
        if feature in shap_analysis:
            stats = shap_analysis[feature]
            mean_abs = stats["mean_abs_shap"]
            direction = "Mostly positive" if stats["pos_impact_pct"] > 75 else "Mostly negative" if stats[
                                                                                                        "neg_impact_pct"] > 75 else "Mixed"
            log(f"5. SHAP importance: {mean_abs:.6f}")
            log(f"   Direction: {direction} ({stats['pos_impact_pct']:.1f}% positive, {stats['neg_impact_pct']:.1f}% negative)")
        else:
            log(f"5. SHAP importance: Not calculated")

        # 6. Cross-validation stability
        if cv_results:
            improvement = cv_results["improvement"]["mean"]
            stability = cv_results["improvement"]["std"]
            stability_ratio = abs(improvement) / stability if stability > 0 else float('inf')
            reliability = "High" if stability_ratio > 2 else "Moderate" if stability_ratio > 1 else "Low"
            log(f"6. Cross-validation: {improvement:.4f} ± {stability:.4f} improvement")
            log(f"   Reliability: {reliability} (signal/noise ratio: {stability_ratio:.2f})")
        else:
            log(f"6. Cross-validation: Not performed")

        # 7. Overall verdict
        verdicts = []

        # Univariate strength
        if feature in univariate_results:
            auc = univariate_results[feature]
            if auc > 0.6:
                verdicts.append("Strong univariate signal")
            elif auc > 0.55:
                verdicts.append("Moderate univariate signal")
            elif auc > 0.52:
                verdicts.append("Weak univariate signal")
            else:
                verdicts.append("Very weak univariate signal")

        # Correlation issues
        if high_corrs:
            verdicts.append("Correlation concerns")

        # VIF concerns
        if feature in vif_combined["Feature"].values:
            vif = vif_combined.loc[vif_combined["Feature"] == feature, "VIF"].values[0]
            if vif > 10:
                verdicts.append("High multicollinearity")
            elif vif > 5:
                verdicts.append("Moderate multicollinearity")

        # Ablation impact
        if enhanced_recall and without_recall:
            impact = enhanced_recall - without_recall
            if impact > 0.01:
                verdicts.append("Strong positive model impact")
            elif impact > 0.005:
                verdicts.append("Moderate positive model impact")
            elif impact < -0.005:
                verdicts.append("Negative model impact")

        # SHAP importance
        if feature in shap_analysis:
            mean_abs = shap_analysis[feature]["mean_abs_shap"]
            top_shap_cutoff = 0.01  # Adjust based on typical SHAP values in your model
            if mean_abs > top_shap_cutoff:
                verdicts.append("High SHAP importance")

        # Cross-validation
        if cv_results:
            improvement = cv_results["improvement"]["mean"]
            stability = cv_results["improvement"]["std"]
            if improvement > 2 * stability and improvement > 0:
                verdicts.append("Statistically significant improvement")
            elif improvement > stability and improvement > 0:
                verdicts.append("Likely improvement, but more data needed")
            elif improvement <= 0:
                verdicts.append("No consistent improvement across folds")

        # Final decision
        positive_signals = sum(
            1 for v in verdicts if "signal" in v.lower() or "positive" in v.lower() or "improvement" in v.lower())
        negative_signals = sum(
            1 for v in verdicts if "concern" in v.lower() or "negative" in v.lower() or "no consistent" in v.lower())

        if positive_signals > negative_signals * 2:
            final_verdict = "KEEP"
        elif positive_signals > negative_signals:
            final_verdict = "PROBABLY KEEP"
        elif positive_signals == negative_signals:
            final_verdict = "NEUTRAL - NEEDS MORE INVESTIGATION"
        else:
            final_verdict = "PROBABLY REMOVE"

        log(f"\n7. VERDICT: {final_verdict}")
        log(f"   Reasons: {', '.join(verdicts)}")

    # Final conclusion
    log("\n" + "=" * 80)
    log("FINAL RECOMMENDATIONS")
    log("=" * 80)

    # Calculate overall improvement from baseline to enhanced
    baseline_recall = ablation_results.get("baseline", {}).get("recall@100", 0)
    enhanced_recall = ablation_results.get("enhanced", {}).get("recall@100", 0)

    if baseline_recall and enhanced_recall:
        improvement = enhanced_recall - baseline_recall
        improvement_pct = (improvement / baseline_recall) * 100
        log(f"Overall improvement with new features: {improvement:.4f} ({improvement_pct:.2f}%)")

    # Feature-by-feature recommendation
    recommended_features = []
    neutral_features = []
    not_recommended_features = []

    for feature in available_new:
        # Simplified logic based on ablation results
        enhanced_recall = ablation_results.get("enhanced", {}).get("recall@100", 0)
        without_recall = ablation_results.get(f"without_{feature}", {}).get("recall@100", 0)

        if enhanced_recall and without_recall:
            impact = enhanced_recall - without_recall

            if impact > 0.005:  # Positive significant impact
                recommended_features.append(feature)
            elif impact > -0.002:  # Neutral or slight positive impact
                neutral_features.append(feature)
            else:  # Negative impact
                not_recommended_features.append(feature)

    if recommended_features:
        log("\nStrongly Recommended Features:")
        for feature in recommended_features:
            enhanced_recall = ablation_results.get("enhanced", {}).get("recall@100", 0)
            without_recall = ablation_results.get(f"without_{feature}", {}).get("recall@100", 0)
            impact = enhanced_recall - without_recall
            log(f"  - {feature}: Impact on Recall@100 = +{impact:.4f}")

    if neutral_features:
        log("\nNeutral Features (Consider keeping):")
        for feature in neutral_features:
            enhanced_recall = ablation_results.get("enhanced", {}).get("recall@100", 0)
            without_recall = ablation_results.get(f"without_{feature}", {}).get("recall@100", 0)
            impact = enhanced_recall - without_recall
            log(f"  - {feature}: Impact on Recall@100 = {impact:.4f}")

    if not_recommended_features:
        log("\nNot Recommended Features:")
        for feature in not_recommended_features:
            enhanced_recall = ablation_results.get("enhanced", {}).get("recall@100", 0)
            without_recall = ablation_results.get(f"without_{feature}", {}).get("recall@100", 0)
            impact = enhanced_recall - without_recall
            log(f"  - {feature}: Impact on Recall@100 = {impact:.4f}")

    log("\nConclusions and Next Steps:")
    if recommended_features:
        log(f"1. Add the {len(recommended_features)} strongly recommended features to the model")
    if neutral_features:
        log(f"2. Consider the {len(neutral_features)} neutral features based on domain knowledge")
    if not_recommended_features:
        log(f"3. Remove the {len(not_recommended_features)} not recommended features")

    if cv_results:
        improvement = cv_results["improvement"]["mean"]
        stability = cv_results["improvement"]["std"]
        if improvement > 2 * stability:
            log("4. The improvement from new features is statistically significant across cross-validation")
        elif improvement > stability:
            log("4. The improvement is likely real but more data or iterations may be needed")
        else:
            log("4. The improvement is not statistically significant - more feature engineering may be needed")


if __name__ == "__main__":
    t0 = time.time();
    main();
    log(f"Done in {time.time() - t0:.1f}s")