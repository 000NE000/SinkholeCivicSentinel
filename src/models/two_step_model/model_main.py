from __future__ import annotations

import os
import io
import time
import random
from datetime import datetime
from typing import List, Tuple, Dict, Sequence, Optional
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.wkb
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc, average_precision_score)
from sklearn.model_selection import train_test_split, GroupKFold, ParameterGrid
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
import lightgbm as lgb
VERBOSE = False

# ────────────────────────────────────────────────────────────────────────────────
# 1. Utility
# ────────────────────────────────────────────────────────────────────────────────
def log(msg: str, level: int = 1) -> None:
    """
    level=1 : fold summary, final summary
    level=2 : detailed (load_dataset, each fold entry, etc)
    """
    if level == 1 or (level == 2 and VERBOSE):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ────────────────────────────────────────────────────────────────────────────────
# 2. Data loading
# ────────────────────────────────────────────────────────────────────────────────

def load_dataset(table: str = "feature_matrix_25_geo",
                 chunksize: int = 100_000) -> gpd.GeoDataFrame:
    """Stream the entire table, ensure geometry, return GeoDataFrame."""
    load_dotenv()
    dsn = os.getenv("DB_DSN")
    if not dsn:
        raise RuntimeError("Environment variable DB_DSN is not set.")
    engine = create_engine(dsn)

    gdf_parts: list[pd.DataFrame] = []
    sql = f"SELECT * FROM {table}"
    for chunk in pd.read_sql_query(sql, engine, chunksize=chunksize):
        if "geom" not in chunk.columns:
            raise ValueError("Column 'geom' missing – confirm PostGIS WKB hex present.")
        chunk["geometry"] = chunk["geom"].apply(lambda h: shapely.wkb.loads(bytes.fromhex(h)))
        chunk = chunk.drop(columns=["geom"])
        gdf_parts.append(chunk)
    gdf = gpd.GeoDataFrame(pd.concat(gdf_parts, ignore_index=True), geometry="geometry")
    if "subsidence_occurrence" not in gdf.columns:
        raise ValueError("Required label column 'subsidence_occurrence' missing.")
    pos = int(gdf["subsidence_occurrence"].sum())
    log(f"Loaded {len(gdf):,} rows  •  positives: {pos} ({pos / len(gdf):.4%})", level=2)
    return gdf
#==================




# ────────────────────────────────────────────────────────────────────────────────
# 3. Metrics
# ────────────────────────────────────────────────────────────────────────────────

def evaluate_reports_impact_pu(
        report_tables: List[str],
        k_vals: List[int] = [100, 200, 500],
        n_iters: int = 5,  # 반복 횟수
        random_state: int = 42
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    가상의 레이블 제약 상황(Positive-Unlabeled)에서
    report 수별 성능 변화를 시뮬레이션 평가
    """
    rng = np.random.default_rng(random_state)
    agg_results = {}
    silent_ids = load_silent_grid_ids("survey_grid_strict")
    log(f"Starting PU-learning simulation across {len(report_tables)} report counts", level=1)
    for table in report_tables:
        report_count = int(''.join(filter(str.isdigit, table)))
        log(f"Simulating with n_reports={report_count}...", level=1)

        # 원본 데이터 & 진짜 레이블
        gdf = load_dataset(table=table)
        X_full = gdf.drop(columns=['grid_id', 'geometry', 'subsidence_occurrence', 'subsidence_count'])
        y_full = gdf['subsidence_occurrence'].astype(int).values
        pos_idx = np.where(y_full == 1)[0]

        n_pos_total = len(pos_idx)

        # 반복마다 metrics 수집
        metrics_accum = {k: [] for k in k_vals}
        for it in range(n_iters):
            # 1) 랜덤으로 report_count 개만 positive 라벨로 남김
            sampled = rng.choice(pos_idx, size=min(report_count, n_pos_total), replace=False)
            y_pu = np.zeros_like(y_full)
            y_pu[sampled] = 1

            # 2) Spatial CV로 평가
            gdf_pu = gdf.copy()
            gdf_pu['subsidence_occurrence'] = y_pu
            cv_res = evaluate_spatial_cv(gdf_pu, n_folds=5, k_vals=k_vals, silent_grid_ids=silent_ids)

            # Changed part - collect multiple metrics
            for k in k_vals:
                for m in ['recall', 'precision', 'lift']:
                    key = f'{m}@{k}'
                    if key not in metrics_accum:
                        metrics_accum[key] = []
                    metrics_accum[key].append(cv_res['aggregate'][key]['mean'])
            # PR-AUC also stored separately
            metrics_accum.setdefault('pr_auc', []).append(
                cv_res['aggregate']['pr_auc']['mean'])

        # 평균·표준편차 계산
        summary = {}
        # Changed part - process all metrics
        for k in k_vals:
            for m in ['recall', 'precision', 'lift']:
                arr = np.array(metrics_accum[f'{m}@{k}'])
                summary[f'{m}@{k}_mean'] = arr.mean()
                summary[f'{m}@{k}_std'] = arr.std()
        # PR-AUC
        pa = np.array(metrics_accum['pr_auc'])
        summary['pr_auc_mean'] = pa.mean()
        summary['pr_auc_std'] = pa.std()

        agg_results[report_count] = summary
        log(f"n_reports={report_count} → " +
            ", ".join([f"recall@{k}: {summary[f'recall@{k}_mean']:.3f}±{summary[f'recall@{k}_std']:.3f}"
                       for k in k_vals]))
    return agg_results


def plot_pu_report_impact(results: Dict, k_vals: List[int] = [100, 200, 500]) -> None:
    """
    Plot how metrics improve with increasing number of reports in PU-learning simulation
    """
    report_counts = sorted(results.keys())

    fig, axs = plt.subplots(len(k_vals), 1, figsize=(10, 3 * len(k_vals)), sharex=True)
    if len(k_vals) == 1:
        axs = [axs]  # Make it iterable when only one k value

    for i, k in enumerate(k_vals):
        for m, ls in zip(['recall', 'precision', 'lift'],
                        ['-', '--', '-.']):
            means = [results[count][f'{m}@{k}_mean'] for count in report_counts]
            stds = [results[count][f'{m}@{k}_std'] for count in report_counts]

            axs[i].errorbar(report_counts, means, yerr=stds,
                           linestyle=ls, marker='o', linewidth=2,
                           capsize=5, label=m.capitalize())
        axs[i].set_ylabel(f'Metrics@{k}')
        axs[i].grid(True, linestyle='--', alpha=0.7)
        axs[i].set_title(f'PU Simulation – @{k}')
        axs[i].legend()

    axs[-1].set_xlabel('Number of Reports')
    plt.tight_layout()

    # Save plot
    plt.savefig('sinkhole_pu_report_impact.png', dpi=300)
    log("PU simulation plot saved as 'sinkhole_pu_report_impact.png'")

    # Also create a combined view of all k values
    plt.figure(figsize=(10, 6))

    for k in k_vals:
        means = [results[count][f'recall@{k}_mean'] for count in report_counts]
        stds = [results[count][f'recall@{k}_std'] for count in report_counts]

        plt.errorbar(report_counts, means, yerr=stds, marker='o', linestyle='-',
                     linewidth=2, capsize=5, label=f'Recall@{k}')

    plt.xlabel('Number of Reports')
    plt.ylabel('Recall Value')
    plt.title('PU Learning: Performance Improvement with Increasing Reports')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Save combined plot
    plt.savefig('sinkhole_pu_combined_metrics.png', dpi=300)
    log("Combined PU metrics plot saved as 'sinkhole_pu_combined_metrics.png'")

def topk_metrics(y: Sequence[int], scores: Sequence[float], k_list: List[int]) -> Dict[str, float]:
    """Compute recall, precision and lift at given K values."""
    y_arr = np.asarray(y)
    scores_arr = np.asarray(scores)
    order = np.argsort(scores_arr)[::-1]
    y_sorted = y_arr[order]

    total_pos = y_arr.sum()
    overall_rate = total_pos / len(y_arr)

    out: dict[str, float] = {}
    for k in k_list:
        if k > len(y_arr):
            continue
        topk = y_sorted[:k].sum()
        recall_k = topk / total_pos if total_pos else 0
        prec_k = topk / k
        lift_k = prec_k / overall_rate if overall_rate else 0
        out[f"recall@{k}"] = recall_k
        out[f"precision@{k}"] = prec_k
        out[f"lift@{k}"] = lift_k
    return out


def calculate_iou(y_true: np.ndarray, scores: np.ndarray, top_k: int) -> float:
    """
    Calculate Intersection over Union for top-K predictions vs ground truth
    """
    order = np.argsort(scores)[::-1]
    top_pred_indices = set(order[:top_k])
    true_indices = set(np.where(y_true == 1)[0])

    intersection = len(top_pred_indices.intersection(true_indices))
    union = len(top_pred_indices.union(true_indices))

    return intersection / union if union > 0 else 0.0


def compute_comprehensive_metrics(y_true: np.ndarray, scores: np.ndarray, k_list: List[int]) -> Dict[str, float]:
    """Compute comprehensive set of metrics including ROC-AUC, PR-AUC, and TopK"""
    # ROC-AUC
    roc_auc = roc_auc_score(y_true, scores)

    # PR-AUC
    pr_auc = average_precision_score(y_true, scores)

    # IoU
    iou_100 = calculate_iou(y_true, scores, 100)

    # TopK metrics
    topk = topk_metrics(y_true, scores, k_list)

    # Combine all metrics
    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "iou@100": iou_100
    }

    # Add the topk metrics to our dictionary
    metrics.update(topk)

    return metrics


# ────────────────────────────────────────────────────────────────────────────────
# 4. Spatial CV
# ────────────────────────────────────────────────────────────────────────────────

def create_spatial_blocks(gdf: gpd.GeoDataFrame, n_blocks: int = 5) -> np.ndarray:
    """
    Create spatial blocks for cross-validation based on coordinates
    Returns array of block indices for each row in gdf
    """
    # Extract centroids if geometry column exists
    if 'geometry' in gdf.columns:
        X = np.array([(geom.centroid.x, geom.centroid.y) for geom in gdf.geometry])
    else:
        raise ValueError("GeoDataFrame must have 'geometry' column")

    # Apply KMeans clustering to create spatial blocks
    kmeans = KMeans(n_clusters=n_blocks, random_state=42)
    blocks = kmeans.fit_predict(X)

    log(f"Created {n_blocks} spatial blocks with sizes: {pd.Series(blocks).value_counts().to_dict()}")

    return blocks


# ────────────────────────────────────────────────────────────────────────────────
# 5. Models & Focal Loss
# ────────────────────────────────────────────────────────────────────────────────

# Focal Loss implementation for LightGBM
def focal_loss_lgb(y_pred, dtrain, gamma=2.0, alpha=0.25):
    """Focal Loss for LightGBM custom objective function"""
    y_true = dtrain.label
    # Convert prediction to probabilities
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))

    # Calculate the focal loss
    pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
    loss = -alpha_t * (1 - pt) ** gamma * np.log(pt + 1e-7)

    # Calculate the gradient and hessian
    grad = alpha_t * ((1 - pt) ** gamma) * (gamma * pt * np.log(pt + 1e-7) + pt - 1)
    hess = alpha_t * (1 - pt) ** gamma * pt * (1 - pt) * (1 + gamma * np.log(pt + 1e-7))

    return grad, hess


# Function to evaluate focal loss (for monitoring)
def eval_focal_loss(y_pred, dtrain, gamma=2.0, alpha=0.25):
    """Evaluation metric for focal loss"""
    y_true = dtrain.get_label()
    # Convert prediction to probabilities
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))

    # Calculate the focal loss
    pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
    loss = -alpha_t * (1 - pt) ** gamma * np.log(pt + 1e-7)

    return "focal_loss", np.mean(loss), False



def fit(self, X: pd.DataFrame, y: pd.Series, spatial_blocks: np.ndarray = None) -> None:
    """Fit both Stage 1 and Stage 2 models with spatial CV and iterative threshold relaxation."""
    # 1. Auto-tune threshold percentile if not provided
    if self.threshold_percentile is None:
        self.threshold_percentile = self._tune_threshold_percentile(X, y, spatial_blocks)

    # 2. Split into train/val
    if spatial_blocks is not None:
        gkf = GroupKFold(n_splits=min(5, len(np.unique(spatial_blocks))))
        train_idx, val_idx = next(gkf.split(X, y, groups=spatial_blocks))
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        log(f"Using spatial CV: {len(X_train)} train, {len(X_val)} validation samples")
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    # 3. Train Stage 1 (proximity-only)
    log("Training Stage 1 model...")
    self.stage1_model, train_s1_scores, val_s1_scores, self.stage1_metrics = train_stage1_model(
        X_train, y_train, X_val, y_val,
        proximity_feat=self.proximity_feat,
        model_type=self.stage1_model_type
    )

    # 4. Compute threshold
    self.threshold_value = np.percentile(train_s1_scores, self.threshold_percentile)
    log(f"Stage 1 threshold (@{self.threshold_percentile}th pct): {self.threshold_value:.4f}")

    # 5. Build Stage 2 masks (attempt hidden‐risk = low‐confidence)
    train_mask = train_s1_scores < self.threshold_value
    val_mask = val_s1_scores < self.threshold_value

    train_pos = int(y_train[train_mask].sum())
    val_pos = int(y_val[val_mask].sum())
    if train_mask.sum() == 0 or train_pos == 0 or val_pos == 0:
        log("Insufficient hidden‐risk samples → switching to high‐risk branch")
        train_mask = train_s1_scores >= self.threshold_value
        val_mask = val_s1_scores >= self.threshold_value

    # 6. If no positives in hidden-risk, switch to high‐risk branch
    if y_train[train_mask].sum() == 0 or y_val[val_mask].sum() == 0:
        log("No positives in low-confidence set → switching to high-confidence (top-risk) mask")
        train_mask = train_s1_scores >= self.threshold_value
        val_mask = val_s1_scores >= self.threshold_value

    # 7. If still zero positives, relax threshold downward until we find some
    min_pos = 5
    if y_train[train_mask].sum() < min_pos or y_val[val_mask].sum() < min_pos:
        for pct in range(self.threshold_percentile - 5, 0, -5):
            thr = np.percentile(train_s1_scores, pct)
            mask = train_s1_scores >= thr  # now always use high-risk logic
            if y_train[mask].sum() >= min_pos:
                self.threshold_percentile = pct
                self.threshold_value = thr
                train_mask = mask
                val_mask = val_s1_scores >= thr
                log(f"Relaxed high‐risk threshold to {pct}th pct → thr={thr:.4f}, pos={y_train[mask].sum()}")
                break

    # 9. Train Stage 2
    log("Training Stage 2 model...")
    self.stage2_model, self.stage2_metrics = train_stage2_model(
        X_train[train_mask], y_train[train_mask],
        X_val[val_mask], y_val[val_mask],
        exclude_features=self.observed_sinkhole_features,
        use_focal_loss=self.use_focal_loss,
        feature_fraction=self.feature_fraction
    )

    # 10. Final evaluation
    final_scores = self.predict(X_val)
    self.final_metrics = compute_comprehensive_metrics(y_val, final_scores, [100, 200, 500])
    log(f"Final validation metrics: {self.final_metrics}")

    # 11. Optionally log feature importance
    if self.stage2_model is None:
        log("Stage-2 skipped → no feature-importance to display.")
    else:
        feats = [f for f in X_val.columns if f not in self.observed_sinkhole_features]
        imp = getattr(self.stage2_model, "feature_importances_", None) or \
              getattr(self.stage2_model, "feature_importance.py", None)
        if imp is not None:
            top = (
                pd.DataFrame({"feature": feats, "importance": imp})
                .nlargest(10, "importance")["feature"]
                .tolist()
            )
            log(f"Top Stage 2 features: {top}")


# ────────────────────────────────────────────────────────────────────────────────
# 6. Two-Stage Sinkhole Screening Pipeline with Auto-Tuning
# ────────────────────────────────────────────────────────────────────────────────



# ────────────────────────────────────────────────────────────────────────────────
# 1. Utility
# ────────────────────────────────────────────────────────────────────────────────
def log(msg: str, level: int = 1) -> None:
    """
    level=1 : fold summary, final summary
    level=2 : detailed (load_dataset, each fold entry, etc)
    """
    if level == 1 or (level == 2 and VERBOSE):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

class TwoStageSinkholeScreener:
    """
    Two-stage sinkhole risk screening model with unsupervised fallback

    Stage 1: Use only historical location rule (proximity)
    Stage 2: Use supervised if enough labels, otherwise use unsupervised anomaly detection
    """

    def __init__(self,
                 proximity_feat: str = "min_distance_to_sinkhole",
                 stage1_model_type: str = "lgbm",
                 threshold_percentile: float = None,
                 use_focal_loss: bool = True,
                 feature_fraction: float = 0.6,
                 ensemble_weight: float = 0.5):  # Weight of stage1 vs stage2
        self.proximity_feat = proximity_feat
        self.stage1_model_type = stage1_model_type
        self.threshold_percentile = threshold_percentile
        self.use_focal_loss = use_focal_loss
        self.feature_fraction = feature_fraction
        self.ensemble_weight = ensemble_weight  # α in final_score = α*s1 + (1-α)*s2

        self.stage1_model = None
        self.stage2_model = None
        self.is_unsupervised = False
        self.threshold_value = None

        self.observed_sinkhole_features = [
            "min_distance_to_sinkhole",
            "weighted_sinkhole_density"
        ]

        self.stage1_metrics = {}
        self.stage2_metrics = {}
        self.final_metrics = {}

    def _tune_threshold_percentile(self, X: pd.DataFrame, y: pd.Series,
                                   spatial_blocks: np.ndarray = None) -> float:
        """
        Tune the threshold percentile using cross-validation
        """
        log("Tuning threshold percentile...")

        # Candidate percentiles to try
        percentiles = [70, 75, 80, 85, 90]
        results = {}

        # Set up cross-validation
        if spatial_blocks is not None:
            cv = GroupKFold(n_splits=min(5, len(np.unique(spatial_blocks))))
            splits = list(cv.split(X, y, groups=spatial_blocks))
        else:
            # Fallback to standard stratified split if no blocks provided
            train_idx, val_idx = train_test_split(
                np.arange(len(X)), test_size=0.2, random_state=42, stratify=y
            )
            splits = [(train_idx, val_idx)]

        for percentile in percentiles:
            fold_ious = []

            for fold, (train_idx, val_idx) in enumerate(splits):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Train stage 1 model
                _, train_s1_scores, val_s1_scores, _ = train_stage1_model(
                    X_train, y_train, X_val, y_val,
                    proximity_feat=self.proximity_feat,
                    model_type=self.stage1_model_type
                )

                # Calculate threshold
                threshold = np.percentile(train_s1_scores, percentile)

                # Create stage 2 training mask
                train_mask = train_s1_scores < threshold
                val_mask = val_s1_scores < threshold

                # Skip if too few positives in validation
                train_pos = y_train[train_mask].sum()
                val_pos = y_val[val_mask].sum()
                if train_mask.sum() == 0 or val_mask.sum() == 0 or train_pos < 5 or val_pos < 5:
                    continue

                # Safe division to avoid warnings
                train_pos_pct = train_pos / train_mask.sum() if train_mask.sum() > 0 else 0
                val_pos_pct = val_pos / val_mask.sum() if val_mask.sum() > 0 else 0

                log(f"Percentile {percentile}: Stage 2 train: {train_mask.sum()} samples, {train_pos} pos ({train_pos_pct:.4%})")
                log(f"Percentile {percentile}: Stage 2 val: {val_mask.sum()} samples, {val_pos} pos ({val_pos_pct:.4%})")

                # Train stage 2 model
                stage2_model, _, _ = train_stage2_model(
                    X_train[train_mask], y_train[train_mask],
                    X_val[val_mask], y_val[val_mask],
                    exclude_features=self.observed_sinkhole_features,
                    use_focal_loss=self.use_focal_loss,
                    feature_fraction=self.feature_fraction
                )

                # Generate final scores
                final_scores = val_s1_scores.copy()
                if val_mask.sum() > 0:
                    X_val_filtered = X_val.loc[val_mask]
                    features = [f for f in X_val.columns if f not in self.observed_sinkhole_features]
                    stage2_scores = stage2_model.predict_proba(X_val_filtered[features])[:, 1]
                    combined_scores = stage2_scores * (1 - val_s1_scores[val_mask])
                    final_scores[val_mask] = combined_scores

                # Calculate IoU metric
                iou = calculate_iou(y_val.values, final_scores, 100)
                fold_ious.append(iou)
                log(f"Percentile {percentile}, Fold {fold + 1}: IoU@100 = {iou:.4f}")

            # Average IoU across folds
            if fold_ious:
                mean_iou = np.mean(fold_ious)
                std_iou = np.std(fold_ious)
                results[percentile] = (mean_iou, std_iou)
                log(f"Percentile {percentile}: IoU@100 = {mean_iou:.4f} ± {std_iou:.4f}")

        # Select best percentile
        if results:
            best_percentile = max(results.keys(), key=lambda p: results[p][0])
            log(f"Best threshold percentile: {best_percentile} with IoU = {results[best_percentile][0]:.4f}")
            return best_percentile
        else:
            log("Warning: Could not tune percentile, using default 80")
            return 80.0

    def fit_with_silent_zones(self, X: pd.DataFrame, y: pd.Series, silent_grid_ids: set[int],
                              spatial_blocks: np.ndarray = None) -> None:
        """
        Enhanced fitting that incorporates silent zone information
        as weak positive labels using nnPU-style approach.

        Args:
            X: Feature matrix
            y: Labels
            silent_grid_ids: Set of grid IDs for silent zones
            spatial_blocks: Optional spatial CV blocks
        """
        # First perform normal fitting
        self.fit(X, y, spatial_blocks)

        # Get silent mask
        if 'grid_id' not in X.columns:
            log("Cannot perform silent zone enhancement - grid_id column missing", level=1)
            return

        silent_mask = X['grid_id'].isin(silent_grid_ids).values

        if silent_mask.sum() == 0:
            log("No silent zones found in data", level=1)
            return

        log(f"Enhancing model with {silent_mask.sum()} silent zones as weak positives", level=1)

        # Split data if needed
        if spatial_blocks is not None:
            gkf = GroupKFold(n_splits=min(5, len(np.unique(spatial_blocks))))
            train_idx, val_idx = next(gkf.split(X, y, groups=spatial_blocks))
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            silent_train = silent_mask[train_idx]
        else:
            # Simple split
            X_train, X_val, y_train, y_val, silent_train = train_test_split(
                X, y, silent_mask, test_size=0.2, random_state=42, stratify=y
            )

        # Features excluding observed sinkhole features
        features = [c for c in X_train.columns if c not in self.observed_sinkhole_features
                    and c != 'grid_id']

        # Apply pseudo-labeling with silent zones as weak positives
        base_model = lgb.LGBMClassifier(**LGB_PARAMS_BASE)
        enhanced_model = pseudo_labeling_nnpu(
            X_train[features], y_train,
            X_val[features], silent_train,
            base_model=base_model,
            unlabeled_weight=0.5
        )

        # Replace Stage 2 model with the enhanced model
        self.stage2_model = enhanced_model
        self.is_unsupervised = False

        # Optimize threshold and alpha
        stage1_scores = self.stage1_model.predict_proba(X_val[[self.proximity_feat]])[:, 1]
        silent_val = silent_mask[val_idx] if spatial_blocks is not None else silent_train

        opt_pct, opt_alpha, best_res = optimize_threshold_alpha(
            stage1_scores, y_val.values, silent_val, k=100,
            thr_grid=[10, 20, 30, 40, 50, 60, 70, 80, 90],
            alpha_grid=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        )

        # Update threshold and ensemble weight
        self.threshold_percentile = opt_pct
        self.threshold_value = np.percentile(stage1_scores, opt_pct)
        self.ensemble_weight = opt_alpha

        log(f"Enhanced model with optimized params: threshold_pct={opt_pct}, alpha={opt_alpha}, "
            f"silent_recall@100={best_res['silent_recall']:.4f}", level=1)

        # Final evaluation
        final_scores = self.predict(X_val)
        self.final_metrics = compute_comprehensive_metrics(y_val, final_scores, [100, 200, 500])

        # Calculate silent zone metrics
        silent_metrics = evaluate_silent_zone(final_scores, silent_val, [100, 200, 500])
        self.final_metrics.update(silent_metrics)

        log(f"Final enhanced model metrics: {self.final_metrics}", level=1)

    # Add a method to get active learning suggestions
    def get_active_learning_suggestions(self, X: pd.DataFrame, silent_grid_ids: set[int],
                                        top_k: int = 100) -> list[int]:
        """
        Get suggestions for grids to investigate next based on active learning principles.
        Returns grid_ids of silent zones with lowest confidence scores.
        """
        if 'grid_id' not in X.columns:
            raise ValueError("grid_id column required for active learning suggestions")

        # Get predictions
        scores = self.predict(X)

        # Create silent mask
        silent_mask = X['grid_id'].isin(silent_grid_ids).values

        # Get indices of silent zones with lowest confidence
        low_conf_indices = select_active_learning_samples(scores, silent_mask, top_k)

        # Return grid_ids
        return X.iloc[low_conf_indices]['grid_id'].tolist()

    def fit(self, X: pd.DataFrame, y: pd.Series, spatial_blocks: np.ndarray = None) -> None:
        """
        Fit both stage 1 and stage 2 models, falling back to unsupervised if needed

        Args:
            X: Feature matrix
            y: Labels (can be None or have few positives - will use unsupervised in this case)
            spatial_blocks: Optional spatial CV blocks
        """
        # Handle the case where y is None
        if y is None:
            log("No labels provided - will use unsupervised approach for Stage 2", level=2)
            has_labels = False
            # Create a dummy y of zeros
            y = pd.Series(0, index=X.index)
        else:
            has_labels = True

        # Split into train and validation
        if spatial_blocks is not None and has_labels:
            gkf = GroupKFold(n_splits=min(5, len(np.unique(spatial_blocks))))
            train_idx, val_idx = next(gkf.split(X, y, groups=spatial_blocks))
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            log(f"Using spatial CV: {len(X_train)} train, {len(X_val)} validation samples", level=2)
        else:
            # Simple split without stratification if no labels
            split_strat = y if has_labels else None
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=split_strat
            )

        # Train stage 1 model with only proximity feature
        log("Training Stage 1 model...", level=2)
        self.stage1_model, train_s1_scores, val_s1_scores, self.stage1_metrics = train_stage1_model(
            X_train, y_train, X_val, y_val,
            proximity_feat=self.proximity_feat,
            model_type=self.stage1_model_type
        )

        # For Stage 2, we'll try both approaches:
        # 1. Use all data if unsupervised
        # 2. Try to use thresholded data if supervised

        # If we're using thresholded approach
        if self.threshold_percentile is not None:
            self.threshold_value = np.percentile(train_s1_scores, self.threshold_percentile)
            log(f"Stage 1 threshold ({self.threshold_percentile}th pct): {self.threshold_value:.4f}", level=2)

            # Create masks for stage 2
            train_mask = train_s1_scores < self.threshold_value
            val_mask = val_s1_scores < self.threshold_value

            # Check if we have enough data in the masks
            train_has_data = train_mask.sum() >= 10
            val_has_data = val_mask.sum() >= 10

            # If masks have data, use them
            if train_has_data and val_has_data:
                X_train_s2 = X_train[train_mask]
                X_val_s2 = X_val[val_mask]
                y_train_s2 = y_train[train_mask]
                y_val_s2 = y_val[val_mask]
            else:
                # Otherwise use all data
                X_train_s2 = X_train
                X_val_s2 = X_val
                y_train_s2 = y_train
                y_val_s2 = y_val
        else:
            # No thresholding, use all data
            X_train_s2 = X_train
            X_val_s2 = X_val
            y_train_s2 = y_train
            y_val_s2 = y_val

        # Train stage 2 model (supervised or unsupervised)
        log("Training Stage 2 model...", level=2)
        self.stage2_model, self.stage2_metrics, self.is_unsupervised = train_stage2_model(
            X_train_s2, y_train_s2 if has_labels else None,
            X_val_s2, y_val_s2 if has_labels else None,
            exclude_features=self.observed_sinkhole_features,
            use_focal_loss=self.use_focal_loss,
            feature_fraction=self.feature_fraction
        )

        # Evaluate on validation data
        if has_labels:
            final_scores = self.predict(X_val)
            self.final_metrics = compute_comprehensive_metrics(y_val, final_scores, [100, 200, 500])
            log(f"Final validation metrics: {self.final_metrics}", level=1)

        # Log what kind of model we ended up with
        model_type = "unsupervised (IsolationForest)" if self.is_unsupervised else "supervised (LightGBM)"
        log(f"Final model uses Stage 1: {self.stage1_model_type} and Stage 2: {model_type}", level=2)

        # Log feature importance if available and not unsupervised
        if not self.is_unsupervised and hasattr(self.stage2_model, "feature_importances_"):
            features = [f for f in X_val.columns if f not in self.observed_sinkhole_features]
            importance = self.stage2_model.feature_importances_
            feat_imp = pd.DataFrame({'feature': features, 'importance': importance})
            feat_imp = feat_imp.sort_values('importance', ascending=False).head(10)
            log(f"Top Stage 2 features: {feat_imp.feature.tolist()}", level=1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate combined risk scores using either supervised or unsupervised approach

        In unsupervised mode, uses ensemble_weight to combine stage1 and anomaly scores
        """
        if self.stage1_model is None:
            raise RuntimeError("Models must be trained first")

        # Stage 1 scores (always available)
        stage1_scores = self.stage1_model.predict_proba(X[[self.proximity_feat]])[:, 1]

        # If no stage 2 model, just return stage 1 scores
        if self.stage2_model is None:
            return stage1_scores

        # Generate stage 2 scores using the appropriate model
        stage2_features = [f for f in X.columns if f not in self.observed_sinkhole_features]

        if self.is_unsupervised:
            # For IsolationForest, get anomaly scores (higher = more anomalous)
            anomaly_scores = -self.stage2_model.decision_function(X[stage2_features])

            # Normalize to [0,1]
            min_score, max_score = anomaly_scores.min(), anomaly_scores.max()
            if max_score > min_score:
                stage2_scores = (anomaly_scores - min_score) / (max_score - min_score)
            else:
                stage2_scores = np.zeros_like(anomaly_scores)

            # Combine using ensemble weight
            alpha = self.ensemble_weight
            final_scores = alpha * stage1_scores + (1 - alpha) * stage2_scores

        else:
            # Original supervised approach
            if self.threshold_value is not None:
                # Apply threshold
                candidates = stage1_scores < self.threshold_value
                final_scores = stage1_scores.copy()

                if candidates.sum() > 0:
                    s2_preds = self.stage2_model.predict_proba(X.loc[candidates, stage2_features])[:, 1]
                    combined = s2_preds * (1 - stage1_scores[candidates])
                    final_scores[candidates] = combined
            else:
                # No threshold, predict for all
                s2_preds = self.stage2_model.predict_proba(X[stage2_features])[:, 1]
                final_scores = self.ensemble_weight * stage1_scores + (1 - self.ensemble_weight) * s2_preds

        return final_scores

    def fit_with_silent_zones(self, X: pd.DataFrame, y: pd.Series, silent_grid_ids: set[int],
                              spatial_blocks: np.ndarray = None) -> None:
        """
        Enhanced fitting that incorporates silent zone information
        as weak positive labels using nnPU-style approach.

        Args:
            X: Feature matrix
            y: Labels
            silent_grid_ids: Set of grid IDs for silent zones
            spatial_blocks: Optional spatial CV blocks
        """
        # First perform normal fitting
        self.fit(X, y, spatial_blocks)

        # Get silent mask
        if 'grid_id' not in X.columns:
            log("Cannot perform silent zone enhancement - grid_id column missing", level=1)
            return

        silent_mask = X['grid_id'].isin(silent_grid_ids).values

        if silent_mask.sum() == 0:
            log("No silent zones found in data", level=1)
            return

        log(f"Enhancing model with {silent_mask.sum()} silent zones as weak positives", level=1)

        # Split data if needed
        if spatial_blocks is not None:
            gkf = GroupKFold(n_splits=min(5, len(np.unique(spatial_blocks))))
            train_idx, val_idx = next(gkf.split(X, y, groups=spatial_blocks))
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Important: Create silent masks for the training and validation sets
            silent_train_mask = X_train['grid_id'].isin(silent_grid_ids).values
            silent_val_mask = X_val['grid_id'].isin(silent_grid_ids).values
        else:
            # Simple split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            # Create masks after splitting
            silent_train_mask = X_train['grid_id'].isin(silent_grid_ids).values
            silent_val_mask = X_val['grid_id'].isin(silent_grid_ids).values

        # Features excluding observed sinkhole features
        features = [c for c in X_train.columns if c not in self.observed_sinkhole_features
                    and c != 'grid_id']

        # Apply pseudo-labeling with silent zones as weak positives
        base_model = lgb.LGBMClassifier(**LGB_PARAMS_BASE)
        enhanced_model = pseudo_labeling_nnpu(
            X_train[features], y_train,
            X_val[features], silent_val_mask,  # Use the validation silent mask
            base_model=base_model,
            unlabeled_weight=0.5
        )

        # Replace Stage 2 model with the enhanced model
        self.stage2_model = enhanced_model
        self.is_unsupervised = False

        # Optimize threshold and alpha
        stage1_scores = self.stage1_model.predict_proba(X_val[[self.proximity_feat]])[:, 1]

        opt_pct, opt_alpha, best_res = optimize_threshold_alpha(
            stage1_scores, y_val.values, silent_val_mask, k=100,
            thr_grid=[10, 20, 30, 40, 50, 60, 70, 80, 90],
            alpha_grid=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        )

        # Update threshold and ensemble weight
        self.threshold_percentile = opt_pct
        self.threshold_value = np.percentile(stage1_scores, opt_pct)
        self.ensemble_weight = opt_alpha

        log(f"Enhanced model with optimized params: threshold_pct={opt_pct}, alpha={opt_alpha}, "
            f"silent_recall@100={best_res['silent_recall']:.4f}", level=1)

        # Final evaluation
        final_scores = self.predict(X_val)
        self.final_metrics = compute_comprehensive_metrics(y_val, final_scores, [100, 200, 500])

        # Calculate silent zone metrics
        silent_metrics = evaluate_silent_zone(final_scores, silent_val_mask, [100, 200, 500])
        self.final_metrics.update(silent_metrics)

        log(f"Final enhanced model metrics: {self.final_metrics}", level=1)

    # 액티브 러닝 추천을 얻는 메서드 추가
    def get_active_learning_suggestions(self, X: pd.DataFrame, silent_grid_ids: set[int],
                                        top_k: int = 100) -> list[int]:
        """
        Get suggestions for grids to investigate next based on active learning principles.
        Returns grid_ids of silent zones with lowest confidence scores.
        """
        if 'grid_id' not in X.columns:
            raise ValueError("grid_id column required for active learning suggestions")

        # Get predictions
        scores = self.predict(X)

        # Create silent mask
        silent_mask = X['grid_id'].isin(silent_grid_ids).values

        # Get indices of silent zones with lowest confidence
        low_conf_indices = select_active_learning_samples(scores, silent_mask, top_k)

        # Return grid_ids
        return X.iloc[low_conf_indices]['grid_id'].tolist()


def pseudo_labeling_nnpu(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_unlabeled: pd.DataFrame,
    silent_mask: np.ndarray,
    base_model=None,
    unlabeled_weight: float = 0.5
) -> lgb.LGBMClassifier:
    """
    Simple nnPU-style pseudo-label: treat silent_mask==True as weak positives
    """
    assert len(silent_mask) == len(
        X_unlabeled), f"Silent mask size {len(silent_mask)} doesn't match X_unlabeled size {len(X_unlabeled)}"

    # create pseudo-labels
    y_unlabeled = np.zeros(len(X_unlabeled), dtype=int)
    y_unlabeled[silent_mask] = 1

    # combine datasets
    X_comb = pd.concat([X_train, X_unlabeled], ignore_index=True)
    y_comb = pd.concat([y_train, pd.Series(y_unlabeled)], ignore_index=True)

    # sample weights: boost weight for pseudo-labels
    w_train = np.ones(len(y_train))
    w_unl = np.ones(len(y_unlabeled)) * unlabeled_weight
    sample_weight = np.concatenate([w_train, w_unl])

    # train model
    model = base_model or lgb.LGBMClassifier(
        objective='binary', metric='auc', verbose=-1
    )
    model.fit(
        X_comb, y_comb,
        sample_weight=sample_weight
    )
    return model


def select_active_learning_samples(
    scores: np.ndarray,
    silent_mask: np.ndarray,
    top_k: int = 100
) -> np.ndarray:
    """
    Return indices of silent zones with lowest confidence (closest to decision boundary)
    """
    # for positive silent zones, low confidence = low predicted score
    silent_scores = np.where(silent_mask, scores, np.nan)
    idx = np.argsort(silent_scores)
    return idx[:top_k]


def optimize_threshold_alpha(
        scores: np.ndarray,
        y_true: np.ndarray,
        silent_mask: np.ndarray,
        k: int = 100,
        thr_grid: list[float] = None,
        alpha_grid: list[float] = None
) -> Tuple[float, float, dict]:
    """
    Grid search over threshold percentile and ensemble weight alpha to maximize silent_recall@k
    Returns default values if optimization fails
    """
    thr_grid = thr_grid or [10, 20, 30, 40, 50]
    alpha_grid = alpha_grid or [0.2, 0.5, 0.8]

    # Initialize with default values
    best_percentile = thr_grid[0]
    best_alpha = alpha_grid[0]
    best = {'silent_recall': 0, 'percentile': best_percentile, 'alpha': best_alpha}

    # Check if we have any silent zones
    if not np.any(silent_mask):
        log(f"No silent zones found, using default values: percentile={best_percentile}, alpha={best_alpha}", level=1)
        return best_percentile, best_alpha, best

    for pct in thr_grid:
        thr = np.percentile(scores, pct)
        s1 = scores.copy()
        # stage2 replaced by 1 - s1 for below threshold
        mask = s1 < thr

        # Skip if mask is all False
        if not np.any(mask):
            continue

        s2 = (1 - s1) * mask
        for alpha in alpha_grid:
            final = alpha * s1 + (1 - alpha) * s2
            # compute recall@k
            order = np.argsort(final)[::-1]
            topk = order[:k]

            rec = np.sum(silent_mask[topk]) / np.sum(silent_mask)
            if rec > best['silent_recall']:
                best.update({'silent_recall': rec, 'percentile': pct, 'alpha': alpha})

    # Log results
    log(f"Threshold optimization: best percentile={best['percentile']}, alpha={best['alpha']}, silent_recall@{k}={best['silent_recall']:.4f}",
        level=1)

    return best['percentile'], best['alpha'], best


LGB_PARAMS_BASE = dict(
    objective="binary",
    metric="auc",
    boosting_type="gbdt",
    learning_rate=0.05,
    num_leaves=31,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_data_in_leaf=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    max_depth=7,
    verbose=-1,
)


def train_stage1_model(X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series,
                       proximity_feat: str = "min_distance_to_sinkhole",
                       model_type: str = "lgbm") -> Tuple[object, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Train Stage 1 model using only proximity feature

    Returns:
        model: Trained model
        train_scores: Stage 1 scores for training set
        val_scores: Stage 1 scores for validation set
        metrics: Performance metrics for stage 1
    """
    # Calculate correct scale_pos_weight
    neg = (y_train == 0).sum()
    pos = y_train.sum()
    spw = neg / pos if pos > 0 else 100

    if model_type == "lgbm":
        params = LGB_PARAMS_BASE.copy()
        params["scale_pos_weight"] = spw
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train[[proximity_feat]], y_train,
            eval_set=[(X_val[[proximity_feat]], y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        train_scores = model.predict_proba(X_train[[proximity_feat]])[:, 1]
        val_scores = model.predict_proba(X_val[[proximity_feat]])[:, 1]
    elif model_type == "logistic":
        model = LogisticRegression(class_weight='balanced', max_iter=1000)
        model.fit(X_train[[proximity_feat]], y_train)
        train_scores = model.decision_function(X_train[[proximity_feat]])
        val_scores = model.decision_function(X_val[[proximity_feat]])
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Calculate performance metrics
    auc_s1 = roc_auc_score(y_val, val_scores)
    pr, rc, _ = precision_recall_curve(y_val, val_scores)
    aurpr_s1 = auc(rc, pr)

    metrics = {
        "auc": auc_s1,
        "pr_auc": aurpr_s1
    }

    log(f"Stage 1 ({model_type}): AUC={auc_s1:.4f}, PR-AUC={aurpr_s1:.4f}, scale_pos_weight={spw:.1f}", level=2)

    return model, train_scores, val_scores, metrics


def train_stage2_model(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        exclude_features: List[str] = None,
        use_focal_loss: bool = True,
        feature_fraction: float = 0.6,
        min_positives: int = 5
) -> Tuple[Optional[object], Dict[str, float], bool]:
    """
    Train Stage 2 model - using supervised or unsupervised approach depending on available labels

    Returns:
        model: Trained model or None if skipped
        metrics: Performance metrics for stage 2
        is_unsupervised: Flag indicating if an unsupervised model was used
    """
    # Filter features
    features = [c for c in X_train.columns if not exclude_features or c not in exclude_features]
    X_train_filtered = X_train[features]
    X_val_filtered = X_val[features]

    # Check if we have enough labeled data for supervised learning
    supervised_ok = (y_train is not None and y_train.sum() >= min_positives and
                     y_val is not None and y_val.sum() >= min_positives)

    if supervised_ok:
        # Traditional supervised approach
        params = LGB_PARAMS_BASE.copy()
        neg = (y_train == 0).sum()
        pos = y_train.sum()
        spw = neg / pos if pos > 0 else 200

        params.update({
            "scale_pos_weight": spw,
            "feature_fraction": feature_fraction,
            "reg_alpha": 5.0,
        })

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train_filtered, y_train,
            eval_set=[(X_val_filtered, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        # Compute metrics
        val_preds = model.predict_proba(X_val_filtered)[:, 1]
        try:
            auc_s2 = roc_auc_score(y_val, val_preds)
            pr, rc, _ = precision_recall_curve(y_val, val_preds)
            aurpr_s2 = auc(rc, pr)
        except Exception:
            auc_s2 = 0.0
            aurpr_s2 = 0.0

        metrics = {"auc": auc_s2, "pr_auc": aurpr_s2}
        log(f"Stage 2 (supervised): AUC={auc_s2:.4f}, PR-AUC={aurpr_s2:.4f}, spw={spw:.1f}", level=2)
        return model, metrics, False

    else:
        # Unsupervised anomaly detection approach
        log("Insufficient labeled data for Stage 2, using unsupervised anomaly detection", level=2)

        # Use IsolationForest for anomaly detection
        contamination = 0.05  # Proportion of outliers expected
        iso = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )

        # Fit on combined train+val since we don't need labels
        combined_X = pd.concat([X_train_filtered, X_val_filtered])
        iso.fit(combined_X)

        # Get anomaly scores for validation (higher = more anomalous)
        val_scores = -iso.decision_function(X_val_filtered)

        # Normalize to [0,1] for better combination with Stage 1
        min_score, max_score = val_scores.min(), val_scores.max()
        if max_score > min_score:
            val_scores = (val_scores - min_score) / (max_score - min_score)

        # Calculate metrics if we happen to have labels
        metrics = {}
        if y_val is not None and y_val.sum() > 0:
            try:
                auc_iso = roc_auc_score(y_val, val_scores)
                metrics["auc"] = auc_iso
                log(f"Stage 2 (unsupervised): AUC={auc_iso:.4f}", level=2)
            except Exception:
                log("Unable to calculate metrics for unsupervised model", level=2)

        return iso, metrics, True



# ────────────────────────────────────────────────────────────────────────────────
# 7. Spatial CV Evaluation with Incremental Data
# ────────────────────────────────────────────────────────────────────────────────

def evaluate_spatial_cv(
        gdf: gpd.GeoDataFrame,
        n_folds: int = 5,
        k_vals: List[int] = [100, 200, 500],
        silent_grid_ids: Optional[set[int]] = None ,
) -> Dict:
    """
    Evaluate model using spatial cross-validation

    Args:
        gdf: GeoDataFrame with data and geometry
        n_folds: Number of spatial folds
        k_vals: List of K values for topK metrics

    Returns:
        Dictionary with evaluation metrics
    """
    log(f"Performing {n_folds}-fold spatial cross-validation", level=1)

    # Prepare data
    y = gdf["subsidence_occurrence"].astype(int)
    X = gdf.drop(columns=[c for c in [
        "grid_id", "subsidence_occurrence", "subsidence_count", "geometry"]
                          if c in gdf.columns])

    # Create spatial blocks
    spatial_blocks = create_spatial_blocks(gdf, n_blocks=n_folds)

    # Initialize metrics collectors
    fold_metrics = []

    # For each fold
    gkf = GroupKFold(n_splits=n_folds)
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=spatial_blocks)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        blocks_train = spatial_blocks[train_idx]

        # Train two-stage model with auto-tuning
        screener = TwoStageSinkholeScreener(
            proximity_feat="min_distance_to_sinkhole",
            threshold_percentile=20,  # Auto-tune
            use_focal_loss=True,
            feature_fraction=0.6
        )
        screener.fit(X_train, y_train, spatial_blocks=blocks_train)

        # Evaluate on test fold
        test_scores = screener.predict(X_test)
        metrics = compute_comprehensive_metrics(y_test, test_scores, k_vals)

        if silent_grid_ids:
            silent_mask = gdf.iloc[test_idx]["grid_id"].isin(silent_grid_ids).values
            silent_metrics = evaluate_silent_zone(test_scores, silent_mask, k_vals)
            metrics.update(silent_metrics)
        metrics["fold"] = fold
        # Store metrics
        fold_metrics.append(metrics)
        log(f"[Fold{fold + 1}] IoU@100={metrics['iou@100']:.3f} "
            f"SilentRecall@100={metrics.get('silent_recall@100', 0):.3f}", level=1)

    # Aggregate metrics
    aggregate_metrics = {}
    for key in fold_metrics[0].keys():
        if key != "fold":
            values = [m[key] for m in fold_metrics]
            aggregate_metrics[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "values": values
            }

    # Log summary
    log("\n=== Spatial CV Summary ===")
    for key, stats in aggregate_metrics.items():
        log(f"{key}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    return {
        "fold_metrics": fold_metrics,
        "aggregate": aggregate_metrics
    }

# ────────────────────────────────────────────────────────────────────────────────
# 8. Hyperparameter Optimization for Two-Stage Model
# ────────────────────────────────────────────────────────────────────────────────

def optimize_hyperparameters(X: pd.DataFrame, y: pd.Series, spatial_blocks: np.ndarray = None) -> Dict:
    """
    Optimize hyperparameters for the two-stage model

    Returns:
        Dictionary with best hyperparameters and performance metrics
    """
    log("Optimizing hyperparameters for two-stage model...", level=1)

    # Define parameter grid
    param_grid = {
        'threshold_percentile': [70, 75, 80, 85, 90],
        'feature_fraction': [0.5, 0.6, 0.7],
        'proximity_feat': ['min_distance_to_sinkhole', 'weighted_sinkhole_density'],
        'stage1_model_type': ['lgbm', 'logistic']
    }

    # Generate all combinations
    param_combinations = list(ParameterGrid(param_grid))
    log(f"Testing {len(param_combinations)} parameter combinations", level=2)

    # Setup spatial CV
    if spatial_blocks is not None:
        gkf = GroupKFold(n_splits=min(3, len(np.unique(spatial_blocks))))
        splits = list(gkf.split(X, y, groups=spatial_blocks))
    else:
        # Fallback to standard stratified split
        train_idx, val_idx = train_test_split(
            np.arange(len(X)), test_size=0.2, random_state=42, stratify=y
        )
        splits = [(train_idx, val_idx)]

    # Track best parameters
    best_score = 0
    best_params = None
    all_results = []

    # Evaluate each combination
    for params in param_combinations:
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(splits):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            try:
                # Train model with current parameters
                screener = TwoStageSinkholeScreener(
                    proximity_feat=params['proximity_feat'],
                    stage1_model_type=params['stage1_model_type'],
                    threshold_percentile=params['threshold_percentile'],
                    feature_fraction=params['feature_fraction'],
                    use_focal_loss=True
                )
                screener.fit(X_train, y_train)

                # Evaluate
                val_preds = screener.predict(X_val)
                iou = calculate_iou(y_val.values, val_preds, 100)
                fold_scores.append(iou)
            except Exception as e:
                log(f"Error with params {params}: {str(e)}")
                continue

        # Calculate average score across folds
        if fold_scores:
            avg_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)

            # Store result
            result = {
                **params,
                'iou@100_mean': avg_score,
                'iou@100_std': std_score
            }
            all_results.append(result)

            # Update best parameters
            if avg_score > best_score:
                best_score = avg_score
                best_params = params

    if best_params:
        log(f"\nBest parameters: {best_params}", level=1)
        log(f"Best IoU@100: {best_score:.4f}", level=1)
    else:
        log("No valid parameter combinations found", level=1)

    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(all_results)

    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': results_df
    }
# ────────────────────────────────────────────────────────────────────────────────
# 10. Silent Zone Evaluation
# ────────────────────────────────────────────────────────────────────────────────
def evaluate_silent_zone(scores: np.ndarray,
                         silent_mask: np.ndarray,
                         k_vals: List[int] = [100, 200, 500]) -> Dict[str, float]:
    """
    Evaluate model performance on silent zones (조사 필요 지역)

    Args:
        scores: Model prediction scores for all samples
        silent_mask: Boolean array where True = 조사 필요 지역
        k_vals: List of K values to compute metrics at

    Returns:
        Dictionary of recall, precision, lift @ K
    """
    scores = np.asarray(scores)
    silent_mask = np.asarray(silent_mask)
    total_silent = silent_mask.sum()
    overall_rate = total_silent / len(silent_mask) if len(silent_mask) > 0 else 0

    order = np.argsort(scores)[::-1]
    silent_sorted = silent_mask[order]

    results = {}
    for k in k_vals:
        if k > len(scores):
            continue
        topk_count = silent_sorted[:k].sum()
        recall_k = topk_count / total_silent if total_silent else 0
        precision_k = topk_count / k
        lift_k = precision_k / overall_rate if overall_rate else 0
        results[f"silent_recall@{k}"] = recall_k
        results[f"silent_precision@{k}"] = precision_k
        results[f"silent_lift@{k}"] = lift_k
    return results

# ────────────────────────────────────────────────────────────────────────────────
# Silent-Zone helpers
# ────────────────────────────────────────────────────────────────────────────────
def load_silent_grid_ids(table: str = "survey_grid_strict") -> set[int]:
    """
    PostGIS 테이블에서 조사-필요(Grid) id 집합을 가져온다.
    조건: sinkhole 이력이 없는 grid 만 포함 (테이블 자체가 이미 필터된 상태라고 가정)
    """
    load_dotenv()
    engine = create_engine(os.getenv("DB_DSN"))
    q = f"SELECT DISTINCT grid_id FROM {table}"
    return set(pd.read_sql_query(q, engine)["grid_id"].tolist())

def evaluate_silent_zone(scores: np.ndarray,
                         silent_mask: np.ndarray,
                         k_vals: List[int] = [100, 200, 500]) -> Dict[str, float]:
    """Recall / Precision / Lift @K for Silent-Zone."""
    order = scores.argsort()[::-1]
    silent_sorted = silent_mask[order]
    total_silent = silent_mask.sum()
    overall_rate = total_silent / len(scores)
    out = {}
    for k in k_vals:
        if k > len(scores): continue
        topk = silent_sorted[:k].sum()
        recall = topk / total_silent if total_silent else 0
        prec   = topk / k
        lift   = prec / overall_rate if overall_rate else 0
        out[f"silent_recall@{k}"]    = recall
        out[f"silent_precision@{k}"] = prec
        out[f"silent_lift@{k}"]      = lift
    return out
# ────────────────────────────────────────────────────────────────────────────────
# 9. Main Pipeline
# ────────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # List of report tables to evaluate
    report_tables = [
        # "feature_matrix_25_geo",
        # "feature_matrix_50_geo",
        # "feature_matrix_75_geo",
        "feature_matrix_100_geo"
    ]

    # 1. Load the 100-report dataset for hyperparameter optimization
    log("\n=== Step 1: Hyperparameter Optimization ===", level=1)
    gdf_100 = load_dataset(table="feature_matrix_100_geo")

    # Prepare data - grid_id 컬럼 유지 (silent zone 통합에 필요)
    y = gdf_100["subsidence_occurrence"].astype(int)
    X = gdf_100.drop(columns=[c for c in [
        "subsidence_occurrence", "subsidence_count", "geometry"]
                              if c in gdf_100.columns])

    # Create spatial blocks
    spatial_blocks = create_spatial_blocks(gdf_100, n_blocks=5)

    # Get silent grid IDs
    silent_ids = load_silent_grid_ids("survey_grid_strict")
    log(f"Loaded {len(silent_ids)} silent grid IDs", level=1)

    # Optimize hyperparameters
    X_for_opt = X.drop('grid_id', axis=1) if 'grid_id' in X.columns else X  # grid_id는 feature로 사용하지 않음
    opt_results = optimize_hyperparameters(X_for_opt, y, spatial_blocks)

    # 2. 사일런트 존 통합으로 향상된 모델 학습
    log("\n=== Step 2: Training Enhanced Model with Silent Zone Integration ===", level=1)

    # Use the best parameters from optimization
    best_params = opt_results['best_params']

    # If optimization failed, use defaults
    if best_params is None:
        best_params = {
            'proximity_feat': 'min_distance_to_sinkhole',
            'stage1_model_type': 'lgbm',
            'threshold_percentile': 80,
            'feature_fraction': 0.6
        }

    # Create model with best parameters
    log(f"Using parameters: {best_params}", level=1)

    # 향상된 모델 생성 및 학습
    enhanced_model = TwoStageSinkholeScreener(
        proximity_feat=best_params['proximity_feat'],
        stage1_model_type=best_params['stage1_model_type'],
        threshold_percentile=best_params['threshold_percentile'],
        feature_fraction=best_params['feature_fraction']
    )

    # Train enhanced model and get active learning suggestions
    try:
        # Only call once
        enhanced_model.fit_with_silent_zones(X, y, silent_ids, spatial_blocks)

        # 액티브 러닝 추천 얻기
        top_suggestions = enhanced_model.get_active_learning_suggestions(X, silent_ids, top_k=50)
        log(f"Top 50 grid IDs for investigation: {top_suggestions[:10]}... (and 40 more)", level=1)

        # 추천 결과 저장
        pd.DataFrame({'grid_id': top_suggestions}).to_csv('active_learning_suggestions.csv', index=False)
        log("Active learning suggestions saved to 'active_learning_suggestions.csv'", level=1)
    except Exception as e:
        log(f"Error in enhanced model training or active learning: {str(e)}", level=1)
        log("Continuing with standard model and evaluation...", level=1)
        # Fall back to basic model if enhanced fails
        enhanced_model.fit(X, y, spatial_blocks)

    # 3. Evaluate with PU-learning spatial CV across different report counts
    log("\n=== Step 3: Evaluating Report Impact with PU-Learning Simulation ===", level=1)

    # Evaluate report impact using PU-learning simulation
    pu_results = evaluate_reports_impact_pu(
        report_tables,
        k_vals=[100, 200, 500],
        n_iters=10,
        random_state=42
    )

    # 4. Create results table and plots
    log("\n=== Step 4: Creating PU-Learning Report Impact Analysis ===", level=1)

    # Create results table with expanded metrics
    report_counts = sorted(pu_results.keys())

    # Table columns with all metrics
    metrics = ([f"recall@{k}" for k in [100, 200, 500]] +
               [f"precision@{k}" for k in [100, 200, 500]] +
               [f"lift@{k}" for k in [100, 200, 500]] +
               ['pr_auc'])

    table_data = []
    for count in report_counts:
        row = [count]
        for metric in metrics:
            mean = pu_results[count][f"{metric}_mean"]
            std = pu_results[count][f"{metric}_std"]
            row.append(f"{mean:.4f} ± {std:.4f}")
        table_data.append(row)

    table_df = pd.DataFrame(table_data, columns=["Report Count"] + metrics)
    log("\nPU-Learning Results table:", level=1)
    log(table_df.to_string(index=False), level=1)

    # Save results table
    table_df.to_csv('baseline.csv', index=False)
    log("PU results saved to 'baseline.csv'", level=1)

    # Plot improvement with increasing reports
    plot_pu_report_impact(pu_results, k_vals=[100, 200, 500])

    log("Analysis complete.", level=1)

if __name__ == "__main__":
    t0 = time.time()
    main()
    log(f"Total runtime {(time.time() - t0):.1f}s")
