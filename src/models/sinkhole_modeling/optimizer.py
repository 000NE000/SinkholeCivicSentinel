"""
Optimization utilities for sinkhole screening with two-stage caching approach
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Any, Optional
from src.models.sinkhole_modeling.config import log, THRESHOLD_PERCENTILE_GRID, ALPHA_GRID, FEATURE_FRACTION_GRID
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import hashlib
import json
import torch
import random

from src.models.sinkhole_modeling.evaluation import calculate_iou
from src.models.sinkhole_modeling.stage1_model import TwoStageSinkholeScreener


class TwoStageSinkholeScreenerSplit(TwoStageSinkholeScreener):
    """Extension of TwoStageSinkholeScreener with separate stage 1 and 2 training"""

    def fit_stage1(self, X: pd.DataFrame, y: pd.Series, spatial_blocks: np.ndarray = None):
        """Train only the stage 1 model"""
        # 1) Save geometry if present for later use
        geom = X.pop("geometry") if "geometry" in X.columns else None
        X = X.drop(columns=["grid_id", "subsidence_occurrence", "subsidence_count"], errors="ignore")

        # 2) Create train/val split for early stopping
        from src.models.sinkhole_modeling.stage1_model import get_stratified_group_splits
        splits = get_stratified_group_splits(X, y, groups=spatial_blocks, n_splits=5, seed=42)
        train_idx, val_idx = splits[0]
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 3) Train stage 1 model with proximity feature
        from src.models.sinkhole_modeling.stage1_model import train_stage1_model
        import geopandas as gpd

        # Restore geometry for calculation of proximity features
        if geom is not None:
            X_train = gpd.GeoDataFrame(
                X_train,
                geometry=geom.loc[X_train.index],
                crs="EPSG:5179"
            )
            X_val = gpd.GeoDataFrame(
                X_val,
                geometry=geom.loc[X_val.index],
                crs="EPSG:5179"
            )

        self.stage1_model, _, _, self.stage1_metrics = train_stage1_model(
            X_train, y_train, X_val, y_val,
            proximity_feat=self.proximity_feat,
            model_type=self.stage1_model_type
        )

        # Put back geometry for future use
        if geom is not None:
            X["geometry"] = geom

        return self

    def predict_stage1_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Get raw scores from stage 1 model"""
        if self.stage1_model is None:
            raise RuntimeError("Stage 1 model must be trained first")

        # Generate stage 1 scores
        if hasattr(self.stage1_model, 'predict_proba'):
            return self.stage1_model.predict_proba(X[[self.proximity_feat]])[:, 1]
        else:
            return self.stage1_model.decision_function(X[[self.proximity_feat]])


def _get_param_hash(params: dict) -> str:
    """Generate a hash for parameter combination to use as cache key"""
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()


def _evaluate_stage1_params(params: dict,
                           X: pd.DataFrame,
                           y: pd.Series,
                           fold_id: int,
                           train_idx: np.ndarray,
                           val_idx: np.ndarray,
                           spatial_blocks: np.ndarray = None) -> Dict:
    """
    Train stage 1 model with given parameters and return scores for validation set
    """
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    blocks_tr = spatial_blocks[train_idx] if spatial_blocks is not None else None

    if 'geometry' not in X_tr.columns:
        log("Skipping fold: 'geometry' column missing in training data", level=2)
        return None

    # Initialize stage 1 model only
    screener = TwoStageSinkholeScreenerSplit(
        proximity_feat=params['proximity_feat'],
        stage1_model_type=params['stage1_model_type'],
        threshold_percentile=params['threshold_percentile'],
        feature_fraction=params['feature_fraction'],
        use_focal_loss=True,
        # Default values for stage 2 params, not relevant for stage 1
        ensemble_weight=0.5,
        stage2_model_type='lightgbm',
        margin_low=0.4,
        margin_high=0.6
    )

    # Train stage 1 model and get scores
    screener.fit_stage1(X_tr, y_tr, spatial_blocks=blocks_tr)
    stage1_scores = screener.predict_stage1_scores(X_val)


    if 'is_silent' in X_val.columns:
        is_silent = X_val['is_silent'].values
    else:
        is_silent = np.zeros(len(X_val), dtype=int)

    return {
        'fold_id': fold_id,
        'params': params,
        'stage1_scores': stage1_scores,
        'y_val': y_val.values,
        'val_idx': val_idx,
        'is_silent': is_silent
    }


def _evaluate_stage2_params(stage1_result: Dict,
                           stage2_params: Dict,
                           X: pd.DataFrame) -> Dict:
    """
    Evaluate stage 2 parameters using pre-computed stage 1 scores
    """
    if stage1_result is None:
        return None

    # Extract data from stage1 result
    fold_id = stage1_result['fold_id']
    stage1_params = stage1_result['params']
    stage1_scores = stage1_result['stage1_scores']
    y_val = stage1_result['y_val']
    val_idx = stage1_result['val_idx']
    is_silent = stage1_result.get('is_silent', np.zeros_like(y_val))

    X_val = X.iloc[val_idx]

    # Get threshold and compute stage 2 scores
    threshold = np.percentile(stage1_scores, stage1_params['threshold_percentile'])
    mask = stage1_scores < threshold

    # Skip if mask is all False (no candidates for stage 2)
    if not np.any(mask):
        return {
            'fold_id': fold_id,
            'params': {**stage1_params, **stage2_params},
            'iou': 0.0
        }

    # Compute final ensemble scores
    stage2_scores = (1 - stage1_scores) * mask
    alpha = stage2_params['alpha']
    final_scores = alpha * stage1_scores + (1 - alpha) * stage2_scores

    # Get top K predictions
    top_k = 100
    order = np.argsort(final_scores)[::-1]
    topk_indices = order[:top_k]
    preds = np.zeros_like(y_val)
    preds[topk_indices] = 1

    # Calculate IoU
    iou = calculate_iou(y_val, preds, top_k=top_k)

    # Calculate silent recall if we have silent zones
    silent_recall = 0
    if np.any(is_silent):
        silent_topk = is_silent[topk_indices]
        silent_recall = np.sum(silent_topk) / max(1, np.sum(is_silent))

    return {
        'fold_id': fold_id,
        'params': {**stage1_params, **stage2_params},
        'iou': iou,
        'silent_recall': silent_recall
    }


def optimize_threshold_alpha(
        scores: np.ndarray,
        y_true: np.ndarray,
        silent_mask: np.ndarray,
        k: int = 100,
        thr_grid: List[float] = None,
        alpha_grid: List[float] = None
) -> Tuple[float, float, Dict]:
    """
    Grid search over threshold percentile and ensemble weight alpha to maximize silent_recall@k

    Args:
        scores: Prediction scores from stage 1 model
        y_true: True labels
        silent_mask: Boolean mask for silent zones
        k: K value for evaluation metrics
        thr_grid: List of threshold percentiles to try
        alpha_grid: List of alpha values to try

    Returns:
        Tuple of (best_percentile, best_alpha, metric_dict)
    """
    # Use smaller grids for faster optimization
    thr_grid = thr_grid or [50, 60, 70, 80, 90]  # Reduced from THRESHOLD_PERCENTILE_GRID
    alpha_grid = alpha_grid or [0.3, 0.5, 0.7]   # Reduced from ALPHA_GRID

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


def optimize_hyperparameters(X: pd.DataFrame, y: pd.Series, spatial_blocks: np.ndarray = None,
                            n_jobs: int = -1, use_randomized: bool = True) -> Dict:
    """
    Optimized hyperparameter search with two-stage approach and caching

    Args:
        X: Feature DataFrame
        y: Target Series
        spatial_blocks: Spatial block identifiers for spatial CV
        n_jobs: Number of parallel jobs
        use_randomized: Whether to use randomized search instead of full grid

    Returns:
        Dictionary with best parameters and results
    """
    from sklearn.model_selection import ParameterGrid, GroupKFold, train_test_split

    log("Optimizing hyperparameters with two-stage approach...", level=1)

    # 1) Define separate parameter grids for stage 1 and stage 2
    stage1_param_grid = {
        'threshold_percentile': THRESHOLD_PERCENTILE_GRID,
        'feature_fraction': FEATURE_FRACTION_GRID,
        'proximity_feat': ['min_distance_to_sinkhole', 'weighted_sinkhole_density'],
        'stage1_model_type': ['lgbm', 'logistic'],
    }

    stage2_param_grid = {
        'model_type': ['lightgbm', 'gnn', 'um_gnn'],
        'margin_low': [0.3, 0.4, 0.5],
        'margin_high': [0.5, 0.6, 0.7],
        'alpha': ALPHA_GRID
    }

    # Generate parameter combinations
    stage1_params_list = list(ParameterGrid(stage1_param_grid))
    stage2_params_list = list(ParameterGrid(stage2_param_grid))

    # Optionally reduce search space using randomized search
    if use_randomized:
        n_stage1 = min(10, len(stage1_params_list))
        n_stage2 = min(10, len(stage2_params_list))
        random.seed(42)
        stage1_params_list = random.sample(stage1_params_list, n_stage1)
        stage2_params_list = random.sample(stage2_params_list, n_stage2)

    log(f"Stage 1 parameter combinations: {len(stage1_params_list)}", level=1)
    log(f"Stage 2 parameter combinations: {len(stage2_params_list)}", level=1)
    log(f"Total evaluations: {len(stage1_params_list) * len(stage2_params_list)}", level=1)

    # 2) Prepare CV splits (using fewer folds)
    if spatial_blocks is not None:
        # Use only 2 folds for spatial cross-validation
        gkf = GroupKFold(n_splits=min(2, len(np.unique(spatial_blocks))))
        splits = list(gkf.split(X, y, groups=spatial_blocks))
    else:
        # Use a single train-test split
        train_idx, val_idx = train_test_split(
            np.arange(len(X)), test_size=0.2, random_state=42, stratify=y
        )
        splits = [(train_idx, val_idx)]

    # 3) First stage: Train all stage 1 models and cache results
    log("Stage 1: Training initial models and caching scores...", level=1)
    stage1_results = {}

    stage1_tasks = []
    for params in stage1_params_list:
        for fold_id, (train_idx, val_idx) in enumerate(splits):
            stage1_tasks.append((params, fold_id, train_idx, val_idx))

    # Execute stage 1 evaluations in parallel
    stage1_outputs = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_evaluate_stage1_params)(
            params, X, y, fold_id, train_idx, val_idx, spatial_blocks
        )
        for params, fold_id, train_idx, val_idx in tqdm(stage1_tasks, desc="Stage 1 Grid Search")
    )

    # Store results in cache
    for result in stage1_outputs:
        if result is not None:
            params_hash = _get_param_hash(result['params'])
            stage1_results[(params_hash, result['fold_id'])] = result

    # 4) Second stage: Evaluate stage 2 parameters using cached stage 1 results
    log("Stage 2: Evaluating ensemble models using cached scores...", level=1)

    stage2_tasks = []
    for stage1_params in stage1_params_list:
        for stage2_params in stage2_params_list:
            for fold_id in range(len(splits)):
                params_hash = _get_param_hash(stage1_params)
                stage1_result = stage1_results.get((params_hash, fold_id))
                if stage1_result is not None:
                    stage2_tasks.append((stage1_result, stage2_params))

    # Execute stage 2 evaluations in parallel
    stage2_outputs = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_evaluate_stage2_params)(stage1_result, stage2_params, X)
        for stage1_result, stage2_params in tqdm(stage2_tasks, desc="Stage 2 Grid Search")
    )

    # 5) Aggregate results and find best parameters
    all_results = []
    for result in stage2_outputs:
        if result is not None:
            all_results.append({
                **result['params'],
                'iou@100': result['iou'],
                'silent_recall@100': result.get('silent_recall', 0),
                'fold_id': result['fold_id']
            })

    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(all_results)

    # Calculate mean scores across folds
    mean_results = results_df.groupby([col for col in results_df.columns
                                      if col not in ['fold_id', 'iou@100', 'silent_recall@100']]
                                     ).agg({
                                         'iou@100': ['mean', 'std'],
                                         'silent_recall@100': ['mean', 'std']
                                     }).reset_index()

    # Flatten multi-index columns
    mean_results.columns = ['_'.join(col).strip('_') for col in mean_results.columns.values]

    # Sort by IoU score
    mean_results = mean_results.sort_values('iou@100_mean', ascending=False).reset_index(drop=True)

    # Get best parameters
    best_params = mean_results.iloc[0].to_dict()
    best_params = {k: v for k, v in best_params.items() if not k.endswith(('_mean', '_std'))}
    best_score = mean_results.iloc[0]['iou@100_mean']

    log(f"Best IoU@100 = {best_score:.4f} with params: {best_params}", level=1)

    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': mean_results
    }


def _evaluate_params(params: dict,
                     X: pd.DataFrame,
                     y: pd.Series,
                     splits: List,
                     spatial_blocks: np.ndarray) -> Dict:
    """
    Single, non-cached parameter evaluation method
    (Preserved for backward compatibility or simple runs)
    """
    scores = []
    for train_idx, val_idx in splits:
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        blocks_tr = spatial_blocks[train_idx] if spatial_blocks is not None else None
        if 'geometry' not in X_tr.columns:
            log("Skipping fold: 'geometry' column missing in training data", level=2)
            continue

        # TwoStage 모델 초기화 & 학습
        screener = TwoStageSinkholeScreener(
            proximity_feat=params['proximity_feat'],
            stage1_model_type=params['stage1_model_type'],
            threshold_percentile=params['threshold_percentile'],
            feature_fraction=params['feature_fraction'],
            use_focal_loss=True,
            ensemble_weight=params['alpha'],
            stage2_model_type=params['model_type'],
            margin_low=params['margin_low'],
            margin_high=params['margin_high']
        )
        screener.fit(X_tr, y_tr, spatial_blocks=blocks_tr)
        # 예측 및 IoU 계산
        preds = screener.predict(X_val)
        scores.append(calculate_iou(y_val.values, preds, top_k=100))

    return {
        'params': params,
        'mean_iou': np.mean(scores),
        'std_iou': np.std(scores)
    }