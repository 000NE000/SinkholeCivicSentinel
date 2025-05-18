"""
Stage 1 model implementation for sinkhole screening
"""
import pandas as pd
import geopandas as gpd
from typing import Tuple, Dict, List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import GroupKFold, train_test_split
import lightgbm as lgb
from src.models.sinkhole_modeling.config import log, LGB_PARAMS_BASE, OBSERVED_SINKHOLE_FEATURES, DEFAULT_K_VALS, GRAPHSAGE_PARAMS
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold

def get_stratified_group_splits(X, y, groups=None, n_splits=5, seed=42):
    if groups is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(skf.split(X, y))
    else:
        # GroupKFold 후 각 fold 안에서 StratifiedKFold 2-way split
        unique = np.unique(groups)
        y_grp = np.array([y[groups==g].mean() for g in unique])
        gkf = GroupKFold(n_splits=min(n_splits, len(unique)))
        splits = []
        for train_g, val_g in gkf.split(unique, y_grp, groups=unique):
            train_groups, val_groups = unique[train_g], unique[val_g]
            idx_tr = np.where(np.isin(groups, train_groups))[0]
            idx_val= np.where(np.isin(groups, val_groups))[0]
            # 그 안에서 한 번만 StratifiedKFold 로 쪼갬
            skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
            sub_tr, sub_val = next(skf.split(X.iloc[idx_tr], y.iloc[idx_tr]))
            splits.append((idx_tr[sub_tr], idx_tr[sub_val]))
        return splits


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


# The TwoStageSinkholeScreener class defined here, but with implementation in methods below
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
                 ensemble_weight: float = 0.5,
                 stage2_model_type: str = "lightgbm",  # 추가
                 margin_low: float = 0.4,  # 추가
                 margin_high: float = 0.6):  # 추가
        self.device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
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
        self.observed_sinkhole_features = OBSERVED_SINKHOLE_FEATURES

        self.stage2_model_type = stage2_model_type  # 'lightgbm' | 'gnn' | 'um_gnn'
        self.margin_low = margin_low
        self.margin_high = margin_high

        self.stage1_metrics = {}
        self.stage2_metrics = {}
        self.final_metrics = {}

    def _tune_threshold_percentile(self, X: pd.DataFrame, y: pd.Series,
                                   spatial_blocks: np.ndarray = None) -> float:
        """
        Tune the threshold percentile using cross-validation
        """
        from evaluation import calculate_iou
        from stage2_model import train_stage2_model

        log("Tuning threshold percentile...", level=2)

        # Candidate percentiles to try
        percentiles = [70, 75, 80, 85, 90]
        results = {}

        # Set up cross-validation
        if spatial_blocks is not None:
            cv = GroupKFold(n_splits=min(5, len(np.unique(spatial_blocks))))
            list(get_stratified_group_splits(X, y, spatial_blocks))
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

                log(f"Percentile {percentile}: Stage 2 train: {train_mask.sum()} samples, {train_pos} pos ({train_pos_pct:.4%})",
                    level=2)
                log(f"Percentile {percentile}: Stage 2 val: {val_mask.sum()} samples, {val_pos} pos ({val_pos_pct:.4%})",
                    level=2)

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
                log(f"Percentile {percentile}, Fold {fold + 1}: IoU@100 = {iou:.4f}", level=2)

            # Average IoU across folds
            if fold_ious:
                mean_iou = np.mean(fold_ious)
                std_iou = np.std(fold_ious)
                results[percentile] = (mean_iou, std_iou)
                log(f"Percentile {percentile}: IoU@100 = {mean_iou:.4f} ± {std_iou:.4f}", level=2)

        # Select best percentile
        if results:
            best_percentile = max(results.keys(), key=lambda p: results[p][0])
            log(f"Best threshold percentile: {best_percentile} with IoU = {results[best_percentile][0]:.4f}", level=2)
            return best_percentile
        else:
            log("Warning: Could not tune percentile, using default 80", level=2)
            return 80.0

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            spatial_blocks: np.ndarray = None,
            model_type: str = "lightgbm",
            margin_low: float = 0.4,
            margin_high: float = 0.6
            ) -> None:
        """
        Fit both stage 1 and stage 2 models, falling back to unsupervised if needed

        Args:
            X: Feature matrix
            y: Labels (can be None or have few positives - will use unsupervised in this case)
            spatial_blocks: Optional spatial CV blocks
        """
        from src.models.sinkhole_modeling.stage2_model import train_stage2_model
        from src.models.sinkhole_modeling.evaluation import compute_comprehensive_metrics

        self.stage2_feature_names = X.columns.tolist()

        # --- (1) geometry / non-numeric 컬럼 제거 ---
        geom = X.pop("geometry") if "geometry" in X.columns else None
        X = X.drop(columns=["grid_id", "subsidence_occurrence", "subsidence_count"], errors="ignore")

        has_labels = (y.sum() > 0)
        # --- (2) split ---
        splits = get_stratified_group_splits(X, y, groups=spatial_blocks if has_labels else None, n_splits=5, seed=42)
        train_idx, val_idx = splits[0]
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # --- coords 계산 (geometry 사용) ---
        if geom is not None:
            # GeoSeries 로 변환하여 centroid 계산
            g_ser = gpd.GeoSeries(geom, crs="EPSG:5179")
            coords_train = np.c_[
                g_ser.iloc[train_idx].centroid.x.values,
                g_ser.iloc[train_idx].centroid.y.values
            ]
            coords_val = np.c_[
                g_ser.iloc[val_idx].centroid.x.values,
                g_ser.iloc[val_idx].centroid.y.values
            ]
        else:
            # geometry 가 없을 때는 GNN·Spatial 기능 OFF
            coords_train = coords_val = None

        # Train stage 1 model with only proximity feature
        log("Training Stage 1 model...", level=2)
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



        self.stage1_model, train_s1_scores, val_s1_scores, self.stage1_metrics = train_stage1_model(
            X_train, y_train, X_val, y_val,
            proximity_feat=self.proximity_feat,
            model_type=self.stage1_model_type
        )


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
            X_train,
            y_train,
            X_val,
            y_val,
            train_s1_scores,
            val_s1_scores,
            coords_train=coords_train,
            coords_val=coords_val,
            exclude_features=self.observed_sinkhole_features,
            feature_fraction=self.feature_fraction,
            model_type=model_type,
            uncertainty_low=margin_low,
            uncertainty_high=margin_high
        )
        forbidden = set(self.observed_sinkhole_features + ['geometry', 'grid_id'])
        self.stage2_feature_names = [
            c for c in X_train_s2.columns
            if c not in forbidden and not pd.api.types.is_object_dtype(X_train_s2[c])
        ]

        # Evaluate on validation data
        if has_labels:
            final_scores = self.predict(X_val)
            self.final_metrics = compute_comprehensive_metrics(y_val, final_scores, DEFAULT_K_VALS)
            log(f"Final validation metrics: {self.final_metrics}", level=1)

        # Log what kind of model we ended up with
        model_type = "unsupervised (IsolationForest)" if self.is_unsupervised else "supervised (LightGBM)"
        log(f"Final model uses Stage 1: {self.stage1_model_type} and Stage 2: {model_type}", level=2)

        # Log feature importance if available and not unsupervised
        if not self.is_unsupervised and hasattr(self.stage2_model, "feature_importances_"):
            try:
                feat_names = self.stage2_model.booster_.feature_name()
            except:
                feat_names = getattr(self.stage2_model, "feature_name_", None)
            importance = self.stage2_model.feature_importances_
            if feat_names is not None and len(feat_names) == len(importance):
                feat_imp = pd.DataFrame({'feature': feat_names, 'importance': importance})
                feat_imp = feat_imp.sort_values('importance', ascending=False).head(10)
                log(f"Top Stage 2 features: {feat_imp.feature.tolist()}", level=1)
            else:
                log("Skipping feature importance (length mismatch)", level=2)

    # def predict(self, X: pd.DataFrame) -> np.ndarray:
    #     """
    #     Generate combined risk scores using either supervised or unsupervised approach
    #
    #     In unsupervised mode, uses ensemble_weight to combine stage1 and anomaly scores
    #     """
    #     if self.stage1_model is None:
    #         raise RuntimeError("Models must be trained first")
    #
    #     # Stage 1 scores (always available)
    #     stage1_scores = self.stage1_model.predict_proba(X[[self.proximity_feat]])[:, 1]
    #
    #     # GNN 모드일 때 geometry 기반 coords 생성
    #     if isinstance(self.stage2_model, torch.nn.Module):
    #         coords = np.vstack([X.geometry.centroid.x, X.geometry.centroid.y]).T
    #         # PyG Data 생성 생략하고 직접 forward
    #         edge_index = radius_graph(
    #             torch.tensor(coords, dtype=torch.float),
    #             r=GRAPHSAGE_PARAMS.get('radius', 1000), loop=True  # 기본값 1000 제공
    #         ).to(self.device)
    #
    #         x_feat = torch.tensor(
    #             X.drop(columns=self.observed_sinkhole_features + ['grid_id', 'geometry'], errors='ignore').values,
    #             dtype=torch.float, device=self.device
    #         )
    #         # UncertaintyMaskedGraphSAGE는 stage1_scores 텐서 입력
    #         s1 = torch.tensor(stage1_scores, dtype=torch.float, device=self.device)
    #         with torch.no_grad():
    #             s2 = self.stage2_model(x_feat, edge_index, s1).cpu().numpy()
    #         # α-blending
    #         alpha = self.ensemble_weight
    #         return alpha * stage1_scores + (1 - alpha) * s2
    #
    #     # If no stage 2 model, just return stage 1 scores
    #     if self.stage2_model is None:
    #         return stage1_scores
    #
    #     # Generate stage 2 scores using the appropriate model
    #     forbidden = set(self.observed_sinkhole_features + ['geometry', 'grid_id'])
    #     stage2_features = [
    #         c for c in X.columns
    #         if c not in forbidden and not pd.api.types.is_object_dtype(X[c])
    #     ]
    #
    #     if self.is_unsupervised:
    #         # For IsolationForest, get anomaly scores (higher = more anomalous)
    #         X_s2 = X[self.stage2_feature_names]
    #         anomaly_scores = -self.stage2_model.decision_function(X_s2)
    #
    #         # Normalize to [0,1]
    #         min_score, max_score = anomaly_scores.min(), anomaly_scores.max()
    #         if max_score > min_score:
    #             stage2_scores = (anomaly_scores - min_score) / (max_score - min_score)
    #         else:
    #             stage2_scores = np.zeros_like(anomaly_scores)
    #
    #         # Combine using ensemble weight
    #         alpha = self.ensemble_weight
    #         final_scores = alpha * stage1_scores + (1 - alpha) * stage2_scores
    #
    #     else:
    #         # Original supervised approach
    #         candidates = None
    #         if self.threshold_value is not None:
    #             candidates = stage1_scores < self.threshold_value
    #
    #         # 기본 final_scores 를 stage1_scores 로 초기화
    #         final_scores = stage1_scores.copy()
    #
    #         if candidates is not None:
    #             # thresholding 모드
    #             idx = np.where(candidates)[0]
    #             if idx.size > 0:
    #                 s2_preds = self.stage2_model.predict_proba(
    #                     X.iloc[idx][self.stage2_feature_names]
    #                 )[:, 1]
    #                 # α-blend
    #                 final_scores[idx] = s2_preds * (1 - stage1_scores[idx])
    #         else:
    #             # 전체 데이터 예측
    #             s2_preds = self.stage2_model.predict_proba(
    #                 X.drop(columns=forbidden, errors='ignore')[self.stage2_feature_names]
    #             )[:, 1]
    #             final_scores = self.ensemble_weight * stage1_scores + (1 - self.ensemble_weight) * s2_preds
    #
    #     return final_scores

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate combined risk scores using either supervised or unsupervised approach"""
        if self.stage1_model is None:
            raise RuntimeError("Models must be trained first")

        # Stage 1 scores (always available)
        stage1_scores = self.stage1_model.predict_proba(X[[self.proximity_feat]])[:, 1]

        # GNN model mode
        if isinstance(self.stage2_model, torch.nn.Module):
            coords = np.vstack([X.geometry.centroid.x, X.geometry.centroid.y]).T
            edge_index = radius_graph(
                torch.tensor(coords, dtype=torch.float),
                r=GRAPHSAGE_PARAMS.get('radius', 1000), loop=True
            ).to(self.device)

            # FIX: Use exactly the same features as during training
            if hasattr(self, "gnn_features") and self.gnn_features:
                gnn_cols = self.gnn_features
            elif hasattr(self, "stage2_feature_names"):
                gnn_cols = self.stage2_feature_names
            else:
                raise RuntimeError("GNN feature list is not defined")

            # Use only the selected features, ensuring dimensions match
            x_feat = torch.tensor(
                X[gnn_cols].values,
                dtype=torch.float32,
                device=self.device
            )

            # UncertaintyMaskedGraphSAGE needs stage1_scores tensor
            s1 = torch.tensor(stage1_scores, dtype=torch.float, device=self.device)
            with torch.no_grad():
                s2 = self.stage2_model(x_feat, edge_index, s1).cpu().numpy()
            # α-blending
            alpha = self.ensemble_weight
            return alpha * stage1_scores + (1 - alpha) * s2


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
        from nnpu_utils import pseudo_labeling_nnpu
        from evaluation import evaluate_silent_zone, compute_comprehensive_metrics
        from optimizer import optimize_threshold_alpha

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
            unlabeled_weight=0.5,
            method="nnpu"
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

        log(f"Enhanced model with optimized params: threshold_pct={opt_pct}, alpha={opt_alpha}", level=1)
        if 'silent_recall' in best_res:
            log(f"Silent recall@100={best_res['silent_recall']:.4f}", level=1)

        # Final evaluation
        final_scores = self.predict(X_val)
        self.final_metrics = compute_comprehensive_metrics(y_val, final_scores, DEFAULT_K_VALS)

        # Calculate silent zone metrics
        silent_metrics = evaluate_silent_zone(final_scores, silent_val_mask, DEFAULT_K_VALS)
        self.final_metrics.update(silent_metrics)

        log(f"Final enhanced model metrics: {self.final_metrics}", level=1)

    def get_active_learning_suggestions(self, X: pd.DataFrame, silent_grid_ids: set[int],
                                        top_k: int = 100) -> list[int]:
        """
        Get suggestions for grids to investigate next based on active learning principles.
        Returns grid_ids of silent zones with lowest confidence scores.
        """
        from active_learning import select_active_learning_samples

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