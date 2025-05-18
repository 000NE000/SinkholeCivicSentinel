from __future__ import annotations

import os
import time
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.wkb
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import lightgbm as lgb


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}")


def load_dataset(table: str = "feature_matrix_25_geo", chunksize: int = 100_000) -> gpd.GeoDataFrame:
    load_dotenv()
    dsn = os.getenv("DB_DSN")
    if not dsn:
        raise RuntimeError("DB_DSN not set.")
    engine = create_engine(dsn)
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_sql_query(f"SELECT * FROM {table}", engine, chunksize=chunksize):
        chunk["geometry"] = chunk["geom"].apply(lambda h: shapely.wkb.loads(bytes.fromhex(h)))
        parts.append(chunk.drop(columns=["geom"]))
    gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry")
    log(f"Loaded {len(gdf):,} rows; positives {gdf['subsidence_occurrence'].sum()}")
    return gdf

from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import PowerTransformer


def add_kde_risk_score(df: gpd.GeoDataFrame, bandwidth=50, sample_size=10000) -> gpd.GeoDataFrame:
    """
    비지도 방식으로 공간적 밀도 기반 위험 점수 계산
    대용량 데이터셋을 위해 샘플링 적용
    """
    log("  Creating KDE risk score (this may take a moment)...")

    # 샘플링으로 계산 속도 향상
    if len(df) > sample_size:
        sample_idx = np.random.choice(len(df), sample_size, replace=False)
        sample_df = df.iloc[sample_idx]
        # 샘플로 KDE 모델 학습
        coords_sample = np.array([[g.centroid.x, g.centroid.y] for g in sample_df.geometry])
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian').fit(coords_sample)
    else:
        # 전체 데이터로 KDE 모델 학습
        coords = np.array([[g.centroid.x, g.centroid.y] for g in df.geometry])
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian').fit(coords)

    # 전체 데이터에 대한 점수 계산
    coords_all = np.array([[g.centroid.x, g.centroid.y] for g in df.geometry])

    # 계산 속도를 위해 배치 처리
    batch_size = 5000
    log_dens = np.zeros(len(df))

    for i in range(0, len(df), batch_size):
        end_idx = min(i + batch_size, len(df))
        log_dens[i:end_idx] = kde.score_samples(coords_all[i:end_idx])

    # log-density → exp 로 양수화 후 정규화
    risk_score = np.exp(log_dens)
    min_score = risk_score.min()
    range_score = risk_score.max() - min_score

    df['kde_risk_score'] = (risk_score - min_score) / (range_score + 1e-6)

    log(f"  Created: kde_risk_score")
    log(f"    - Min: {df['kde_risk_score'].min():.6f}")
    log(f"    - Max: {df['kde_risk_score'].max():.6f}")
    log(f"    - Mean: {df['kde_risk_score'].mean():.6f}")

    return df


def power_scale(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Box-Cox 변환으로 피처 분포 개선"""
    for col in cols:
        if col in df.columns:
            # 한 컬럼씩 변환 (오류 방지)
            try:
                pt = PowerTransformer(method="yeo-johnson")
                df[col] = pt.fit_transform(df[col].values.reshape(-1, 1)).flatten()
                log(f"  Applied power transform to: {col}")
            except Exception as e:
                log(f"  Warning: Could not transform {col}: {str(e)}")
    return df


def create_safe_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create new features that don't cause label leakage."""
    df = gdf.copy()
    log("Creating safe features (no label information)...")

    # 1. Basic feature - road_soil_stress_diff (no leakage potential)
    df['road_soil_stress_diff'] = df['risk_road_load'] - df['soil_road_stress']
    log(f"  Created: road_soil_stress_diff")
    log(f"    - Min: {df['road_soil_stress_diff'].min():.6f}")
    log(f"    - Max: {df['road_soil_stress_diff'].max():.6f}")
    log(f"    - Mean: {df['road_soil_stress_diff'].mean():.6f}")

    # 2. Geological-Infrastructure Stress Interaction
    # Combine geological risk factors with infrastructure load factors
    geological_cols = [col for col in df.columns if 'risk_' in col and col not in ['risk_road_load']]
    if geological_cols:
        # Normalize values using sigmoid function to avoid extreme values
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Create geological vulnerability index
        geo_features = []
        for col in geological_cols:
            if col in df.columns:
                # Standard scale before sigmoid to center values
                scaled_values = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)
                geo_features.append(scaled_values)

        if geo_features:
            # Combine geological features
            df['geological_vulnerability'] = sigmoid(sum(geo_features))

            # Create infrastructure load index
            infra_cols = ['risk_road_load', 'soil_road_stress']
            infra_features = []
            for col in infra_cols:
                if col in df.columns:
                    scaled_values = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)
                    infra_features.append(scaled_values)

            if infra_features:
                df['infrastructure_load'] = sigmoid(sum(infra_features))

                # Create interaction feature
                df['geo_infra_interaction'] = df['geological_vulnerability'] * df['infrastructure_load']
                log(f"  Created: geo_infra_interaction")
                log(f"    - Min: {df['geo_infra_interaction'].min():.6f}")
                log(f"    - Max: {df['geo_infra_interaction'].max():.6f}")
                log(f"    - Mean: {df['geo_infra_interaction'].mean():.6f}")

    # 3. Create spatial clustering feature (without using label information)
    # This uses DBSCAN to find natural clusters in the data
    coords = np.array([[g.centroid.x, g.centroid.y] for g in df.geometry])

    # Run DBSCAN clustering
    clustering = DBSCAN(eps=0.01, min_samples=3).fit(coords)
    df['spatial_cluster'] = clustering.labels_

    # Calculate cluster sizes
    cluster_sizes = df.groupby('spatial_cluster').size()
    df['cluster_size'] = df['spatial_cluster'].map(cluster_sizes)

    # Create a feature based on cluster size and distance to cluster center
    # This captures spatial density information without using label information
    cluster_centers = {}
    for cluster_id in df['spatial_cluster'].unique():
        if cluster_id != -1:  # Skip noise points
            cluster_points = coords[df['spatial_cluster'] == cluster_id]
            cluster_centers[cluster_id] = np.mean(cluster_points, axis=0)

    # Calculate distance to cluster center
    df['dist_to_cluster_center'] = np.nan
    for idx, row in df.iterrows():
        cluster_id = row['spatial_cluster']
        if cluster_id != -1 and cluster_id in cluster_centers:
            center = cluster_centers[cluster_id]
            point = [row.geometry.centroid.x, row.geometry.centroid.y]
            distance = np.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
            df.at[idx, 'dist_to_cluster_center'] = distance

    # Create a spatial density feature
    # Calculate distances for all points, not just cluster members
    df['dist_to_cluster_center'] = df['dist_to_cluster_center'].fillna(
        df['dist_to_cluster_center'].mean() * 2)  # Use higher value for outliers

    # Use cluster size directly for noise points (-1) to avoid division issues
    df['spatial_density'] = np.where(
        df['spatial_cluster'] != -1,
        df['cluster_size'] / (df['dist_to_cluster_center'] + 1e-6),
        df['cluster_size'].min() / 10  # Small value for noise points
    )

    # Normalize and log transform
    max_density = df['spatial_density'].max()
    df['spatial_density_norm'] = df['spatial_density'] / (max_density + 1e-6)
    df['spatial_density_log'] = np.log1p(df['spatial_density_norm'] * 100)
    log(f"  Created: spatial_density_log")
    log(f"    - Min: {df['spatial_density_log'].min():.6f}")
    log(f"    - Max: {df['spatial_density_log'].max():.6f}")
    log(f"    - Mean: {df['spatial_density_log'].mean():.6f}")

    # 4. Apply Box-Cox scaling to improve feature distributions
    scaled_cols = ['road_soil_stress_diff', 'geo_infra_interaction', 'spatial_density_log']
    df = power_scale(df, scaled_cols)

    # 5. Add KDE risk score (non-supervised spatial density)
    df = add_kde_risk_score(df)

    return df


def create_cv_fold_features(X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, geo_train: gpd.GeoSeries,
                            geo_val: gpd.GeoSeries) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create features using only the training fold's labels to avoid leakage.
    These features should only be used during CV, not for final prediction.
    """
    log("Creating fold-specific features (using only training fold labels)...")

    # Create copies for modification
    X_train_new = X_train.copy()
    X_val_new = X_val.copy()

    # 1. Distance-Density Weighted Indicator
    # Create a safe_risk_distance_ratio that uses only training data labels
    # First identify positive and negative samples in training set
    safe_mask = y_train == 0
    risk_mask = y_train == 1

    # Only proceed if we have both positive and negative examples in training set
    if risk_mask.sum() > 0 and safe_mask.sum() > 0:
        # Convert training geometries to coordinate arrays for faster distance calculation
        safe_coords = np.array([[g.centroid.x, g.centroid.y] for g in geo_train[safe_mask]])
        risk_coords = np.array([[g.centroid.x, g.centroid.y] for g in geo_train[risk_mask]])

        # Calculate distances for training set
        safe_nn = NearestNeighbors(n_neighbors=1).fit(safe_coords)
        risk_nn = NearestNeighbors(n_neighbors=1).fit(risk_coords)

        # Training set distances
        train_coords = np.array([[g.centroid.x, g.centroid.y] for g in geo_train])
        train_dist_to_safe = safe_nn.kneighbors(train_coords, return_distance=True)[0].flatten()
        train_dist_to_risk = risk_nn.kneighbors(train_coords, return_distance=True)[0].flatten()

        # Validation set distances
        val_coords = np.array([[g.centroid.x, g.centroid.y] for g in geo_val])
        val_dist_to_safe = safe_nn.kneighbors(val_coords, return_distance=True)[0].flatten()
        val_dist_to_risk = risk_nn.kneighbors(val_coords, return_distance=True)[0].flatten()

        # Calculate ratio (add small epsilon to avoid division by zero)
        epsilon = 1e-6
        train_ratio = np.log1p(train_dist_to_risk / (train_dist_to_safe + epsilon))
        val_ratio = np.log1p(val_dist_to_risk / (val_dist_to_safe + epsilon))

        # Scale between 0 and 1 using training set min/max
        min_ratio = train_ratio.min()
        max_ratio = train_ratio.max()
        range_ratio = max_ratio - min_ratio

        if range_ratio > 0:
            # Apply soft scaling (0.2 factor reduces the impact of this feature)
            train_soft_ratio = 0.2 * (train_ratio - min_ratio) / range_ratio
            val_soft_ratio = 0.2 * (val_ratio - min_ratio) / range_ratio

            # Add to dataframes
            X_train_new['safe_risk_distance_ratio'] = train_soft_ratio
            X_val_new['safe_risk_distance_ratio'] = val_soft_ratio

            log(f"  Created fold-specific: safe_risk_distance_ratio")
            log(f"    - Train Mean: {train_soft_ratio.mean():.6f}")
            log(f"    - Val Mean: {val_soft_ratio.mean():.6f}")

    return X_train_new, X_val_new


def calculate_sample_weights(X_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
    """Calculate sample weights with simplified approach."""
    w = np.ones(len(y_train))

    # High risk areas (high road load and soil stress)
    if 'risk_road_load' in X_train.columns and 'soil_road_stress' in X_train.columns:
        hi_load = (X_train['risk_road_load'] > X_train['risk_road_load'].quantile(0.8)) & \
                  (X_train['soil_road_stress'] > X_train['soil_road_stress'].quantile(0.8))
        w[hi_load] *= 0.3  # Reduce weight for potential FP areas

    # Increase weights for positive samples
    w[y_train == 1] *= 2.0

    # Normalize weights
    return w / w.mean()


def train_lightgbm_with_blending(X_train: pd.DataFrame, y_train: pd.Series,
                                 X_val: pd.DataFrame, sample_weights: np.ndarray = None) -> Tuple[
    lgb.LGBMClassifier, Dict]:
    """
    Train a LightGBM model with score blending to reduce false positives.
    Returns the model and a dictionary of information for blending.
    """
    # Basic model parameters
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "max_depth": 7,
        "verbose": -1,
        "scale_pos_weight": 300,
        "feature_name": 'auto'  # 자동으로 피처 이름 저장
    }

    # Train the model
    model = lgb.LGBMClassifier(**params)

    if sample_weights is not None:
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    # Get feature importances
    importance = dict(zip(X_train.columns, model.feature_importances_))

    # Store feature names for future prediction
    feature_names = list(X_train.columns)

    # Prepare blending info
    blending_info = {
        "model": model,
        "importance": importance,
        "feature_names": feature_names
    }

    # If we have road_soil_stress_diff, calculate threshold for blending
    if 'road_soil_stress_diff' in X_train.columns:
        # Get high percentile to use as threshold
        blending_info["stress_diff_threshold"] = np.percentile(X_train['road_soil_stress_diff'], 75)
        blending_info["stress_diff_weight"] = 0.15  # Penalty weight for high stress diff

    return model, blending_info

def blend_predictions(X: pd.DataFrame, base_preds: np.ndarray, blending_info: Dict) -> np.ndarray:
    """
    Apply score blending to reduce false positives.
    Higher road_soil_stress_diff values get a penalty.
    """
    # Start with base predictions
    blended_preds = base_preds.copy()

    # Apply road_soil_stress_diff penalty if available
    if ('road_soil_stress_diff' in X.columns and
            'stress_diff_threshold' in blending_info and
            'stress_diff_weight' in blending_info):

        # Identify samples with high stress difference (potential FP)
        high_stress_mask = X['road_soil_stress_diff'] > blending_info['stress_diff_threshold']

        # Calculate penalty based on how far they exceed the threshold
        if high_stress_mask.sum() > 0:
            penalty = (X.loc[high_stress_mask, 'road_soil_stress_diff'] -
                       blending_info['stress_diff_threshold']) / X['road_soil_stress_diff'].max()

            # Apply penalty to predictions
            penalty_factor = 1.0 - (blending_info['stress_diff_weight'] * penalty)
            blended_preds[high_stress_mask] *= penalty_factor.values

    return blended_preds


def topk_metrics(y_true, scores, ks=[100, 200, 500]):
    """
    Calculate recall, precision, and other metrics for top-K predictions.
    Consistent implementation for use throughout the code.
    """
    results = {}
    y_arr = np.array(y_true)
    score_arr = np.array(scores)

    # Sort by score (descending)
    idx = np.argsort(score_arr)[::-1]
    sorted_y = y_arr[idx]

    # Calculate metrics for each k
    for k in ks:
        # Ensure k is not larger than the dataset
        k = min(k, len(y_true))

        # Get top-k predictions
        top_k_y = sorted_y[:k]

        # Calculate metrics
        total_pos = y_arr.sum()
        if total_pos > 0:
            recall = top_k_y.sum() / total_pos
        else:
            recall = 0.0

        precision = top_k_y.sum() / k if k > 0 else 0.0

        # Store metrics
        results[f"recall_{k}"] = recall
        results[f"precision_{k}"] = precision

    return results


def evaluate_model(model, X, y, features, blending_info=None):
    """
    Evaluate a model with consistent metrics.
    Returns a dictionary of evaluation metrics.
    """
    # 모델이 훈련된 피처 목록과 예측 시 사용할 피처 목록 간의 불일치 처리
    if hasattr(model, 'feature_name_'):
        trained_features = model.feature_name_
        # 훈련에 사용된 모든 피처가 있는지 확인
        missing_features = [f for f in trained_features if f not in X.columns]
        if missing_features:
            log(f"  Warning: Features missing for prediction: {missing_features}")
            # 누락된 피처를 0으로 채움
            for feat in missing_features:
                X[feat] = 0

        # 모델이 훈련된 피처 순서대로 데이터 전달
        predict_X = X[trained_features]
    else:
        # 피처 이름 정보가 없으면 전달받은 피처 그대로 사용
        predict_X = X[features]

    # Get raw predictions
    try:
        # predict_disable_shape_check=True 옵션 추가
        raw_preds = model.predict_proba(predict_X, predict_disable_shape_check=True)[:, 1]
    except TypeError:
        # 이전 버전의 LightGBM은 predict_disable_shape_check를 지원하지 않을 수 있음
        raw_preds = model.predict_proba(predict_X)[:, 1]

    # Apply blending if available
    if blending_info:
        preds = blend_predictions(X, raw_preds, blending_info)
    else:
        preds = raw_preds

    # Calculate AUC
    auc = roc_auc_score(y, preds)

    # Calculate top-k metrics
    topk = topk_metrics(y, preds, ks=[100, 200, 500])

    # Calculate FP/FN rates (using 0.5 threshold)
    threshold = 0.5
    fp_mask = (y == 0) & (preds > threshold)
    fn_mask = (y == 1) & (preds <= threshold)

    fp_count = fp_mask.sum()
    fn_count = fn_mask.sum()

    fp_rate = fp_count / len(y)
    fn_rate = fn_count / len(y)

    # Combine all metrics
    metrics = {
        "auc": auc,
        **topk,
        "fp_count": fp_count,
        "fn_count": fn_count,
        "fp_rate": fp_rate,
        "fn_rate": fn_rate
    }

    return metrics, preds


def run_cv_fold(X, y, geo_series, base_features, new_features, fold_idx, train_idx, val_idx):
    """
    Run a single cross-validation fold, training both base and enhanced models.
    Returns evaluation metrics for both models.
    """
    # Split data for this fold
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    geo_train_fold, geo_val_fold = geo_series.iloc[train_idx], geo_series.iloc[val_idx]

    # Create fold-specific features (using only training labels to avoid leakage)
    X_train_fold_enhanced, X_val_fold_enhanced = create_cv_fold_features(
        X_train_fold, y_train_fold, X_val_fold, geo_train_fold, geo_val_fold
    )

    # Check for CV-specific features created
    cv_specific_features = [col for col in X_train_fold_enhanced.columns
                            if col not in X_train_fold.columns]

    # Calculate sample weights
    sample_weights = calculate_sample_weights(X_train_fold_enhanced, y_train_fold)

    # Train base model (without new features)
    base_model, _ = train_lightgbm_with_blending(
        X_train_fold[base_features], y_train_fold,
        X_val_fold[base_features], sample_weights
    )

    # Evaluate base model
    base_metrics, base_preds = evaluate_model(
        base_model, X_val_fold, y_val_fold, base_features
    )

    enhanced_features = base_features + new_features + cv_specific_features

    # Ensure consistent feature sets for training and prediction
    X_train_enhanced = X_train_fold_enhanced[enhanced_features].copy()
    X_val_enhanced = X_val_fold_enhanced[enhanced_features].copy()

    # Filter out CV-specific features for production
    production_features = [f for f in enhanced_features if not f.startswith('safe_')]

    enhanced_model, blending_info = train_lightgbm_with_blending(
        X_train_enhanced, y_train_fold,
        X_val_enhanced, sample_weights
    )

    # Evaluate enhanced model with proper feature handling
    enhanced_metrics, enhanced_preds = evaluate_model(
        enhanced_model, X_val_fold_enhanced, y_val_fold,
        enhanced_features, blending_info  # 전체 enhanced_features 전달
    )
    # Log results for this fold
    log(f"  Fold {fold_idx + 1}:")
    log(f"    Base Model: AUC={base_metrics['auc']:.4f}, Recall@100={base_metrics['recall_100']:.4f}")
    log(f"    Base Model: FP_rate={base_metrics['fp_rate']:.2%}, FN_rate={base_metrics['fn_rate']:.2%}")
    log(f"    Enhanced Model: AUC={enhanced_metrics['auc']:.4f}, Recall@100={enhanced_metrics['recall_100']:.4f}")
    log(f"    Enhanced Model: FP_rate={enhanced_metrics['fp_rate']:.2%}, FN_rate={enhanced_metrics['fn_rate']:.2%}")

    # Log feature importance for enhanced model
    importance = blending_info['importance']
    log("    Top feature importances:")
    for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
        log(f"      - {feat}: {imp:.6f}")

    # Log new features importance
    for feat in new_features + cv_specific_features:
        if feat in importance:
            rank = sorted(importance.keys(), key=lambda x: importance[x], reverse=True).index(feat) + 1
            percentile = (len(importance) - rank) / len(importance) * 100
            log(f"      - {feat}: {importance[feat]:.6f} (Rank {rank}/{len(importance)}, {percentile:.1f}th percentile)")

    return {
        "base": base_metrics,
        "enhanced": enhanced_metrics,
        "importance": importance,
        "cv_specific_features": cv_specific_features,
        "production_features": production_features
    }


def check_for_leakage(gdf: gpd.GeoDataFrame, y: pd.Series) -> List[str]:
    """
    Check for potential label leakage using stratified sampling.
    Returns list of features to remove.
    """
    log("Checking for potential label leakage...")

    # Use stratified sampling to ensure we have sufficient positives
    X_sample, _, y_sample, _ = train_test_split(
        gdf, y, stratify=y, train_size=min(10000, len(gdf)), random_state=42
    )

    all_columns = [col for col in X_sample.columns if col not in ['grid_id', 'subsidence_occurrence', 'geometry']]
    potential_leakage = []

    for col in all_columns:
        try:
            auc = roc_auc_score(y_sample, X_sample[col])
            if auc > 0.95:
                potential_leakage.append((col, auc))
            elif auc < 0.05:
                potential_leakage.append((col, 1 - auc))
        except:
            pass

    # Sort by AUC descending
    potential_leakage.sort(key=lambda x: x[1], reverse=True)

    if potential_leakage:
        log("Potential label leakage detected in these features:")
        for col, auc in potential_leakage[:5]:
            log(f"  {col}: AUC={auc:.4f}")

    # Features to remove
    features_to_remove = ['subsidence_count']
    for col, auc in potential_leakage:
        if auc > 0.98:  # Very strict threshold for automatic removal
            features_to_remove.append(col)

    return list(set(features_to_remove))


def main():
    t0 = time.time()

    # Load dataset
    gdf = load_dataset()

    # Create safe features (no label information used)
    gdf = create_safe_features(gdf)

    # Define new features
    new_features = ['road_soil_stress_diff', 'geo_infra_interaction',
                    'spatial_density_log', 'kde_risk_score']

    # Extract target
    y = gdf['subsidence_occurrence'].astype(int)

    # Check for label leakage
    features_to_remove = check_for_leakage(gdf, y)
    log(f"Removing {len(features_to_remove)} features with potential label leakage: {features_to_remove}")

    # Drop leaked features
    X = gdf.drop(columns=['grid_id', 'subsidence_occurrence', 'geometry'] + features_to_remove)

    # Update new_features to only include those that weren't removed
    new_features = [f for f in new_features if f in X.columns]
    log(f"New features after removing leakage: {new_features}")

    if not new_features:
        log("WARNING: All new features were removed due to potential label leakage!")
        return

    # Base features (exclude new features and fn_risk_interaction)
    base_features = [col for col in X.columns if col not in ['fn_risk_interaction'] + new_features]

    # Use cross-validation for robust evaluation
    log("Running 5-fold cross-validation with fold-specific feature engineering:")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Dictionaries to store CV results
    cv_results = []

    # Run CV
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        fold_results = run_cv_fold(
            X, y, gdf.geometry, base_features, new_features,
            fold_idx, train_idx, val_idx
        )
        cv_results.append(fold_results)

    # Aggregate CV results
    base_metrics = {
        "auc": [],
        "recall_100": [],
        "fp_rate": [],
        "fn_rate": []
    }

    enhanced_metrics = {
        "auc": [],
        "recall_100": [],
        "fp_rate": [],
        "fn_rate": []
    }

    # Feature importance across folds
    feature_importances = {}

    for fold_result in cv_results:
        # Collect metrics
        for metric in base_metrics:
            if metric in fold_result["base"]:
                base_metrics[metric].append(fold_result["base"][metric])

        for metric in enhanced_metrics:
            if metric in fold_result["enhanced"]:
                enhanced_metrics[metric].append(fold_result["enhanced"][metric])

        # Collect feature importances
        importance = fold_result["importance"]
        for feature, imp in importance.items():
            if feature not in feature_importances:
                feature_importances[feature] = []
            feature_importances[feature].append(imp)

    # Calculate mean and std of metrics
    base_summary = {
        metric: {
            "mean": np.mean(values),
            "std": np.std(values)
        }
        for metric, values in base_metrics.items()
    }

    enhanced_summary = {
        metric: {
            "mean": np.mean(values),
            "std": np.std(values)
        }
        for metric, values in enhanced_metrics.items()
    }

    # Calculate average feature importance
    avg_importance = {
        feature: np.mean(values)
        for feature, values in feature_importances.items()
    }

    # Log CV summary
    log("\nCross-validation Summary:")
    log(f"Base Model: AUC={base_summary['auc']['mean']:.4f}±{base_summary['auc']['std']:.4f}, "
        f"Recall@100={base_summary['recall_100']['mean']:.4f}±{base_summary['recall_100']['std']:.4f}")
    log(f"Base Model: FP_rate={base_summary['fp_rate']['mean']:.2%}±{base_summary['fp_rate']['std']:.2%}, "
        f"FN_rate={base_summary['fn_rate']['mean']:.2%}±{base_summary['fn_rate']['std']:.2%}")

    log(f"Enhanced Model: AUC={enhanced_summary['auc']['mean']:.4f}±{enhanced_summary['auc']['std']:.4f}, "
        f"Recall@100={enhanced_summary['recall_100']['mean']:.4f}±{enhanced_summary['recall_100']['std']:.4f}")
    log(f"Enhanced Model: FP_rate={enhanced_summary['fp_rate']['mean']:.2%}±{enhanced_summary['fp_rate']['std']:.2%}, "
        f"FN_rate={enhanced_summary['fn_rate']['mean']:.2%}±{enhanced_summary['fn_rate']['std']:.2%}")

    # Calculate improvements
    auc_improvement = enhanced_summary['auc']['mean'] - base_summary['auc']['mean']
    recall_improvement = enhanced_summary['recall_100']['mean'] - base_summary['recall_100']['mean']
    fp_reduction = (base_summary['fp_rate']['mean'] - enhanced_summary['fp_rate']['mean']) / base_summary['fp_rate'][
        'mean']
    fn_reduction = (base_summary['fn_rate']['mean'] - enhanced_summary['fn_rate']['mean']) / base_summary['fn_rate'][
        'mean']

    log("\nOverall Improvements:")
    log(f"  AUC: {auc_improvement:+.4f} ({(auc_improvement / base_summary['auc']['mean']) * 100:+.2f}%)")
    log(f"  Recall@100: {recall_improvement:+.4f} ({(recall_improvement / base_summary['recall_100']['mean']) * 100:+.2f}%)")
    log(f"  FP Reduction: {fp_reduction:.2%}")
    log(f"  FN Reduction: {fn_reduction:.2%}")

    # Feature importance analysis
    log("\nFeature Importance Analysis:")

    # Sort features by average importance
    sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

    # Log top 10 features
    log("Top 10 features by importance:")
    for feature, importance in sorted_features[:10]:
        log(f"  {feature}: {importance:.6f}")

    # Analyze new features specifically
    log("\nNew Feature Importance:")
    for feature in new_features:
        if feature in avg_importance:
            # Calculate feature rank
            rank = [f for f, _ in sorted_features].index(feature) + 1
            percentile = (len(sorted_features) - rank) / len(sorted_features) * 100
            log(f"  {feature}: {avg_importance[feature]:.6f} (Rank {rank}/{len(sorted_features)}, {percentile:.1f}th percentile)")

    # Final conclusion with improved decision logic
    log("\nFINAL CONCLUSION:")

    # Consider both metric improvements and FP/FN reduction
    if auc_improvement > 0 and recall_improvement > 0 and (fp_reduction > 0.05 or fn_reduction > 0.05):
        log("✓ The new features provide meaningful performance improvements")

        # Categorize features by their impact
        strong_features = []
        moderate_features = []

        for feature in new_features:
            if feature in avg_importance:
                importance = avg_importance[feature]
                rank = [f for f, _ in sorted_features].index(feature) + 1
                percentile = (len(sorted_features) - rank) / len(sorted_features) * 100
                # Classify feature strength based on importance and percentile
                if importance > 0.01 or percentile > 75:
                    strong_features.append(feature)
                else:
                    moderate_features.append(feature)
        if strong_features:
            log(f"✓ Strongly recommended features: {strong_features}")
        if moderate_features:
            log(f"⚠ Consider including these features: {moderate_features}")

            # Detailed recommendations
        log("\nImplementation Strategy:")
        log("1. For production models:")
        log("   - Include the strongly recommended features")
        log("   - Apply score blending to reduce false positives using road_soil_stress_diff penalty")

        if fp_reduction > 0.05:
            log(f"   - Significant FP reduction ({fp_reduction:.2%}) observed - implement blending")

        if fn_reduction > 0.05:
            log(f"   - Significant FN reduction ({fn_reduction:.2%}) observed - consider using fold-specific")
            log("     features with sample weighting during model training")

        log("2. For cross-validation and model training:")
        log("   - Use the fold-specific safe distance-ratio features for CV training only")
        log("   - Apply sample weighting to emphasize FN-prone areas")

        log("3. For operational use:")
        log("   - Implement dual threshold approach with:")
        log("     a) Primary prediction from the model")
        log("     b) Secondary rule-based adjustment using road_soil_stress_diff")
    elif auc_improvement > 0 and recall_improvement > 0:
        log("⚠ The new features show metric improvements but without significant FP/FN reduction")
        log("  Consider including only if computational cost is minimal")
    else:
        log("✗ The new features do not consistently improve important metrics")
        log("  Do not include these features in the production model")

    log(f"\nDone in {time.time() - t0:.1f}s")

if __name__ == '__main__':
   main()