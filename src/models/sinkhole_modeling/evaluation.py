"""
Evaluation metrics for sinkhole model
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Sequence, Optional, Tuple
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score
from sklearn.model_selection import GroupKFold, train_test_split
from src.models.sinkhole_modeling.config import log, DEFAULT_K_VALS


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


def evaluate_silent_zone(scores: np.ndarray,
                         silent_mask: np.ndarray,
                         k_vals: List[int] = DEFAULT_K_VALS) -> Dict[str, float]:
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


def evaluate_uncertainty_masking(
        scores: np.ndarray,
        y_true: np.ndarray,
        low_threshold: float = 0.4,
        high_threshold: float = 0.6
) -> Dict[str, float]:
    """
    Evaluate the effectiveness of uncertainty masking

    Args:
        scores: Prediction scores
        y_true: True labels
        low_threshold: Lower threshold for uncertainty region
        high_threshold: Upper threshold for uncertainty region

    Returns:
        Dictionary of uncertainty metrics
    """
    # Calculate uncertainty mask
    uncertainty_mask = (scores >= low_threshold) & (scores <= high_threshold)

    # Calculate metrics
    uncertainty_ratio = uncertainty_mask.mean()
    uncertain_pos_ratio = y_true[uncertainty_mask].mean() if uncertainty_mask.sum() > 0 else 0
    certain_pos_ratio = y_true[~uncertainty_mask].mean() if (~uncertainty_mask).sum() > 0 else 0

    # Information gain from masking
    entropy_before = -(y_true.mean() * np.log(y_true.mean() + 1e-10) +
                       (1 - y_true.mean()) * np.log(1 - y_true.mean() + 1e-10))

    entropy_after = 0
    if uncertainty_mask.sum() > 0:
        p_uncertain = uncertainty_mask.mean()
        p_u_pos = y_true[uncertainty_mask].mean()
        entropy_uncertain = -(p_u_pos * np.log(p_u_pos + 1e-10) +
                              (1 - p_u_pos) * np.log(1 - p_u_pos + 1e-10))
        entropy_after += p_uncertain * entropy_uncertain

    if (~uncertainty_mask).sum() > 0:
        p_certain = (~uncertainty_mask).mean()
        p_c_pos = y_true[~uncertainty_mask].mean()
        entropy_certain = -(p_c_pos * np.log(p_c_pos + 1e-10) +
                            (1 - p_c_pos) * np.log(1 - p_c_pos + 1e-10))
        entropy_after += p_certain * entropy_certain

    information_gain = entropy_before - entropy_after

    return {
        'uncertainty_ratio': uncertainty_ratio,
        'uncertain_pos_ratio': uncertain_pos_ratio,
        'certain_pos_ratio': certain_pos_ratio,
        'information_gain': information_gain
    }


def evaluate_spatial_cv(
        gdf: gpd.GeoDataFrame,
        n_folds: int = 5,
        k_vals: List[int] = DEFAULT_K_VALS,
        silent_grid_ids: Optional[set[int]] = None,
        include_silent_metrics: bool = False,
        model_type: str = "lightgbm",
        use_uncertainty_masking: bool = False,
) -> Dict:
    """
    Evaluate model using spatial cross-validation

    Args:
        gdf: GeoDataFrame with data and geometry
        n_folds: Number of spatial folds
        k_vals: List of K values for topK metrics
        silent_grid_ids: Optional set of grid IDs for silent zones
        include_silent_metrics: Whether to include silent zone metrics
        model_type: Model type to use (lightgbm, gnn, um_gnn)
        use_uncertainty_masking: Whether to use uncertainty masking

    Returns:
        Dictionary with evaluation metrics
    """
    from src.models.sinkhole_modeling.data_loader import create_spatial_blocks
    from src.models.sinkhole_modeling.stage1_model import TwoStageSinkholeScreener

    log(f"Performing {n_folds}-fold spatial cross-validation", level=1)

    # Prepare data - 원본 GeoDataFrame 유지하고 필요한 열만 제거
    y = gdf["subsidence_occurrence"].astype(int)
    X = gdf.copy()  # 모든 열을 포함하여 복사 (geometry 포함)

    # subsidence_occurrence와 subsidence_count만 제거 (geometry는 유지)
    X = X.drop(columns=["subsidence_occurrence", "subsidence_count"], errors="ignore")

    # Keep grid_id for silent zone metrics if needed
    if include_silent_metrics and 'grid_id' not in X.columns and 'grid_id' in gdf.columns:
        X['grid_id'] = gdf['grid_id']

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

        # Train two-stage model with selected options
        screener = TwoStageSinkholeScreener(
            proximity_feat="min_distance_to_sinkhole",
            threshold_percentile=None,  # Auto-tune
            use_focal_loss=True,
            feature_fraction=0.6,
            stage2_model_type=model_type,
            margin_low=0.4,
            margin_high=0.6
        )

        # 모델 학습 - X_train에는 geometry 열이 포함되어 있음
        screener.fit(X_train, y_train, spatial_blocks=blocks_train, model_type=model_type)
        test_scores = screener.predict(X_test)

        # Evaluate basic metrics
        metrics = compute_comprehensive_metrics(y_test, test_scores, k_vals)

        # Evaluate uncertainty masking if used
        if use_uncertainty_masking and hasattr(screener, 'get_stage1_scores'):
            try:
                uncertainty_metrics = evaluate_uncertainty_masking(
                    screener.get_stage1_scores(X_test),
                    y_test.values,
                    getattr(screener, 'uncertainty_threshold_low', 0.4),
                    getattr(screener, 'uncertainty_threshold_high', 0.6)
                )
                metrics.update({f"uncertainty_{k}": v for k, v in uncertainty_metrics.items()})
            except Exception as e:
                log(f"Error evaluating uncertainty: {str(e)}", level=2)

        # Only calculate silent metrics if specifically requested
        if include_silent_metrics and silent_grid_ids and 'grid_id' in gdf.columns:
            silent_mask = gdf.iloc[test_idx]["grid_id"].isin(silent_grid_ids).values
            silent_metrics = evaluate_silent_zone(test_scores, silent_mask, k_vals)
            metrics.update(silent_metrics)

        metrics["fold"] = fold
        # Store metrics
        fold_metrics.append(metrics)

        log_msg = f"[Fold{fold + 1}] IoU@100={metrics['iou@100']:.3f}"
        if 'uncertainty_information_gain' in metrics:
            log_msg += f", InfoGain={metrics['uncertainty_information_gain']:.3f}"
        log(log_msg, level=1)

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


def evaluate_reports_impact_pu(
        report_tables: List[str],
        k_vals: List[int] = DEFAULT_K_VALS,
        n_iters: int = 5,
        include_silent_metrics: bool = False,
        use_graphsage: bool = False,
        use_uncertainty_masking: bool = False,
        random_state: int = 42
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    가상의 레이블 제약 상황(Positive-Unlabeled)에서
    report 수별 성능 변화를 시뮬레이션 평가

    Args:
        report_tables: List of table names with different report counts
        k_vals: List of K values for evaluation
        n_iters: Number of iterations for simulation
        include_silent_metrics: Whether to include silent zone metrics
        use_graphsage: Whether to use GraphSAGE for Stage 2
        use_uncertainty_masking: Whether to use uncertainty masking
        random_state: Random seed

    Returns:
        Dictionary with metrics per report count
    """
    from src.models.sinkhole_modeling.data_loader import load_dataset, get_silent_grid_ids

    rng = np.random.default_rng(random_state)
    agg_results = {}
    log(f"Starting PU-learning simulation across {len(report_tables)} report counts", level=1)
    model_type = "um_gnn" if use_graphsage else "lightgbm"
    log(f"PU evaluation with use_graphsage={use_graphsage}, resulting in model_type={model_type}", level=1)

    for table in report_tables:
        report_count = int(''.join(filter(str.isdigit, table)))
        log(f"Simulating with n_reports={report_count}...", level=1)

        # 원본 데이터 & 진짜 레이블
        gdf = load_dataset(table=table)
        X_full = gdf.drop(columns=['grid_id', 'geometry', 'subsidence_occurrence', 'subsidence_count'])
        y_full = gdf['subsidence_occurrence'].astype(int).values
        pos_idx = np.where(y_full == 1)[0]

        n_pos_total = len(pos_idx)

        # Generate silent grid IDs dynamically if needed
        silent_ids = None
        if include_silent_metrics:
            try:
                silent_ids = get_silent_grid_ids(gdf, gdf['subsidence_occurrence'])
                log(f"Using {len(silent_ids)} dynamically generated silent zones", level=2)
            except Exception as e:
                log(f"Could not generate silent zones: {str(e)}", level=2)

        # 반복마다 metrics 수집
        metrics_accum = {}
        for k in k_vals:
            for m in ['recall', 'precision', 'lift']:
                metrics_accum[f'{m}@{k}'] = []

        metrics_accum['pr_auc'] = []

        if use_uncertainty_masking:
            for m in ['uncertainty_ratio', 'information_gain']:
                metrics_accum[f'uncertainty_{m}'] = []

        for it in range(n_iters):
            # 1) 랜덤으로 report_count 개만 positive 라벨로 남김
            sampled = rng.choice(pos_idx, size=min(report_count, n_pos_total), replace=False)
            y_pu = np.zeros_like(y_full)
            y_pu[sampled] = 1

            # 2) Spatial CV로 평가
            gdf_pu = gdf.copy()
            gdf_pu['subsidence_occurrence'] = y_pu
            cv_res = evaluate_spatial_cv(
                gdf_pu,
                n_folds=5,
                k_vals=k_vals,
                silent_grid_ids=silent_ids,
                include_silent_metrics=include_silent_metrics,
                model_type=model_type,
                use_uncertainty_masking=use_uncertainty_masking
            )

            # Collect multiple metrics
            for k in k_vals:
                for m in ['recall', 'precision', 'lift']:
                    key = f'{m}@{k}'
                    metrics_accum[key].append(cv_res['aggregate'][key]['mean'])

            # PR-AUC
            metrics_accum['pr_auc'].append(cv_res['aggregate']['pr_auc']['mean'])

            # Uncertainty metrics
            if use_uncertainty_masking:
                for m in ['uncertainty_ratio', 'information_gain']:
                    key = f'uncertainty_{m}'
                    if key in cv_res['aggregate']:
                        metrics_accum[key].append(cv_res['aggregate'][key]['mean'])

            # Silent metrics if available and requested
            if include_silent_metrics:
                for k in k_vals:
                    silent_key = f'silent_recall@{k}'
                    if silent_key in cv_res['aggregate']:
                        metrics_accum.setdefault(silent_key, []).append(
                            cv_res['aggregate'][silent_key]['mean'])

        # 평균·표준편차 계산
        summary = {}
        # Process all metrics
        for key, values in metrics_accum.items():
            arr = np.array(values)
            summary[f'{key}_mean'] = arr.mean()
            summary[f'{key}_std'] = arr.std()

        agg_results[report_count] = summary
        log(f"n_reports={report_count} → " +
            ", ".join([f"recall@{k}: {summary[f'recall@{k}_mean']:.3f}±{summary[f'recall@{k}_std']:.3f}"
                       for k in k_vals]))
    return agg_results


def plot_pu_report_impact(results: Dict, k_vals: List[int] = DEFAULT_K_VALS,
                          include_silent_metrics: bool = False,
                          include_uncertainty_metrics: bool = False) -> None:
    """
    Plot how metrics improve with increasing number of reports in PU-learning simulation

    Args:
        results: Dictionary with results from evaluate_reports_impact_pu
        k_vals: List of K values to plot
        include_silent_metrics: Whether to include silent zone metrics
        include_uncertainty_metrics: Whether to include uncertainty metrics
    """
    import matplotlib.pyplot as plt

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

        # Add silent recall if requested and available
        if include_silent_metrics and f'silent_recall@{k}_mean' in results[report_counts[0]]:
            silent_means = [results[count][f'silent_recall@{k}_mean'] for count in report_counts]
            silent_stds = [results[count][f'silent_recall@{k}_std'] for count in report_counts]

            axs[i].errorbar(report_counts, silent_means, yerr=silent_stds,
                            linestyle=':', marker='s', linewidth=2,
                            capsize=5, label='Silent Recall')

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

    # Plot uncertainty metrics if requested
    if include_uncertainty_metrics and 'uncertainty_information_gain_mean' in results[report_counts[0]]:
        plt.figure(figsize=(10, 6))

        # Information gain
        means = [results[count]['uncertainty_information_gain_mean'] for count in report_counts]
        stds = [results[count]['uncertainty_information_gain_std'] for count in report_counts]

        plt.errorbar(report_counts, means, yerr=stds, marker='o', linestyle='-',
                     linewidth=2, capsize=5, label='Information Gain')

        # Uncertainty ratio
        if 'uncertainty_ratio_mean' in results[report_counts[0]]:
            means = [results[count]['uncertainty_ratio_mean'] for count in report_counts]
            stds = [results[count]['uncertainty_ratio_std'] for count in report_counts]

            plt.errorbar(report_counts, means, yerr=stds, marker='s', linestyle='--',
                         linewidth=2, capsize=5, label='Uncertainty Ratio')

        plt.xlabel('Number of Reports')
        plt.ylabel('Value')
        plt.title('Uncertainty Metrics with Increasing Reports')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()

        # Save plot
        plt.savefig('sinkhole_pu_uncertainty_metrics.png', dpi=300)
        log("Uncertainty metrics plot saved as 'sinkhole_pu_uncertainty_metrics.png'")


def compare_model_performance(traditional_results: Dict, graphsage_results: Dict,
                              k_vals: List[int] = DEFAULT_K_VALS) -> None:
    """
    Create a comparison plot between traditional and GraphSAGE models

    Args:
        traditional_results: Results from traditional model
        graphsage_results: Results from GraphSAGE model
        k_vals: List of K values to compare
    """
    import matplotlib.pyplot as plt

    report_counts = sorted(set(traditional_results.keys()) & set(graphsage_results.keys()))

    # Create figure with 3 subplots: Recall, Precision, PR-AUC
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    # Plot Recall@100
    for i, k in enumerate([k_vals[0]]):  # Just use the first k value
        trad_recall_means = [traditional_results[count][f'recall@{k}_mean'] for count in report_counts]
        trad_recall_stds = [traditional_results[count][f'recall@{k}_std'] for count in report_counts]

        gnn_recall_means = [graphsage_results[count][f'recall@{k}_mean'] for count in report_counts]
        gnn_recall_stds = [graphsage_results[count][f'recall@{k}_std'] for count in report_counts]

        axs[0].errorbar(report_counts, trad_recall_means, yerr=trad_recall_stds,
                        marker='o', linestyle='-', linewidth=2, capsize=5,
                        label=f'LightGBM Recall@{k}')

        axs[0].errorbar(report_counts, gnn_recall_means, yerr=gnn_recall_stds,
                        marker='s', linestyle='--', linewidth=2, capsize=5,
                        label=f'GraphSAGE Recall@{k}')

        axs[0].set_ylabel(f'Recall@{k}')
        axs[0].grid(True, linestyle='--', alpha=0.7)
        axs[0].set_title(f'Model Comparison - Recall@{k}')
        axs[0].legend()

    # Plot Precision@100
    for i, k in enumerate([k_vals[0]]):  # Just use the first k value
        trad_prec_means = [traditional_results[count][f'precision@{k}_mean'] for count in report_counts]
        trad_prec_stds = [traditional_results[count][f'precision@{k}_std'] for count in report_counts]

        gnn_prec_means = [graphsage_results[count][f'precision@{k}_mean'] for count in report_counts]
        gnn_prec_stds = [graphsage_results[count][f'precision@{k}_std'] for count in report_counts]

        axs[1].errorbar(report_counts, trad_prec_means, yerr=trad_prec_stds,
                        marker='o', linestyle='-', linewidth=2, capsize=5,
                        label=f'LightGBM Precision@{k}')

        axs[1].errorbar(report_counts, gnn_prec_means, yerr=gnn_prec_stds,
                        marker='s', linestyle='--', linewidth=2, capsize=5,
                        label=f'GraphSAGE Precision@{k}')

        axs[1].set_ylabel(f'Precision@{k}')
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].set_title(f'Model Comparison - Precision@{k}')
        axs[1].legend()

    # Plot PR-AUC
    trad_prauc_means = [traditional_results[count]['pr_auc_mean'] for count in report_counts]
    trad_prauc_stds = [traditional_results[count]['pr_auc_std'] for count in report_counts]

    gnn_prauc_means = [graphsage_results[count]['pr_auc_mean'] for count in report_counts]
    gnn_prauc_stds = [graphsage_results[count]['pr_auc_std'] for count in report_counts]

    axs[2].errorbar(report_counts, trad_prauc_means, yerr=trad_prauc_stds,
                    marker='o', linestyle='-', linewidth=2, capsize=5,
                    label='LightGBM PR-AUC')

    axs[2].errorbar(report_counts, gnn_prauc_means, yerr=gnn_prauc_stds,
                    marker='s', linestyle='--', linewidth=2, capsize=5,
                    label='GraphSAGE PR-AUC')

    axs[2].set_ylabel('PR-AUC')
    axs[2].grid(True, linestyle='--', alpha=0.7)
    axs[2].set_title('Model Comparison - PR-AUC')
    axs[2].legend()

    # Common x-axis label
    plt.xlabel('Number of Reports')
    plt.tight_layout()

    # Save comparison plot
    plt.savefig('sinkhole_model_comparison.png', dpi=300)
    log("Model comparison plot saved as 'sinkhole_model_comparison.png'")