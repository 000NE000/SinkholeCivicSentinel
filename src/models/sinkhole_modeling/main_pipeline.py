"""
Main pipeline for sinkhole risk screening
"""
import time
import pandas as pd
import numpy as np
from src.models.sinkhole_modeling.config import log, DEFAULT_MODEL_PARAMS, DEFAULT_K_VALS
from src.models.sinkhole_modeling.data_loader import load_dataset, get_silent_grid_ids, create_spatial_blocks
from src.models.sinkhole_modeling.evaluation import evaluate_reports_impact_pu, plot_pu_report_impact
from src.models.sinkhole_modeling.optimizer import optimize_hyperparameters
from src.models.sinkhole_modeling.stage1_model import TwoStageSinkholeScreener

def main() -> None:
    # List of report tables to evaluate
    report_tables = [
        "feature_matrix_25_geo",
        "feature_matrix_50_geo",
        "feature_matrix_75_geo",
        "feature_matrix_100_geo"
    ]

    #================== 1. Load the 100-report dataset for hyperparameter optimization
    log("\n=== Step 1: Hyperparameter Optimization ===", level=1)
    gdf_100 = load_dataset(table="feature_matrix_100_geo")

    # Prepare data - grid_id 컬럼 유지 (silent zone 통합에 필요)
    y = gdf_100["subsidence_occurrence"].astype(int)
    X_feat = gdf_100.drop(columns=[c for c in [
        "subsidence_occurrence", "subsidence_count", "geometry"]
                              if c in gdf_100.columns])
    X_geo = gdf_100.copy()

    # Create spatial blocks
    spatial_blocks = create_spatial_blocks(gdf_100, n_blocks=5)
    # Dynamically generate silent grid IDs
    silent_ids = get_silent_grid_ids(X_feat, y, percentile=90)
    log(f"Generated {len(silent_ids)} silent grid IDs", level=1)
    log(f">>> Columns passed to optimize: {list(X_geo.columns)}")
    # Optimize hyperparameters
    opt_results = optimize_hyperparameters(X_geo, y, spatial_blocks)

    # Step 2: Use best params
    log("\n=== Step 2: Training Enhanced Model with Silent Zone Integration ===", level=1)
    best_params = opt_results['best_params']
    if best_params is None:
        best_params = DEFAULT_MODEL_PARAMS
        log(f"Using default model parameters: {best_params}", level=1)
    else:
        best_params = {
            'proximity_feat': DEFAULT_MODEL_PARAMS['proximity_feat'],
            'stage1_model_type': DEFAULT_MODEL_PARAMS['stage1_model_type'],
            'threshold_percentile': DEFAULT_MODEL_PARAMS['threshold_percentile'],
            'feature_fraction': DEFAULT_MODEL_PARAMS['feature_fraction'],
            'alpha': DEFAULT_MODEL_PARAMS.get('ensemble_weight', 0.5),
            'model_type': DEFAULT_MODEL_PARAMS.get('stage2_model_type', 'lightgbm'),
            'margin_low': DEFAULT_MODEL_PARAMS.get('margin_low', 0.4),
            'margin_high': DEFAULT_MODEL_PARAMS.get('margin_high', 0.6)
        }
        log(f"Using optimized parameters: {best_params}", level=1)


            # 향상된 모델 생성 및 학습
    enhanced_model = TwoStageSinkholeScreener(
        proximity_feat=best_params['proximity_feat'],
        stage1_model_type=best_params['stage1_model_type'],
        threshold_percentile=best_params['threshold_percentile'],
        feature_fraction=best_params['feature_fraction'],
        ensemble_weight=best_params['alpha']
    )

    # Train enhanced model and get active learning suggestions
    try:
        enhanced_model.fit(
            X_geo, y, spatial_blocks=spatial_blocks,
            model_type=best_params['model_type'],
            margin_low=best_params['margin_low'],
            margin_high=best_params['margin_high']
        )
        # 액티브 러닝 추천 얻기
        top_suggestions = enhanced_model.get_active_learning_suggestions(X_geo, silent_ids, top_k=50)
        log(f"Top 50 grid IDs for investigation: {top_suggestions[:10]}... (and 40 more)", level=1)

        # 추천 결과 저장
        pd.DataFrame({'grid_id': top_suggestions}).to_csv('active_learning_suggestions.csv', index=False)
        log("Active learning suggestions saved to 'active_learning_suggestions.csv'", level=1)
    except Exception as e:
        log(f"Error in enhanced model training or active learning: {str(e)}", level=1)
        log("Continuing with standard model and evaluation...", level=1)
        # Fall back to basic model if enhanced fails
        enhanced_model.fit(X_geo, y, spatial_blocks)

    # ========================3. Evaluate with PU-learning spatial CV across different report counts
    log("\n=== Step 3: Evaluating Report Impact with PU-Learning Simulation ===", level=1)

    # 모델 타입 결정 (graphsage가 있으면 um_gnn, 아니면 lightgbm)
    use_graphsage = best_params.get('model_type', 'lightgbm') in ['graphsage', 'um_gnn', 'gnn']

    # Evaluate report impact using PU-learning simulation
    pu_results = evaluate_reports_impact_pu(
        report_tables,
        k_vals=DEFAULT_K_VALS,
        n_iters=10,
        include_silent_metrics=False,  # Disable silent metrics
        use_graphsage=use_graphsage,  # model_type 대신 use_graphsage 사용
        use_uncertainty_masking=False,
        random_state=42
    )

    # 4. Create results table and plots
    log("\n=== Step 4: Creating PU-Learning Report Impact Analysis ===", level=1)

    # Create results table with expanded metrics
    report_counts = sorted(pu_results.keys())

    # Table columns with all metrics
    metrics = ([f"recall@{k}" for k in DEFAULT_K_VALS] +
               [f"precision@{k}" for k in DEFAULT_K_VALS] +
               [f"lift@{k}" for k in DEFAULT_K_VALS] +
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

    # Plot improvement with increasing reports (without silent metrics)
    plot_pu_report_impact(pu_results, k_vals=DEFAULT_K_VALS, include_silent_metrics=False)

    log("Analysis complete.", level=1)

if __name__ == "__main__":
    t0 = time.time()
    main()
    log(f"Total runtime {(time.time() - t0):.1f}s")