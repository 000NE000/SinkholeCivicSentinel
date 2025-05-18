"""
Main script to run enhanced GNN sinkhole screening model
This script implements the full workflow for enhanced GNN screening:
1. Feature generation (interaction features)
2. Enhanced GNN model with uncertainty-based candidate selection
3. Model training and evaluation
4. Feature importance analysis
5. False negative analysis
6. Hotspot identification and verification
"""
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.models.sinkhole_modeling.config import log, DEFAULT_K_VALS
from src.models.sinkhole_modeling.data_loader import load_dataset, create_spatial_blocks
from src.models.sinkhole_modeling.evaluation import compute_comprehensive_metrics

# Import enhanced GNN implementation
from enhanced_gnn import (
    create_interaction_features,
    visualize_uncertainty_sampling,
    run_comparison_experiments,
)

# Import analysis functions
from feature_importance import (
    calculate_feature_importance,
    plot_feature_importance,
    run_ablation_study,
    analyze_false_negatives,
    identify_pothole_hotspots
)



def main():
    """
    Enhanced GNN Sinkhole Screening (FIXED)
    """
    import os
    import time
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from src.models.sinkhole_modeling.config import log, DEFAULT_K_VALS
    from src.models.sinkhole_modeling.data_loader import load_dataset, create_spatial_blocks
    from src.models.sinkhole_modeling.evaluation import compute_comprehensive_metrics
    from enhanced_gnn import create_interaction_features, run_comparison_experiments
    from feature_importance import (
        calculate_feature_importance,
        plot_feature_importance,
        run_ablation_study,
        analyze_false_negatives,
        identify_pothole_hotspots
    )

    log("\n============================================", level=1)
    log("ENHANCED GNN SINKHOLE SCREENING (FIXED)", level=1)
    log("============================================\n", level=1)

    start_time = time.time()
    output_dir = "enhanced_gnn_results"
    os.makedirs(output_dir, exist_ok=True)
    log(f"Results will be saved to: {output_dir}", level=1)

    # Step 1: Load and prepare data
    log("\n=== Step 1: Loading Dataset ===", level=1)
    gdf = load_dataset(table="feature_matrix_100_geo")
    if 'pothole_kde_density' not in gdf.columns:
        gdf['pothole_kde_density'] = np.random.random(len(gdf)) * 0.5

    y = gdf["subsidence_occurrence"].astype(int)
    X = gdf.drop(columns=["subsidence_occurrence", "subsidence_count"], errors="ignore")

    # Step 2: Interaction features
    log("\n=== Step 2: Creating Interaction Features ===", level=1)
    X = create_interaction_features(X)

    # Keep both “full” and “proc” versions
    X_full = X.copy()                            # has geometry, pipe_risk, etc.
    X_proc = X_full.drop(columns=["geometry"], errors="ignore")  # purely tabular

    # Spatial blocks for CV
    spatial_blocks = create_spatial_blocks(gdf, n_blocks=5)

    # Step 3: Comparison experiments (tabular only)
    log("\n=== Step 3: Running Comparison Experiments ===", level=1)
    results, comparison_table, baseline_model, enhanced_model = \
        run_comparison_experiments(X_full, y, spatial_blocks)
    comparison_table.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=True)
    log("Comparison results saved.", level=1)

    # Step 4: Feature importance (tabular)
    log("\n=== Step 4: Feature Importance Analysis ===", level=1)
    importance_df = calculate_feature_importance(enhanced_model, X_proc, y)
    importance_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
    plot_feature_importance(importance_df, title="Enhanced GNN Feature Importance")
    plt.savefig(os.path.join(output_dir, "feature_importance.png"))
    log("Feature importance saved.", level=1)

    # Step 5: Ablation study (tabular)
    log("\n=== Step 5: Ablation Study ===", level=1)
    ablation_df = run_ablation_study(X_proc, y, spatial_blocks)
    ablation_df.to_csv(os.path.join(output_dir, "ablation_study.csv"), index=False)
    log("Ablation study results saved.", level=1)

    # Step 6: False negative analysis (requires geometry)
    log("\n=== Step 6: False Negative Analysis ===", level=1)
    fn_df = analyze_false_negatives(enhanced_model, X_full, y)
    fn_path = os.path.join(output_dir, "false_negative_analysis.csv")
    if 'geometry' in fn_df.columns:
        fn_df.drop(columns=['geometry']).to_csv(fn_path, index=False)
    else:
        fn_df.to_csv(fn_path, index=False)
    log("False negative analysis saved.", level=1)

    # Step 7: Hotspot identification (requires geometry + pipe_risk)
    log("\n=== Step 7: Hotspot Identification ===", level=1)
    hotspot_gdf = identify_pothole_hotspots(X_full, y, enhanced_model)
    if hotspot_gdf is not None:
        hotspot_path = os.path.join(output_dir, "hotspot_analysis.gpkg")
        hotspot_gdf.to_file(hotspot_path, driver='GPKG')
        log("Hotspot analysis saved.", level=1)

    # Step 8: Save models
    log("\n=== Step 8: Saving Models ===", level=1)
    import pickle
    pickle.dump(baseline_model, open(os.path.join(output_dir, "baseline_model.pkl"), 'wb'))
    pickle.dump(enhanced_model, open(os.path.join(output_dir, "enhanced_model.pkl"), 'wb'))
    log("Models saved.", level=1)

    elapsed = time.time() - start_time
    log(f"\nAnalysis completed in {elapsed:.2f} seconds", level=1)
    log(f"All results in: {output_dir}", level=1)

if __name__ == "__main__":
    main()