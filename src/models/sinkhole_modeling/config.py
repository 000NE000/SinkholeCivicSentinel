"""
Global configuration parameters for sinkhole modeling
"""
from datetime import datetime

# Logging
VERBOSE = False

def log(msg: str, level: int = 1) -> None:
    """
    level=1 : fold summary, final summary
    level=2 : detailed (load_dataset, each fold entry, etc)
    """
    if level == 1 or (level == 2 and VERBOSE):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# Database Configuration
DB_ENV_VAR = "DB_DSN"

# Cross-validation parameters
DEFAULT_K_VALS = [100, 200, 500]
DEFAULT_N_FOLDS = 5

# LightGBM base parameters
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

# GraphSAGE parameters
GRAPHSAGE_PARAMS = dict(
    hidden_channels=64,
    num_layers=2,
    dropout=0.3,
    jk='cat',  # Jumping knowledge with concatenation
    lr=0.001,
    epochs=100,
    patience=20,
    batch_size=512,
    verbose=False,
    radius=1000
)

# Uncertainty masking parameters
UNCERTAINTY_THRESHOLD_LOW = 0.4
UNCERTAINTY_THRESHOLD_HIGH = 0.6

# TwoStageModel default parameters
DEFAULT_MODEL_PARAMS = {
    'proximity_feat': 'min_distance_to_sinkhole',
    'stage1_model_type': 'lgbm',
    'threshold_percentile': 80,
    'feature_fraction': 0.6,
    'use_focal_loss': True,
    'stage2_model_type': 'graphsage'  # Adding GraphSAGE as default
}

# Features that are direct observations of sinkholes
OBSERVED_SINKHOLE_FEATURES = [
    "min_distance_to_sinkhole",
    "weighted_sinkhole_density"
]

# Parameter grids for optimization
THRESHOLD_PERCENTILE_GRID = [70, 75, 80, 85, 90]
FEATURE_FRACTION_GRID = [0.5, 0.6, 0.7]
ALPHA_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Optuna parameters
OPTUNA_TRIALS = 50
OPTUNA_TIMEOUT = 3600  # 1 hour