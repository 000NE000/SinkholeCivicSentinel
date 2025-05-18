from __future__ import annotations

import os
import io
import time
from datetime import datetime
from typing import List, Tuple, Dict, Sequence

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.wkb
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc)
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# ────────────────────────────────────────────────────────────────────────────────
# 1. Utility
# ────────────────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
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
    log(f"Loaded {len(gdf):,} rows  •  positives: {pos} ({pos/len(gdf):.4%})")
    return gdf

# ────────────────────────────────────────────────────────────────────────────────
# 3. Metrics
# ────────────────────────────────────────────────────────────────────────────────

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

# ────────────────────────────────────────────────────────────────────────────────
# 4. Models
# ────────────────────────────────────────────────────────────────────────────────

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


def train_lightgbm_ranker(X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series,
                          spw: int) -> lgb.LGBMClassifier:
    """Train LightGBM with given scale_pos_weight."""
    params = LGB_PARAMS_BASE.copy()
    params["scale_pos_weight"] = spw
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    return model


def isolation_scores(X: pd.DataFrame, contamination: float = 0.02) -> np.ndarray:
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
    iso.fit(X)
    # decision_function returns *higher is less abnormal*; invert for risk
    return -iso.decision_function(X)

# ────────────────────────────────────────────────────────────────────────────────
# 5. Pipeline
# ────────────────────────────────────────────────────────────────────────────────

def main() -> None:
    gdf = load_dataset()
    y = gdf["subsidence_occurrence"].astype(int)
    X = gdf.drop(columns=[c for c in [
        "grid_id", "subsidence_occurrence", "subsidence_count", "geometry"] if c in gdf.columns])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    log(f"Train size {len(X_train):,} – positives {y_train.sum()} / Val positives {y_val.sum()}")

    # 1️⃣ LightGBM ranking scores
    spw_values = [300, 400]
    lgb_scores_dict: dict[int, np.ndarray] = {}
    for spw in spw_values:
        log(f"Training LightGBM with scale_pos_weight={spw}")
        model = train_lightgbm_ranker(X_train, y_train, X_val, y_val, spw)
        lgb_scores_dict[spw] = model.predict_proba(X_val)[:, 1]
        log(f"   ROC‑AUC {roc_auc_score(y_val, lgb_scores_dict[spw]):.4f}")

    # 2️⃣ Isolation Forest anomaly risk
    iso_score = isolation_scores(X)

    # 3️⃣ Evaluate Top‑K recall/precision on val set
    k_vals = [100, 200, 500]
    for spw, scores in lgb_scores_dict.items():
        m = topk_metrics(y_val, scores, k_vals)
        log(f"LightGBM(spw={spw}) Top‑K: {m}")

    iso_m = topk_metrics(y, iso_score, k_vals)
    log(f"IsolationForest Top‑K: {iso_m}")

    # 4️⃣ Simple ensemble (equal weight)
    best_spw = max(lgb_scores_dict, key=lambda k: roc_auc_score(y_val, lgb_scores_dict[k]))
    ens_scores = 0.5 * (lgb_scores_dict[best_spw]) + 0.5 * iso_score[y_val.index]
    ens_m = topk_metrics(y_val, ens_scores, k_vals)
    log(f"Ensemble Top‑K: {ens_m}")

    log("Done.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    log(f"Total runtime {(time.time() - t0):.1f}s")
