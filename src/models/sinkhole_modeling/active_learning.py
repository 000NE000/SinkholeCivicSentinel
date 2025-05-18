"""
Active learning utilities for sinkhole screening
"""
import numpy as np
from typing import List
from src.models.sinkhole_modeling.config import log
from sklearn.base import clone

def select_active_learning_samples(
    scores: np.ndarray,
    silent_mask: np.ndarray,
    top_k: int = 100
) -> np.ndarray:
    """
    Return indices of silent zones with lowest confidence (closest to decision boundary)

    Args:
        scores: Prediction scores for all samples
        silent_mask: Boolean mask where True indicates silent zones
        top_k: Number of suggestions to return

    Returns:
        Array of indices for most uncertain samples
    """
    np.random.seed = 42
    # Check if we have any silent zones
    if not np.any(silent_mask):
        log("No silent zones found, returning random indices", level=2)
        # Return random indices if no silent zones
        all_indices = np.arange(len(scores))
        np.random.shuffle(all_indices)
        return all_indices[:min(top_k, len(all_indices))]

    # For positive silent zones, low confidence = low predicted score
    # Create array of scores only for silent zones, other values are NaN
    silent_scores = np.where(silent_mask, scores, np.nan)
    silent_scores = np.nan_to_num(silent_scores, nan=np.inf)

    # Get indices of silent zones sorted by score (ascending)
    # We use argsort on negated scores to get ascending order
    idx = np.argsort(silent_scores)

    # Keep only valid indices (not NaN)
    valid_idx = [i for i in idx if not np.isnan(silent_scores[i])]

    # Return top_k or all if fewer
    top_k = min(top_k, len(valid_idx))
    result = valid_idx[:top_k]

    log(f"Selected {len(result)} samples for active learning from {silent_mask.sum()} silent zones", level=2)
    return np.array(result)