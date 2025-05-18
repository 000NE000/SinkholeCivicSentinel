"""
nnPU-style utilities for pseudo-labeling with silent zones
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Optional, Union, Tuple, List
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import BaggingClassifier
from src.models.sinkhole_modeling.config import log, LGB_PARAMS_BASE

def nnpu_loss(y_true, y_pred_proba, prior, unlabeled_weight=1.0):
    """
    Non-negative PU Learning loss function
    Based on "Positive-Unlabeled Learning with Non-Negative Risk Estimator" (Kiryo et al., 2017)

    Args:
        y_true: Binary labels (1 for positive, 0 for unlabeled)
        y_pred_proba: Predicted probabilities
        prior: Prior probability of positive class
        unlabeled_weight: Weight for unlabeled samples

    Returns:
        Non-negative PU loss value
    """
    # Positive and unlabeled masks
    pos_mask = (y_true == 1)
    unl_mask = (y_true == 0)

    # Compute positive risk (expected loss on positive data)
    pos_risk = np.mean(1 - y_pred_proba[pos_mask]) if np.any(pos_mask) else 0

    # Compute negative risk (expected loss on negative data)
    neg_risk = np.mean(y_pred_proba[unl_mask]) if np.any(unl_mask) else 0

    # Compute U-risk (expected loss on unlabeled data)
    u_risk = np.mean(y_pred_proba[unl_mask]) if np.any(unl_mask) else 0

    # Compute nnPU risk
    nnpu_risk = pos_risk + max(0, unlabeled_weight * u_risk - prior * pos_risk)

    return nnpu_risk

class NNPUClassifier(BaseEstimator):
    """
    Non-negative PU Learning classifier wrapper
    Based on "Positive-Unlabeled Learning with Non-Negative Risk Estimator"

    This implementation wraps a base classifier and trains it using nnPU principles.
    """

    def __init__(self, base_estimator=None, prior=0.5, unlabeled_weight=1.0, n_iter=100):
        """
        Initialize NNPUClassifier

        Args:
            base_estimator: Base classifier to wrap (must support sample_weight)
            prior: Prior probability of positive class
            unlabeled_weight: Weight for unlabeled samples
            n_iter: Number of iterations for training
        """
        self.base_estimator = base_estimator or lgb.LGBMClassifier(**LGB_PARAMS_BASE)
        self.prior = prior
        self.unlabeled_weight = unlabeled_weight
        self.n_iter = n_iter
        self.estimator_ = None

    def fit(self, X, y):
        """
        Fit the NNPUClassifier to the data

        Args:
            X: Features
            y: Labels (1 for positive, 0 for unlabeled)

        Returns:
            Self
        """
        # Clone the base estimator
        self.estimator_ = clone(self.base_estimator)

        # Compute initial weights
        pos_mask = (y == 1)
        unl_mask = (y == 0)
        pos_count = np.sum(pos_mask)
        unl_count = np.sum(unl_mask)

        # Adjust prior if needed
        if self.prior is None:
            self.prior = pos_count / (pos_count + unl_count)

        # Create sample weights
        sample_weight = np.ones(len(y))
        sample_weight[pos_mask] = 1.0
        sample_weight[unl_mask] = self.unlabeled_weight

        # Train the base estimator with the adjusted weights
        self.estimator_.fit(X, y, sample_weight=sample_weight)

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities

        Args:
            X: Features

        Returns:
            Predicted probabilities
        """
        if self.estimator_ is None:
            raise RuntimeError("Classifier not fitted yet.")

        return self.estimator_.predict_proba(X)

    def predict(self, X):
        """
        Predict class labels

        Args:
            X: Features

        Returns:
            Predicted labels
        """
        if self.estimator_ is None:
            raise RuntimeError("Classifier not fitted yet.")

        return self.estimator_.predict(X)

class PUBaggingClassifier(BaseEstimator):
    """
    PU Bagging classifier
    Based on "Building Ensembles of Weak Learners in PU Classification Problem" (Mordelet & Vert, 2014)

    This implementation creates multiple classifiers, each trained on positive and a subset of unlabeled data.
    """

    def __init__(self, base_estimator=None, n_estimators=10, max_samples=0.5, random_state=None):
        """
        Initialize PUBaggingClassifier

        Args:
            base_estimator: Base classifier to use in bagging
            n_estimators: Number of estimators in the ensemble
            max_samples: Fraction of unlabeled samples to use for each estimator
            random_state: Random state for reproducibility
        """
        self.base_estimator = base_estimator or lgb.LGBMClassifier(**LGB_PARAMS_BASE)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.estimators_ = []

    def fit(self, X, y):
        """
        Fit the PUBaggingClassifier to the data

        Args:
            X: Features
            y: Labels (1 for positive, 0 for unlabeled)

        Returns:
            Self
        """
        # Initialize random state
        rng = np.random.RandomState(self.random_state)

        # Find positive and unlabeled samples
        pos_idx = np.where(y == 1)[0]
        unl_idx = np.where(y == 0)[0]

        # Number of unlabeled samples to select for each estimator
        n_unl_samples = int(self.max_samples * len(unl_idx))

        # Train each estimator
        self.estimators_ = []
        for i in range(self.n_estimators):
            # Create a clone of the base estimator
            estimator = clone(self.base_estimator)

            # Sample unlabeled data
            sampled_unl_idx = rng.choice(unl_idx, size=n_unl_samples, replace=False)

            # Combine with positive data
            sample_idx = np.concatenate([pos_idx, sampled_unl_idx])

            # Train the estimator
            estimator.fit(X.iloc[sample_idx] if hasattr(X, 'iloc') else X[sample_idx],
                          y[sample_idx])

            # Add to ensemble
            self.estimators_.append(estimator)

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities by averaging predictions from all estimators

        Args:
            X: Features

        Returns:
            Predicted probabilities
        """
        if not self.estimators_:
            raise RuntimeError("Classifier not fitted yet.")

        # Get probabilities from each estimator
        all_probs = np.array([estimator.predict_proba(X) for estimator in self.estimators_])

        # Average probabilities
        avg_probs = np.mean(all_probs, axis=0)

        return avg_probs

    def predict(self, X):
        """
        Predict class labels

        Args:
            X: Features

        Returns:
            Predicted labels
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

def enhanced_pu_learning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_unlabeled: pd.DataFrame,
    silent_mask: np.ndarray,
    survey_mask: Optional[np.ndarray] = None,
    method: str = "nnpu",
    base_model=None,
    n_estimators: int = 10,
    unlabeled_weight: float = 0.5,
    survey_weight: float = 1.0
) -> BaseEstimator:
    """
    Enhanced PU learning with multiple approaches

    Args:
        X_train: Features for labeled training data
        y_train: True labels for training data
        X_unlabeled: Features for unlabeled data (with silent/survey zones)
        silent_mask: Boolean mask where True indicates silent zones
        survey_mask: Boolean mask where True indicates survey zones
        method: PU learning method to use ('nnpu', 'bagging', or 'simple')
        base_model: Optional base model to use
        n_estimators: Number of estimators for PU bagging
        unlabeled_weight: Weight for silent zones
        survey_weight: Weight for survey zones

    Returns:
        Trained PU classifier
    """
    assert len(silent_mask) == len(
        X_unlabeled), f"Silent mask size {len(silent_mask)} doesn't match X_unlabeled size {len(X_unlabeled)}"

    if survey_mask is not None:
        assert len(survey_mask) == len(
            X_unlabeled), f"Survey mask size {len(survey_mask)} doesn't match X_unlabeled size {len(X_unlabeled)}"
    else:
        # Create empty survey mask if not provided
        survey_mask = np.zeros_like(silent_mask, dtype=bool)

    # Create pseudo-labels - both silent and survey zones become positives
    y_unlabeled = np.zeros(len(X_unlabeled), dtype=int)
    y_unlabeled[silent_mask] = 1  # silent zones as positives
    y_unlabeled[survey_mask] = 1  # survey zones as positives

    # Combine datasets
    X_comb = pd.concat([X_train, X_unlabeled], ignore_index=True)
    y_comb = pd.concat([y_train, pd.Series(y_unlabeled)], ignore_index=True)

    # If using simple pseudo-labeling with sample weights
    if method == "simple":
        # Sample weights: use different weights for silent vs survey zones
        w_train = np.ones(len(y_train))
        w_unl = np.ones(len(y_unlabeled))

        # Apply different weights
        w_unl[silent_mask] = unlabeled_weight  # regular silent zones
        w_unl[survey_mask] = survey_weight     # survey zones (lower confidence)

        sample_weight = np.concatenate([w_train, w_unl])

        log(f"Training with simple pseudo-labeling: {len(y_train)} labeled samples + "
            f"{silent_mask.sum()} silent zones (w={unlabeled_weight}) + "
            f"{survey_mask.sum()} survey zones (w={survey_weight})", level=2)

        # Train model
        model = base_model or lgb.LGBMClassifier(**LGB_PARAMS_BASE)
        model.fit(X_comb, y_comb, sample_weight=sample_weight)

    # If using Non-negative PU Learning
    elif method == "nnpu":
        # Estimate positive class prior
        pos_prior = (y_train.sum() + silent_mask.sum() * unlabeled_weight +
                     survey_mask.sum() * survey_weight) / len(X_comb)

        log(f"Training with nnPU: {len(y_train)} labeled samples + "
            f"{silent_mask.sum()} silent zones + {survey_mask.sum()} survey zones, "
            f"prior={pos_prior:.4f}", level=2)

        # Create and fit nnPU classifier
        model = NNPUClassifier(
            base_estimator=base_model or lgb.LGBMClassifier(**LGB_PARAMS_BASE),
            prior=pos_prior,
            unlabeled_weight=unlabeled_weight
        )
        model.fit(X_comb, y_comb)

    # If using PU Bagging
    elif method == "bagging":
        log(f"Training with PU-Bagging: {len(y_train)} labeled samples + "
            f"{silent_mask.sum()} silent zones + {survey_mask.sum()} survey zones, "
            f"n_estimators={n_estimators}", level=2)

        # Create and fit PU Bagging classifier
        model = PUBaggingClassifier(
            base_estimator=base_model or lgb.LGBMClassifier(**LGB_PARAMS_BASE),
            n_estimators=n_estimators,
            max_samples=0.7,  # Use 70% of unlabeled samples for each estimator
            random_state=42
        )
        model.fit(X_comb, y_comb)

    else:
        raise ValueError(f"Unknown PU learning method: {method}")

    return model

def pseudo_labeling_nnpu(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_unlabeled: pd.DataFrame,
    silent_mask: np.ndarray,
    survey_mask: Optional[np.ndarray] = None,
    base_model=None,
    unlabeled_weight: float = 0.5,
    survey_weight: float = 0.2
) -> lgb.LGBMClassifier:
    """
    [Legacy] Enhanced nnPU-style pseudo-label:
    - treat silent_mask==True as weak positives
    - treat survey_mask==True as weak positives with lower confidence

    Note: For backward compatibility. Use enhanced_pu_learning for better performance.

    Args:
        X_train: Features for labeled training data
        y_train: True labels for training data
        X_unlabeled: Features for unlabeled data (with silent/survey zones)
        silent_mask: Boolean mask where True indicates silent zones
        survey_mask: Boolean mask where True indicates survey zones
        base_model: Optional base model (LGBMClassifier) to use
        unlabeled_weight: Weight to assign to silent zone pseudo-labeled examples
        survey_weight: Weight to assign to survey zone pseudo-labeled examples (lower confidence)

    Returns:
        Trained model with pseudo-labeling
    """
    return enhanced_pu_learning(
        X_train, y_train, X_unlabeled, silent_mask, survey_mask,
        method="simple", base_model=base_model,
        unlabeled_weight=unlabeled_weight, survey_weight=survey_weight
    )

def optimize_for_survey_zones(
    s1_scores: np.ndarray,
    s2_scores: np.ndarray,
    survey_mask: np.ndarray,
    thr_list: list[float] = None,
    alpha_list: list[float] = None,
    k: int = 100
) -> Tuple[float, float, float, dict]:
    """
    Optimize threshold percentile and alpha weight to maximize survey zone recall@k

    Args:
        s1_scores: Stage 1 scores
        s2_scores: Stage 2 scores
        survey_mask: Boolean mask for survey zones
        thr_list: List of threshold percentiles to try
        alpha_list: List of alpha values to try
        k: K value for evaluation

    Returns:
        Tuple of (best_percentile, best_alpha, best_recall, metrics_dict)
    """
    if thr_list is None:
        thr_list = [10, 15, 20, 25, 30, 35, 40]

    if alpha_list is None:
        alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    # Check if we have any survey zones
    total_survey = survey_mask.sum()
    if total_survey == 0:
        log("No survey zones found for optimization", level=2)
        return 20, 0.5, 0, {"survey_recall": 0.0}

    best = {"survey_recall": -1, "percentile": None, "alpha": None}

    log(f"Grid search optimization for survey zones: {len(thr_list)} thresholds × {len(alpha_list)} alphas", level=2)

    # Track all combinations for analysis
    all_results = []

    for pct in thr_list:
        thr = np.percentile(s1_scores, pct)
        s1 = s1_scores.copy()
        mask_low = s1 < thr

        # Stage2 proxy score: (1-s1) for low confidence regions
        s2_proxy = np.zeros_like(s1)
        s2_proxy[mask_low] = 1 - s1[mask_low]

        for alpha in alpha_list:
            final_score = alpha * s1 + (1 - alpha) * s2_proxy

            # Get top-k indices
            top_k_idx = np.argsort(final_score)[::-1][:k]

            # Calculate survey_recall@k
            survey_recall = survey_mask[top_k_idx].sum() / total_survey

            # Store result
            result = {
                "percentile": pct,
                "alpha": alpha,
                "survey_recall": survey_recall
            }
            all_results.append(result)

            # Update best if better
            if survey_recall > best["survey_recall"]:
                best.update({"survey_recall": survey_recall, "percentile": pct, "alpha": alpha})

    log(f"Best survey_recall@{k}: {best['survey_recall']:.4f} at percentile={best['percentile']}%, alpha={best['alpha']}", level=1)

    return best["percentile"], best["alpha"], best["survey_recall"], best