# escalation_causal/nuisance/nuisance_fitter.py

from __future__ import annotations

import logging
from typing import Tuple, Optional

import numpy as np
from sklearn.model_selection import KFold

from ..models.model_factory import build_sklearn_model
from ..models.calibrate import maybe_calibrate
from ..utils import clip_prob

logger = logging.getLogger(__name__)


class NuisanceModelFitter:
    """
    Cross‑fit propensity and outcome models using sample weights.

    Returns:
        - g1_hat: P(A=1 | X)  (propensity)
        - Q1_hat: E[Y | A=1, X]
        - Q0_hat: E[Y | A=0, X]
    """

    def __init__(
        self,
        propensity_model_type: str = "xgb",
        outcome_model_type: str = "xgb",
        calibrate_propensity: bool = True,
        calibrate_outcome: bool = False,   # regression, so usually False
        n_folds: int = 5,
        random_state: int = 42,
        min_prob: float = 0.01,
    ):
        self.propensity_model_type = propensity_model_type
        self.outcome_model_type = outcome_model_type
        self.calibrate_propensity = calibrate_propensity
        self.calibrate_outcome = calibrate_outcome
        self.n_folds = n_folds
        self.random_state = random_state
        self.min_prob = min_prob

    def cross_fit(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        weights: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform cross‑fitting.

        Args:
            X: Covariate matrix (n_samples, n_features)
            A: Treatment indicator (0/1), n_samples
            Y: Outcome (continuous), n_samples
            weights: Sample weights (e.g., testing IPW), n_samples

        Returns:
            g1_hat: out‑of‑fold propensity predictions (n_samples,)
            Q1_hat: out‑of‑fold outcome predictions for A=1 (n_samples,)
            Q0_hat: out‑of‑fold outcome predictions for A=0 (n_samples,)
        """
        n = len(X)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        g1_hat = np.zeros(n, dtype=float)
        Q1_hat = np.zeros(n, dtype=float)
        Q0_hat = np.zeros(n, dtype=float)

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            A_train, A_val = A[train_idx], A[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            w_train = weights[train_idx]

            # ----- Propensity model (P(A=1 | X)) -----
            prop_model = build_sklearn_model(
                self.propensity_model_type,
                random_state=self.random_state,
                task="classification",
            )
            if self.calibrate_propensity:
                prop_model = maybe_calibrate(prop_model, calibrate=True, method="isotonic", cv=3)
            prop_model.fit(X_train, A_train, sample_weight=w_train)

            # Predict on validation
            g1_val = prop_model.predict_proba(X_val)[:, 1]
            g1_hat[val_idx] = clip_prob(g1_val, self.min_prob, 1 - self.min_prob)

            # ----- Outcome model (E[Y | A, X]) -----
            # We fit a single model with A as an additional feature.
            X_aug_train = np.column_stack([X_train, A_train])
            X_aug_val_1 = np.column_stack([X_val, np.ones(len(X_val))])
            X_aug_val_0 = np.column_stack([X_val, np.zeros(len(X_val))])

            out_model = build_sklearn_model(
                self.outcome_model_type,
                random_state=self.random_state,
                task="regression",
            )
            # Note: calibration for regression is not typical; we skip it.
            out_model.fit(X_aug_train, Y_train, sample_weight=w_train)

            # Predict for A=1 and A=0
            Q1_val = out_model.predict(X_aug_val_1)
            Q0_val = out_model.predict(X_aug_val_0)

            Q1_hat[val_idx] = Q1_val
            Q0_hat[val_idx] = Q0_val

        return g1_hat, Q1_hat, Q0_hat