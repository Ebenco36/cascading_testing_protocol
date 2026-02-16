"""
Testing Model for IPW Weights
=============================

Estimates P(T_A = 1 | X) for a given trigger antibiotic A,
and produces inverse probability weights for the tested isolates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.model_selection import KFold

from ..models.model_factory import build_sklearn_model
from ..models.calibrate import maybe_calibrate
from ..utils.weights import clip_prob, effective_sample_size, stable_quantiles

logger = logging.getLogger(__name__)


@dataclass
class TestingModelDiagnostics:
    """Diagnostics for the fitted testing model."""
    p_test_quantiles: Dict[str, float]
    w_quantiles: Dict[str, float]
    w_cap: float
    effective_sample_size: float
    n_tested: int
    n_total: int


class TestingModel:
    """
    Fit P(T_A = 1 | X) and compute IPW weights for tested observations.

    The model is fitted on the entire test cohort (tested + non‑tested).
    Weights for tested isolates are 1 / P(T_A = 1 | X), optionally capped.
    """

    def __init__(
        self,
        model_type: str = "xgb",
        calibrate: bool = True,
        calibration_method: str = "isotonic",
        cv_folds: int = 5,
        random_state: int = 42,
        min_prob: float = 0.01,
        weight_cap_percentile: float = 99.0,
        n_folds_cv: int = 5,           # for cross‑fitting the testing model itself (optional)
    ):
        """
        Args:
            model_type: Type of classifier ('xgb', 'rf', 'logit').
            calibrate: Whether to calibrate the model.
            calibration_method: 'isotonic' or 'sigmoid'.
            cv_folds: Number of folds for calibration if calibrate=True.
            random_state: Random seed.
            min_prob: Minimum probability for clipping.
            weight_cap_percentile: Percentile at which to cap weights among tested.
            n_folds_cv: If >1, use cross‑fitting to obtain out‑of‑fold predictions.
        """
        self.model_type = model_type
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.min_prob = min_prob
        self.weight_cap_percentile = weight_cap_percentile
        self.n_folds_cv = n_folds_cv

        self._fitted = False
        self._model = None
        self._is_cross_fitted = n_folds_cv > 1

    def fit(self, X: np.ndarray, T: np.ndarray) -> TestingModel:
        """
        Fit the testing model.

        If n_folds_cv > 1, cross‑fitting is used and out‑of‑fold predictions
        are stored; otherwise, the model is fitted on all data and can be used
        for prediction on new data.
        """
        if self._is_cross_fitted:
            # Cross‑fitting: we won't keep a single model, but rather store
            # out‑of‑fold predictions for the training data.
            self._cross_fit(X, T)
        else:
            # Simple fit on all data
            base = build_sklearn_model(self.model_type, random_state=self.random_state, task="classification")
            model = maybe_calibrate(
                base,
                calibrate=self.calibrate,
                method=self.calibration_method,
                cv=self.cv_folds,
            )
            model.fit(X, T)
            self._model = model

        self._fitted = True
        return self

    def _cross_fit(self, X: np.ndarray, T: np.ndarray) -> None:
        """Perform cross‑fitting to obtain out‑of‑fold predictions."""
        kf = KFold(n_splits=self.n_folds_cv, shuffle=True, random_state=self.random_state)
        self._oof_predictions = np.zeros(len(X), dtype=float)

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            T_train = T[train_idx]

            base = build_sklearn_model(self.model_type, random_state=self.random_state, task="classification")
            model = maybe_calibrate(
                base,
                calibrate=self.calibrate,
                method=self.calibration_method,
                cv=self.cv_folds,
            )
            model.fit(X_train, T_train)

            proba = model.predict_proba(X_val)[:, 1]
            self._oof_predictions[val_idx] = proba

        # No single model stored; we will use OOF predictions for weight computation
        # on the training data. For new data, we would need to refit on all data
        # or use cross‑fitting ensembles; but typically we only need predictions
        # on the same data for weights, so we can just use OOF.
        self._model = None  # no single model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict P(T_A = 1 | X) for new data.

        If the model was cross‑fitted (no single model), this method raises an error.
        In a pipeline, you would normally call `fit` with `n_folds_cv=1` or use
        the out‑of‑fold predictions stored in `get_oof_predictions()`.
        """
        if self._is_cross_fitted:
            raise RuntimeError(
                "Model was cross‑fitted; no single model available. "
                "Use get_oof_predictions() to obtain predictions on the training data."
            )
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        proba = self._model.predict_proba(X)[:, 1]
        return clip_prob(proba, self.min_prob, 1 - self.min_prob)

    def get_oof_predictions(self) -> np.ndarray:
        """Return out‑of‑fold predictions from cross‑fitting (only available if cross‑fitted)."""
        if not self._is_cross_fitted or not hasattr(self, "_oof_predictions"):
            raise RuntimeError("No out‑of‑fold predictions available.")
        return self._oof_predictions

    def compute_weights(
        self,
        p_test: np.ndarray,
        tested_mask: np.ndarray,
    ) -> Tuple[np.ndarray, TestingModelDiagnostics]:
        """
        Compute IPW weights = 1 / P(T_A = 1 | X) for tested observations.

        Args:
            p_test: Array of testing probabilities (same length as data).
            tested_mask: Boolean mask indicating tested isolates.

        Returns:
            weights: Full‑length array, zero for non‑tested, IPW for tested (capped).
            diagnostics: TestingModelDiagnostics object.
        """
        # Clip probabilities to avoid extreme weights (e.g., division by zero)
        p_test = clip_prob(p_test, self.min_prob, 1 - self.min_prob)

        w = np.zeros_like(p_test, dtype=float)
        if tested_mask.any():
            raw_weights = 1.0 / p_test[tested_mask]
            # Cap at percentile
            cap = float(np.percentile(raw_weights, self.weight_cap_percentile))
            w[tested_mask] = np.clip(raw_weights, None, cap)
        else:
            cap = np.nan

        # Diagnostics
        p_tested = p_test[tested_mask] if tested_mask.any() else np.array([])
        w_tested = w[tested_mask] if tested_mask.any() else np.array([])

        p_quantiles = stable_quantiles(p_tested, qs=(0, 0.01, 0.5, 0.99, 1.0))
        w_quantiles = stable_quantiles(w_tested, qs=(0, 0.01, 0.5, 0.99, 1.0))

        ess = effective_sample_size(w_tested) if len(w_tested) > 0 else np.nan

        diag = TestingModelDiagnostics(
            p_test_quantiles=p_quantiles,
            w_quantiles=w_quantiles,
            w_cap=float(cap) if not np.isnan(cap) else np.nan,
            effective_sample_size=ess,
            n_tested=int(tested_mask.sum()),
            n_total=len(p_test),
        )
        return w, diag