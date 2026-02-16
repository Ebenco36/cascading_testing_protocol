"""
Cross‑fitting utilities for machine learning models.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


class CrossFitter:
    """
    Perform cross‑fitting of a model to obtain out‑of‑fold predictions.

    Example:
        def build_model():
            return LogisticRegression()

        X, y, w = ...
        cf = CrossFitter(build_model, n_folds=5, random_state=42)
        oof_pred = cf.fit_predict(X, y, sample_weight=w, task="classification")
        models = cf.get_fitted_models()  # list of models fitted on each training fold
    """

    def __init__(
        self,
        model_builder: Callable[[], object],
        n_folds: int = 5,
        random_state: int = 42,
        shuffle: bool = True,
    ):
        """
        Args:
            model_builder: A zero‑argument callable that returns an unfitted model
                with .fit() and .predict() / .predict_proba() methods.
            n_folds: Number of folds.
            random_state: Random seed for reproducible splits.
            shuffle: Whether to shuffle the data before splitting.
        """
        self.model_builder = model_builder
        self.n_folds = n_folds
        self.random_state = random_state
        self.shuffle = shuffle

        self._fitted_models: List[object] = []
        self._oof_predictions: Optional[np.ndarray] = None

    def fit_predict(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        task: str = "classification",
    ) -> np.ndarray:
        """
        Perform cross‑fitting and return out‑of‑fold predictions.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target vector (n_samples,).
            sample_weight: Optional sample weights.
            task: 'classification' (uses predict_proba) or 'regression' (uses predict).

        Returns:
            Out‑of‑fold predictions of shape (n_samples,) for regression,
            or (n_samples, n_classes) for classification.
        """
        n = len(X)
        kf = KFold(
            n_splits=self.n_folds,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

        self._fitted_models = []
        oof_pred = None  # placeholder

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            # Prepare sample weights for training fold
            if sample_weight is not None:
                w_train = sample_weight[train_idx]
            else:
                w_train = None

            # Build and train model
            model = self.model_builder()
            try:
                if w_train is not None:
                    model.fit(X_train, y_train, sample_weight=w_train)
                else:
                    model.fit(X_train, y_train)
            except Exception as e:
                logger.error(f"Fold {fold} fitting failed: {e}")
                raise

            self._fitted_models.append(model)

            # Predict on validation fold
            if task == "classification":
                if hasattr(model, "predict_proba"):
                    pred = model.predict_proba(X_val)
                else:
                    # Fallback to decision_function and convert to probabilities? Not safe.
                    pred = model.predict(X_val)
                    # If binary classification, convert to 2‑column format
                    if pred.ndim == 1:
                        pred = np.column_stack((1 - pred, pred))
            else:  # regression
                pred = model.predict(X_val)

            # Initialize oof_pred array on first fold
            if oof_pred is None:
                if task == "classification" and pred.ndim == 2:
                    oof_pred = np.zeros((n, pred.shape[1]), dtype=float)
                else:
                    oof_pred = np.zeros(n, dtype=float)

            oof_pred[val_idx] = pred

        self._oof_predictions = oof_pred
        return oof_pred

    def get_fitted_models(self) -> List[object]:
        """Return the list of models fitted on each training fold."""
        if not self._fitted_models:
            raise RuntimeError("No fitted models available; run fit_predict first.")
        return self._fitted_models

    def get_oof_predictions(self) -> np.ndarray:
        """Return out‑of‑fold predictions from the last fit_predict call."""
        if self._oof_predictions is None:
            raise RuntimeError("No OOF predictions available; run fit_predict first.")
        return self._oof_predictions
    
    
# Alias for backward compatibility with older code
CrossFitWrapper = CrossFitter