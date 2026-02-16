"""
Routine Policy Learner and Continuous Escalation Score
=======================================================

Defines the routine testing policy S_D(C) for each target drug D.
Learns P(T_D=1 | C) from training data, validates calibration,
and computes the continuous escalation score Y* = T_D / P(T_D=1|C) on test data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from ..models.model_factory import build_sklearn_model
from ..models.calibrate import maybe_calibrate
from ..utils import safe_series_to_numpy

logger = logging.getLogger(__name__)


@dataclass
class PolicyModel:
    """Container for a fitted policy model for one target drug."""
    code: str
    model: Any                     # sklearn-like classifier with predict_proba
    context_cols: List[str]
    method: str                    # 'empirical' or 'ml'
    calibration_curve: Optional[Dict[str, np.ndarray]] = None
    brier_score: Optional[float] = None
    min_prob: float = 0.01          # for clipping


@dataclass
class PolicyResult:
    """Result of fitting routine policies for all targets."""
    models: Dict[str, PolicyModel]
    context_cols: List[str]
    fit_metadata: Dict[str, Any] = field(default_factory=dict)


class RoutinePolicy:
    """
    Learns routine testing policy P(T_D=1|C) for each target drug D.

    Two methods:
      - 'empirical': direct proportion in each context (requires sufficient n).
      - 'ml': machine learning model (e.g., XGBoost) with calibration.

    After fitting, provides:
      - Calibration assessment on a hold-out set.
      - Computation of continuous escalation score on test data.
    """

    def __init__(
        self,
        context_cols: List[str],
        method: str = "empirical",
        min_context_n: int = 100,
        model_type: str = "xgb",
        calibrate: bool = True,
        calibration_method: str = "isotonic",
        calibration_cv: int = 5,
        min_prob: float = 0.01,
        random_state: int = 42,
    ):
        """
        Args:
            context_cols: List of column names defining context C.
            method: 'empirical' or 'ml'.
            min_context_n: Minimum number of rows in a context for empirical method.
            model_type: ML model type (used if method='ml').
            calibrate: Whether to calibrate ML model (recommended True).
            calibration_method: 'isotonic' or 'sigmoid'.
            calibration_cv: Number of CV folds for calibration.
            min_prob: Minimum probability for clipping (avoids extreme weights).
            random_state: Random seed.
        """
        self.context_cols = context_cols
        self.method = method.lower()
        self.min_context_n = min_context_n
        self.model_type = model_type
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.calibration_cv = calibration_cv
        self.min_prob = min_prob
        self.random_state = random_state

        self._fitted = False
        self._policy_result: Optional[PolicyResult] = None

    def fit(
        self,
        df_train: pd.DataFrame,
        flags_train: pd.DataFrame,
        target_codes: List[str],
    ) -> PolicyResult:
        """
        Fit routine policy for each target code.

        Args:
            df_train: Training DataFrame (contains context columns).
            flags_train: Training flags DataFrame (contains _T columns).
            target_codes: List of antibiotic codes to fit policies for.

        Returns:
            PolicyResult containing fitted models.
        """
        missing = [c for c in self.context_cols if c not in df_train.columns]
        if missing:
            raise ValueError(f"Context columns missing from df_train: {missing}")

        models = {}
        for code in target_codes:
            tcol = f"{code}_T"
            if tcol not in flags_train.columns:
                logger.warning(f"Target code {code} missing tested column {tcol}, skipping.")
                continue

            # Extract context and outcome
            ctx_df = df_train[self.context_cols].copy()
            y = flags_train[tcol].astype(int).to_numpy()

            if self.method == "empirical":
                model = self._fit_empirical(ctx_df, y, code)
            elif self.method == "ml":
                model = self._fit_ml(ctx_df, y, code)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            models[code] = model

        self._policy_result = PolicyResult(
            models=models,
            context_cols=self.context_cols,
            fit_metadata={
                "method": self.method,
                "min_context_n": self.min_context_n,
                "model_type": self.model_type,
                "calibrate": self.calibrate,
                "random_state": self.random_state,
            }
        )
        self._fitted = True
        return self._policy_result

    def _fit_empirical(self, ctx_df: pd.DataFrame, y: np.ndarray, code: str) -> PolicyModel:
        """Empirical: group by context, compute proportion."""
        # Combine context columns into a single key
        ctx_df = ctx_df.astype("string").fillna("NA")
        ctx_key = ctx_df.apply(lambda row: "|".join(row), axis=1)

        # Compute empirical probabilities per context
        grouped = pd.DataFrame({"key": ctx_key, "y": y}).groupby("key")["y"]
        prob_series = grouped.mean()
        count_series = grouped.count()

        # Keep only contexts with sufficient n
        valid_keys = count_series[count_series >= self.min_context_n].index
        prob_dict = prob_series[valid_keys].to_dict()

        # Build a simple lookup model
        class EmpiricalLookup:
            def __init__(self, prob_dict, default_prob):
                self.prob_dict = prob_dict
                self.default_prob = default_prob

            def predict_proba(self, ctx_keys):
                probs = np.array([self.prob_dict.get(k, self.default_prob) for k in ctx_keys])
                # sklearn predict_proba returns (n_samples, 2) with P(0), P(1)
                return np.column_stack((1 - probs, probs))

        # Default probability: overall mean? Could also use 0.5? But better use global mean.
        default_prob = np.mean(y)

        lookup_model = EmpiricalLookup(prob_dict, default_prob)

        # For calibration curve later, we need the mapping; store prob_dict
        model = PolicyModel(
            code=code,
            model=lookup_model,
            context_cols=self.context_cols,
            method="empirical",
            min_prob=self.min_prob,
        )
        # Store additional data for calibration assessment
        model.prob_dict = prob_dict
        model.default_prob = default_prob
        return model

    def _fit_ml(self, ctx_df: pd.DataFrame, y: np.ndarray, code: str) -> PolicyModel:
        """ML: one-hot encode context, train classifier, optionally calibrate."""
        # One-hot encode context columns
        ctx_encoded = pd.get_dummies(ctx_df.astype("string").fillna("NA"), drop_first=True)
        X = ctx_encoded.to_numpy(dtype="float32")

        # Build base model
        base = build_sklearn_model(self.model_type, random_state=self.random_state)

        # Calibrate if requested
        if self.calibrate:
            model = CalibratedClassifierCV(
                base,
                method=self.calibration_method,
                cv=self.calibration_cv,
                ensemble=True,
            )
            model.fit(X, y)
        else:
            base.fit(X, y)
            model = base

        # Wrap in PolicyModel
        pm = PolicyModel(
            code=code,
            model=model,
            context_cols=self.context_cols,
            method="ml",
            min_prob=self.min_prob,
        )
        # Store feature names for prediction
        pm.feature_names = ctx_encoded.columns.tolist()
        return pm

    def validate_calibration(
        self,
        df_val: pd.DataFrame,
        flags_val: pd.DataFrame,
        target_codes: Optional[List[str]] = None,
        n_bins: int = 10,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Assess calibration of fitted policies on a validation set.

        Args:
            df_val: Validation DataFrame.
            flags_val: Validation flags.
            target_codes: Subset of codes to validate; if None, use all fitted.
            n_bins: Number of bins for calibration curve.

        Returns:
            Dictionary: code -> {
                'brier_score': float,
                'calibration_curve': (fraction_positives, mean_predicted),
                'plot_data': (prob_true, prob_pred)
            }
        """
        if not self._fitted:
            raise RuntimeError("Must fit before validation.")

        if target_codes is None:
            target_codes = list(self._policy_result.models.keys())

        results = {}
        for code in target_codes:
            if code not in self._policy_result.models:
                continue

            tcol = f"{code}_T"
            if tcol not in flags_val.columns:
                continue

            y_true = flags_val[tcol].astype(int).to_numpy()
            y_pred = self.predict_proba(df_val, code)[:, 1]  # P(T=1)

            # Brier score
            brier = brier_score_loss(y_true, y_pred)

            # Calibration curve
            prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy="uniform")

            results[code] = {
                "brier_score": brier,
                "calibration_curve": (prob_true, prob_pred),
                "plot_data": (prob_true, prob_pred),
            }

            # Store in model object
            self._policy_result.models[code].brier_score = brier
            self._policy_result.models[code].calibration_curve = {
                "prob_true": prob_true,
                "prob_pred": prob_pred,
            }

        return results

    def predict_proba(self, df: pd.DataFrame, code: str) -> np.ndarray:
        """
        Predict P(T_D=1|C) for each row in df.

        Returns:
            Array of shape (n_samples, 2) with [P(0), P(1)].
        """
        if not self._fitted or code not in self._policy_result.models:
            raise ValueError(f"Policy for code {code} not fitted.")

        model = self._policy_result.models[code]
        ctx_df = df[self.context_cols].copy()

        if model.method == "empirical":
            # Build context keys
            ctx_key = ctx_df.astype("string").fillna("NA").apply(
                lambda row: "|".join(row), axis=1
            )
            probs = np.array([model.prob_dict.get(k, model.default_prob) for k in ctx_key])
            probs = np.clip(probs, self.min_prob, 1 - self.min_prob)
            return np.column_stack((1 - probs, probs))

        else:  # ml
            # One-hot encode using same columns as training
            ctx_encoded = pd.get_dummies(ctx_df.astype("string").fillna("NA"), drop_first=True)
            # Ensure columns match training (fill missing with 0)
            ctx_encoded = ctx_encoded.reindex(columns=model.feature_names, fill_value=0)
            X = ctx_encoded.to_numpy(dtype="float32")
            proba = model.model.predict_proba(X)
            # Clip probabilities
            proba = np.clip(proba, self.min_prob, 1 - self.min_prob)
            return proba

    def compute_escalation_score(
        self,
        df: pd.DataFrame,
        flags: pd.DataFrame,
        code: str,
        *,
        stabilize: bool = True,
        trim_percentile: float = 99.0,
    ) -> np.ndarray:
        """
        Compute continuous escalation score Y* = T_D / P(T_D=1|C).

        Args:
            df: Test DataFrame.
            flags: Test flags.
            code: Target antibiotic code.
            stabilize: If True, cap the weights at the given percentile to avoid extremes.
            trim_percentile: Percentile for capping (e.g., 99.0).

        Returns:
            Array of escalation scores (same length as df).
        """
        tcol = f"{code}_T"
        if tcol not in flags.columns:
            raise ValueError(f"Tested column {tcol} missing from flags.")

        T = flags[tcol].astype(int).to_numpy()
        p = self.predict_proba(df, code)[:, 1]  # P(T=1|C)

        # Compute raw score
        score = np.zeros_like(T, dtype=float)
        tested_mask = (T == 1)
        score[tested_mask] = 1.0 / p[tested_mask]

        if stabilize:
            # Cap at percentile among tested
            cap = np.percentile(score[tested_mask], trim_percentile) if tested_mask.any() else np.inf
            score[tested_mask] = np.clip(score[tested_mask], None, cap)

        return score

    def get_fitted_models(self) -> PolicyResult:
        """Return the fitted policy result."""
        if not self._fitted:
            raise RuntimeError("Not fitted yet.")
        return self._policy_result