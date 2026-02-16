"""
Causal Forest for Heterogeneous Treatment Effects
==================================================

Estimates conditional average treatment effects (CATE) using generalized random forests
via the econml library (if installed). Provides variable importance and plotting utilities.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional import of econml
try:
    from econml.grf import CausalForest
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    logger.warning("econml not installed. CausalForest will not be available.")


class CausalForestWrapper:
    """
    Wrapper for econml's CausalForest to estimate heterogeneous treatment effects.

    Handles sample weights and provides variable importance and plotting methods.
    """

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 20,
        min_samples_leaf: int = 10,
        max_features: Optional[str] = "sqrt",
        random_state: int = 42,
        verbose: int = 0,
    ):
        """
        Args:
            n_estimators: Number of trees.
            max_depth: Maximum depth of each tree.
            min_samples_leaf: Minimum samples in a leaf.
            max_features: Number of features to consider at each split.
            random_state: Random seed.
            verbose: Verbosity level.
        """
        if not ECONML_AVAILABLE:
            raise ImportError(
                "econml is required for CausalForest. Install with: pip install econml"
            )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.verbose = verbose

        self._model: Optional[CausalForest] = None
        self._feature_names: Optional[list] = None
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None,
    ) -> CausalForestWrapper:
        """
        Fit the causal forest.

        Args:
            X: Covariate matrix (n_samples, n_features).
            treatment: Binary treatment indicator (0/1).
            outcome: Continuous outcome (escalation score).
            sample_weight: Optional sample weights.
            feature_names: Optional list of feature names for interpretation.
        """
        self._feature_names = feature_names if feature_names else [f"X{i}" for i in range(X.shape[1])]

        self._model = CausalForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        # Note: econml's CausalForest.fit does not directly accept sample_weight.
        # We need to use the `sample_weight` parameter? Actually, CausalForest does support sample_weight.
        # Check docs: yes, it has `sample_weight` parameter.
        self._model.fit(
            X,
            treatment,
            outcome,
            sample_weight=sample_weight,
        )
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict CATE for new samples."""
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        return self._model.predict(X)

    def feature_importances(self) -> Dict[str, float]:
        """Return variable importance (based on split frequency)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        importances = self._model.feature_importances_
        return dict(zip(self._feature_names, importances))

    def get_cate_summary(
        self,
        X: np.ndarray,
        group_labels: Optional[np.ndarray] = None,
        groups: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Summarize CATE by user-defined groups (e.g., ward type).

        Args:
            X: Covariate matrix for which to predict CATE.
            group_labels: Array of group labels (same length as X).
            groups: Optional list of group names to include.

        Returns:
            DataFrame with columns: group, mean_cate, std_cate, count.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        cate = self.predict(X)
        if group_labels is None:
            # No grouping: return overall summary
            return pd.DataFrame({
                "group": ["all"],
                "mean_cate": [np.mean(cate)],
                "std_cate": [np.std(cate)],
                "count": [len(cate)],
            })
        else:
            df = pd.DataFrame({"group": group_labels, "cate": cate})
            if groups is not None:
                df = df[df["group"].isin(groups)]
            summary = df.groupby("group")["cate"].agg(["mean", "std", "count"]).reset_index()
            summary.columns = ["group", "mean_cate", "std_cate", "count"]
            return summary

    def plot_cate_distribution(
        self,
        X: Optional[np.ndarray] = None,
        ax=None,
        **kwargs,
    ) -> Any:  # matplotlib axes
        """
        Plot histogram of CATE estimates.

        If X is None, uses the training data if stored? Not stored, so must provide.
        """
        import matplotlib.pyplot as plt
        if X is None:
            raise ValueError("X must be provided for prediction.")
        cate = self.predict(X)
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(cate, bins=30, edgecolor="black", **kwargs)
        ax.set_xlabel("Conditional Average Treatment Effect (CATE)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of CATE")
        return ax

    def plot_variable_importance(
        self,
        top_k: int = 20,
        ax=None,
        **kwargs,
    ) -> Any:
        """Plot variable importance as horizontal bar chart."""
        import matplotlib.pyplot as plt
        imp = self.feature_importances()
        df = pd.DataFrame({"feature": list(imp.keys()), "importance": list(imp.values())})
        df = df.sort_values("importance", ascending=False).head(top_k)
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(df))))
        ax.barh(df["feature"], df["importance"], **kwargs)
        ax.set_xlabel("Importance")
        ax.set_title("Causal Forest Variable Importance")
        ax.invert_yaxis()
        return ax