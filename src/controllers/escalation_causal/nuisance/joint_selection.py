"""
Joint Model for Testing and Resistance (Selection Model)
=========================================================

Implements a bivariate probit model with sample selection to jointly estimate
P(T=1|X) and P(A=1|X, T=1). This accounts for unmeasured common causes
of testing and resistance, potentially reducing bias in the propensity scores.

If the correlation ρ between the error terms is zero, the model reduces to two
separate probits. If ρ ≠ 0, the joint model yields more efficient and less
biased estimates of the conditional probabilities.

Uses statsmodels if available; otherwise falls back to separate models.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ..utils.weights import clip_prob, stable_quantiles

logger = logging.getLogger(__name__)

try:
    import statsmodels.api as sm
    from statsmodels.base.model import GenericLikelihoodModel
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not available. JointSelectionModel will use fallback.")


class BivariateProbitSelection(GenericLikelihoodModel if STATSMODELS_AVAILABLE else object):
    """
    Bivariate probit model with sample selection.

    Equations:
        T* = Xβ + ε1,   T = 1 if T* > 0
        A* = Xγ + ε2,   A = 1 if A* > 0, observed only when T=1
        (ε1, ε2) ~ bivariate normal with correlation ρ.

    Log-likelihood is built from:
        - For T=0: P(T=0) = Φ(-Xβ)
        - For T=1, A=1: P(T=1, A=1) = Φ2(Xβ, Xγ, ρ)
        - For T=1, A=0: P(T=1, A=0) = Φ(Xβ) - Φ2(Xβ, Xγ, ρ)
    where Φ is univariate normal CDF, Φ2 is bivariate normal CDF with correlation ρ.
    """

    def __init__(self, endog, exog, **kwargs):
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for BivariateProbitSelection.")
        super().__init__(endog, exog, **kwargs)
        # endog should be a 2-column array: [T, A] where A is np.nan when T=0
        self._initialize()

    def _initialize(self):
        self.n_params = self.exog.shape[1] * 2 + 1  # β, γ, ρ
        self.start_params = np.zeros(self.n_params)
        # Set starting values: fit separate probits for β and γ
        exog = self.exog
        T = self.endog[:, 0]
        A = self.endog[:, 1]

        # Probit for T using all observations
        probit_T = sm.Probit(T, exog)
        probit_T_results = probit_T.fit(disp=0)
        beta_start = probit_T_results.params

        # Probit for A using only observations with T=1
        mask = (T == 1)
        if mask.sum() > 0:
            probit_A = sm.Probit(A[mask], exog[mask])
            probit_A_results = probit_A.fit(disp=0)
            gamma_start = probit_A_results.params
        else:
            gamma_start = np.zeros(exog.shape[1])

        self.start_params[:exog.shape[1]] = beta_start
        self.start_params[exog.shape[1]:2*exog.shape[1]] = gamma_start
        self.start_params[-1] = 0.0  # ρ starting at 0

    def loglike(self, params):
        exog = self.exog
        T = self.endog[:, 0]
        A = self.endog[:, 1]

        k = exog.shape[1]
        beta = params[:k]
        gamma = params[k:2*k]
        rho = params[-1]
        # Clip rho to [-0.99, 0.99] for numerical stability
        rho = np.clip(rho, -0.99, 0.99)

        Xb = exog @ beta
        Xg = exog @ gamma

        ll = 0.0
        for i in range(len(T)):
            if T[i] == 0:
                # P(T=0) = Φ(-Xb)
                ll += np.log(stats.norm.cdf(-Xb[i]) + 1e-12)
            else:
                if A[i] == 1:
                    # P(T=1, A=1) = Φ2(Xb, Xg, ρ)
                    ll += np.log(self._bvnorm_cdf(Xb[i], Xg[i], rho) + 1e-12)
                else:  # A[i] == 0
                    # P(T=1, A=0) = Φ(Xb) - Φ2(Xb, Xg, ρ)
                    prob1 = stats.norm.cdf(Xb[i])
                    prob11 = self._bvnorm_cdf(Xb[i], Xg[i], rho)
                    ll += np.log(prob1 - prob11 + 1e-12)
        return ll

    @staticmethod
    def _bvnorm_cdf(z1, z2, rho):
        """Bivariate normal CDF with correlation rho, zero means, unit variances."""
        from scipy.stats import multivariate_normal
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        return multivariate_normal.cdf([z1, z2], mean=mean, cov=cov)

    def fit(self, start_params=None, method='bfgs', maxiter=1000, disp=0, **kwargs):
        if start_params is None:
            start_params = self.start_params
        return super().fit(start_params=start_params, method=method, maxiter=maxiter, disp=disp, **kwargs)


class JointSelectionModel:
    """
    Joint selection model for testing and resistance.

    Fits a bivariate probit with sample selection to obtain:
        - P(T=1 | X)  (testing propensity)
        - P(A=1 | X, T=1)  (resistance propensity conditional on testing)

    These can be used in place of separate models in the causal pipeline.
    The model is fitted on the full dataset (all isolates, both tested and untested).

    If statsmodels is not available or fitting fails, falls back to separate
    probit models (equivalent to ρ = 0).
    """

    def __init__(
        self,
        min_prob: float = 0.01,
        random_state: int = 42,
    ):
        self.min_prob = min_prob
        self.random_state = random_state
        self._fitted = False
        self._model = None
        self._beta = None   # coefficients for testing equation
        self._gamma = None  # coefficients for resistance equation
        self._rho = 0.0     # correlation

    def fit(self, X: np.ndarray, T: np.ndarray, A: np.ndarray) -> JointSelectionModel:
        """
        Fit the joint model.

        Args:
            X: Covariate matrix (n_samples, n_features). Should include intercept if desired.
            T: Testing indicator (0/1), observed for all.
            A: Resistance indicator (0/1), observed only where T=1; elsewhere can be any value (ignored).
        """
        n = len(X)
        if n == 0:
            raise ValueError("Empty input.")

        # Ensure A is masked for T=0 – convert to float to allow NaN
        A_obs = A.astype(float).copy()
        A_obs[T == 0] = np.nan

        if STATSMODELS_AVAILABLE:
            try:
                # Add constant if not already present? We'll assume X already has intercept if needed.
                # For simplicity, we'll let the user decide; we won't add automatically.
                exog = X
                endog = np.column_stack([T, A_obs])

                model = BivariateProbitSelection(endog, exog)
                res = model.fit(maxiter=1000, disp=0)
                self._model = res
                k = X.shape[1]
                self._beta = res.params[:k]
                self._gamma = res.params[k:2*k]
                self._rho = res.params[-1]
                self._fitted = True
                logger.info("Joint bivariate probit fitted successfully.")
                return self
            except Exception as e:
                logger.warning(f"Joint model fitting failed: {e}. Falling back to separate probits.")
                # Fall through to separate models

        # Fallback: separate probits (approximated by logistic regression)
        from sklearn.linear_model import LogisticRegression
        # Testing model (all data)
        test_model = LogisticRegression(max_iter=1000, solver="lbfgs")
        test_model.fit(X, T)
        self._beta = test_model.coef_[0]  # logit coefficients, not probit – but used for prediction later
        self._test_model = test_model  # store for prediction

        # Resistance model among tested
        tested_mask = (T == 1)
        if tested_mask.sum() > 0:
            resist_model = LogisticRegression(max_iter=1000, solver="lbfgs")
            resist_model.fit(X[tested_mask], A[tested_mask])
            self._gamma = resist_model.coef_[0]
            self._resist_model = resist_model
        else:
            self._gamma = np.zeros(X.shape[1])
            self._resist_model = None

        self._rho = 0.0
        self._fitted = True
        logger.info("Separate logistic models used as fallback.")
        return self

    def predict_p_test(self, X: np.ndarray) -> np.ndarray:
        """Predict P(T=1 | X)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        if self._model is not None:
            # Use joint model: P(T=1) = Φ(Xβ)
            Xb = X @ self._beta
            p = stats.norm.cdf(Xb)
        else:
            # Fallback logistic
            p = self._test_model.predict_proba(X)[:, 1]
        return clip_prob(p, self.min_prob, 1 - self.min_prob)

    def predict_p_resist_given_tested(self, X: np.ndarray) -> np.ndarray:
        """Predict P(A=1 | X, T=1)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        if self._model is not None:
            # P(A=1 | T=1, X) = Φ2(Xβ, Xγ, ρ) / Φ(Xβ)
            Xb = X @ self._beta
            Xg = X @ self._gamma
            prob1 = stats.norm.cdf(Xb)
            prob11 = self._model._bvnorm_cdf(Xb, Xg, self._rho)
            p = prob11 / (prob1 + 1e-12)
            p = np.clip(p, 0, 1)
        else:
            # Fallback logistic
            if self._resist_model is not None:
                p = self._resist_model.predict_proba(X)[:, 1]
            else:
                p = np.zeros(len(X))
        return clip_prob(p, self.min_prob, 1 - self.min_prob)

    def compute_weights(self, p_test: np.ndarray, tested_mask: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Compute IPW weights for tested observations using P(T=1|X) from the joint model.

        Args:
            p_test: Testing probabilities (from predict_p_test).
            tested_mask: Boolean mask for tested observations.

        Returns:
            weights: Array of weights (zero for non‑tested, 1/p_test for tested, capped at 99th percentile).
            diagnostics: Dictionary with quantiles and correlation ρ.
        """
        w = np.zeros_like(p_test, dtype=float)
        if tested_mask.any():
            raw_weights = 1.0 / p_test[tested_mask]
            cap = np.percentile(raw_weights, 99)
            w[tested_mask] = np.clip(raw_weights, None, cap)
        else:
            cap = np.nan

        diag = {
            "p_test_quantiles": stable_quantiles(p_test[tested_mask]) if tested_mask.any() else {},
            "w_quantiles": stable_quantiles(w[tested_mask]) if tested_mask.any() else {},
            "w_cap": float(cap) if not np.isnan(cap) else np.nan,
            "rho": float(self._rho),
        }
        return w, diag