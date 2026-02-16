"""
TMLE for Risk Difference with Continuous Outcome and Sampling Weights
=====================================================================

Implements cross‑fitted Targeted Maximum Likelihood Estimation (TMLE) for the
average treatment effect (risk difference) of a binary treatment on a continuous
outcome, with sample weights (e.g., inverse probability of testing weights).

The parameter of interest is the weighted average:
    ψ = E[ w * (E[Y|A=1,X] - E[Y|A=0,X]) ] / E[w]
where w are the sampling weights.

The TMLE algorithm follows the standard steps:
  1. Obtain initial estimates Q0(A,W) of E[Y|A,W].
  2. Compute the clever covariate for the risk difference parameter.
  3. Update the initial estimates by fitting a logistic regression (or linear,
     depending on outcome bound) of Y on the clever covariate with offset.
  4. Compute the targeted estimates and the final parameter.
  5. Obtain influence curve‑based standard errors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import stats

from ..utils import clip_prob

logger = logging.getLogger(__name__)


@dataclass
class TMLE_Result:
    """Result of a TMLE fit."""
    psi: float                     # Risk difference estimate
    se: float                      # Standard error (influence curve based)
    ci_low: float                  # Lower 95% confidence bound
    ci_high: float                 # Upper 95% confidence bound
    p_value: float                  # Two-sided p-value
    epsilon: float                  # Fluctuation parameter
    converged: bool                 # Whether targeting step converged
    n_used: int                     # Number of observations used (tested, weighted)


class TMLE_RD:
    """
    Targeted Maximum Likelihood Estimation for Risk Difference.

    Requires cross‑fitted initial estimates of:
        - Q1(W) = E[Y | A=1, W]  (for treated)
        - Q0(W) = E[Y | A=0, W]  (for untreated)
        - g(W)  = P(A=1 | W, tested)  (propensity score among tested)
        - (optionally) the testing weights w = 1/P(T=1|X) are provided separately.

    The parameter is the weighted average of (Q1 - Q0) with weights w.
    """

    def __init__(
        self,
        *,
        bounds: Tuple[float, float] = (0.0, 1.0e6),   # outcome bounds (Y* is non‑negative, but no upper bound)
        targeted_regularization: bool = True,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        """
        Args:
            bounds: Lower and upper bounds of the outcome (used to bound the
                    logistic fluctuation if we transform to [0,1] scale).
                    For unbounded outcomes, we can use linear fluctuation.
            targeted_regularization: If True, uses logistic fluctuation with
                                     scaled outcome; if False, uses linear.
            max_iter: Maximum iterations for targeting step (usually converges in 1).
            tol: Convergence tolerance for epsilon change.
        """
        self.bounds = bounds
        self.targeted_regularization = targeted_regularization
        self.max_iter = max_iter
        self.tol = tol

    def compute(
        self,
        A: np.ndarray,
        Y: np.ndarray,
        weights: np.ndarray,
        Q1: np.ndarray,
        Q0: np.ndarray,
        g1: np.ndarray,
        *,
        return_all: bool = False,
    ) -> TMLE_Result:
        """
        Perform TMLE targeting.

        Args:
            A: Treatment indicator (0/1), length n.
            Y: Outcome (continuous), length n.
            weights: Sampling weights (e.g., testing IPW), length n.
            Q1: Initial estimate of E[Y|A=1, X], length n.
            Q0: Initial estimate of E[Y|A=0, X], length n.
            g1: Propensity score P(A=1|X), length n.

        Returns:
            TMLE_Result object.
        """
        n = len(A)
        if not (len(Y) == len(Q1) == len(Q0) == len(g1) == len(weights)):
            raise ValueError("All input arrays must have same length.")

        # Clip extreme propensities for stability
        g1 = clip_prob(g1, 1e-6, 1 - 1e-6)
        g0 = 1 - g1

        # Clever covariate for the risk difference (ATE)
        # H(A,W) = A/g1(W) - (1-A)/g0(W)
        H = A / g1 - (1 - A) / g0

        # Initial prediction of the outcome (based on observed A)
        Q_init = np.where(A == 1, Q1, Q0)

        # If outcome is bounded and we use logistic fluctuation, we need to
        # scale Y to [0,1]. But Y* is unbounded, so we'll use linear fluctuation.
        # However, if we want to respect bounds, we can use a quasi-logistic
        # fluctuation on the logit scale after scaling to [0,1] using min/max.
        # For simplicity, we start with linear fluctuation (unbounded).

        # ---- Targeting step (linear fluctuation) ----
        # Fit epsilon via weighted OLS without intercept, with offset Q_init.
        # We solve: Y - Q_init = epsilon * H + noise, weighted by weights.
        # Equivalent to: fit a weighted linear model with no intercept.
        from sklearn.linear_model import LinearRegression

        # Prepare data
        X_epsilon = H.reshape(-1, 1)
        y_target = Y - Q_init

        # Fit weighted linear regression
        lr = LinearRegression(fit_intercept=False)
        lr.fit(X_epsilon, y_target, sample_weight=weights)
        epsilon = lr.coef_[0]

        # Updated predictions
        Q1_star = Q1 + epsilon * (1 / g1)   # because H for A=1 is 1/g1
        Q0_star = Q0 - epsilon * (1 / g0)   # because H for A=0 is -1/g0

        # Final estimate: weighted average of (Q1_star - Q0_star)
        psi = np.average(Q1_star - Q0_star, weights=weights)

        # ---- Influence curve for standard error ----
        # IC = w * [ H*(Y - Q_star) + (Q1_star - Q0_star - psi) ] / mean(w)
        # But careful: we need the influence function for the weighted estimator.
        # Standard approach: treat weights as known, then IC_i = (weights_i / mean(weights)) * (
        #       H_i * (Y_i - Q_star_i) + (Q1_star_i - Q0_star_i - psi) )
        # This gives variance of the weighted estimator.
        # We'll compute the empirical IC.

        Q_star = np.where(A == 1, Q1_star, Q0_star)
        H_used = H  # already computed

        # Component 1: correction term
        corr = H_used * (Y - Q_star)
        # Component 2: centered difference
        centered = (Q1_star - Q0_star) - psi

        # Influence function (scaled by weights)
        mean_w = np.mean(weights)
        ic = (weights / mean_w) * (corr + centered)

        # Variance of psi
        var_psi = np.var(ic, ddof=1) / n   # asymptotic variance
        se = np.sqrt(var_psi)

        # Confidence interval (assuming normality)
        z = stats.norm.ppf(0.975)
        ci_low = psi - z * se
        ci_high = psi + z * se

        # P-value (two-sided test of psi=0)
        p_value = 2 * (1 - stats.norm.cdf(abs(psi / se))) if se > 0 else 1.0

        # Check convergence (epsilon change)
        converged = True  # one-step TMLE always converges in one step if linear

        result = TMLE_Result(
            psi=psi,
            se=se,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            epsilon=epsilon,
            converged=converged,
            n_used=n,
        )

        if return_all:
            result.Q1_star = Q1_star
            result.Q0_star = Q0_star
            result.ic = ic

        return result


# Alternative fluctuation using bounded outcome transformation
# This version scales Y to [0,1] using empirical min/max, then uses logistic fluctuation,
# then back‑transforms. Useful if outcome has natural bounds.
class TMLE_RD_Bounded(TMLE_RD):
    def compute(self, A, Y, weights, Q1, Q0, g1, return_all=False):
        # Determine bounds from Y (or use provided)
        y_min, y_max = self.bounds
        if y_min is None:
            y_min = Y.min()
        if y_max is None:
            y_max = Y.max()
        if y_max <= y_min:
            raise ValueError("Invalid bounds: max <= min")

        # Scale Y to [0,1]
        Y_scaled = (Y - y_min) / (y_max - y_min)
        # Also scale initial Q predictions (they are on original scale)
        Q1_scaled = (Q1 - y_min) / (y_max - y_min)
        Q0_scaled = (Q0 - y_min) / (y_max - y_min)

        # Compute clever covariate (same as before)
        g1 = clip_prob(g1, 1e-6, 1 - 1e-6)
        g0 = 1 - g1
        H = A / g1 - (1 - A) / g0

        # Initial prediction on scaled scale
        Q_init_scaled = np.where(A == 1, Q1_scaled, Q0_scaled)

        # Logistic fluctuation on scaled outcome (which lies in [0,1])
        # We fit epsilon by solving: logit(Y_scaled) ~ offset + epsilon * H
        # But Y_scaled may be 0 or 1, so we adjust to avoid logit(0/1).
        eps = 1e-4
        Y_logit = np.log((Y_scaled + eps) / (1 - Y_scaled + eps))
        offset = np.log((Q_init_scaled + eps) / (1 - Q_init_scaled + eps))

        # Fit epsilon via weighted linear regression (no intercept) on H
        from sklearn.linear_model import LinearRegression

        X_epsilon = H.reshape(-1, 1)
        y_target = Y_logit - offset
        lr = LinearRegression(fit_intercept=False)
        lr.fit(X_epsilon, y_target, sample_weight=weights)
        epsilon = lr.coef_[0]

        # Update predictions on logit scale
        Q1_logit_star = offset + epsilon * (1 / g1)
        Q0_logit_star = offset - epsilon * (1 / g0)

        # Back‑transform to [0,1]
        def inv_logit(x):
            return 1 / (1 + np.exp(-x))

        Q1_star_scaled = inv_logit(Q1_logit_star)
        Q0_star_scaled = inv_logit(Q0_logit_star)

        # Rescale to original
        Q1_star = Q1_star_scaled * (y_max - y_min) + y_min
        Q0_star = Q0_star_scaled * (y_max - y_min) + y_min

        # Final estimate
        psi = np.average(Q1_star - Q0_star, weights=weights)

        # Influence curve on original scale (complicated; we approximate using delta method or bootstrap)
        # For simplicity, we use the linear version's IC but may be biased.
        # Instead, we can bootstrap or use the linear IC as approximation.
        # We'll just use the linear version's IC for now.
        Q_star = np.where(A == 1, Q1_star, Q0_star)
        corr = H * (Y - Q_star)
        centered = (Q1_star - Q0_star) - psi
        mean_w = np.mean(weights)
        ic = (weights / mean_w) * (corr + centered)
        var_psi = np.var(ic, ddof=1) / len(A)
        se = np.sqrt(var_psi)

        ci_low = psi - 1.96 * se
        ci_high = psi + 1.96 * se
        p_value = 2 * (1 - stats.norm.cdf(abs(psi / se))) if se > 0 else 1.0

        result = TMLE_Result(
            psi=psi,
            se=se,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            epsilon=epsilon,
            converged=True,
            n_used=len(A),
        )
        return result