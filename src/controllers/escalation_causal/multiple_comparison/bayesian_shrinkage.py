"""
Bayesian Hierarchical Shrinkage for Multiple Comparisons
=========================================================

Fits a Bayesian hierarchical model to shrink noisy estimates from multiple
trigger–target pairs, borrowing strength across pairs. Provides shrunken estimates,
posterior probabilities, and visualization utilities.

Model:
    estimate_i ~ Normal(theta_i, se_i^2)
    theta_i ~ Normal(0, tau^2)   (or a heavier‑tailed prior)
    tau ~ HalfCauchy(0, 1)       (or other weakly informative prior)

Uses PyMC for posterior sampling.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional import of PyMC
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    logger.warning("PyMC not installed. Bayesian shrinkage will not be available.")


class BayesianShrinkage:
    """
    Bayesian hierarchical model for shrinkage of multiple estimates.

    Fits:
        y_i ~ N(theta_i, sigma_i^2)   (sigma_i known from SE)
        theta_i ~ N(0, tau^2)
        tau ~ HalfCauchy(1)           (weakly informative)

    Provides shrunken estimates (posterior means) and posterior probabilities.
    """

    def __init__(
        self,
        tau_prior: str = "halfcauchy",
        tau_prior_scale: float = 1.0,
        mu_prior_mean: float = 0.0,
        mu_prior_sd: float = 1.0,
        random_seed: int = 42,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.95,
        progressbar: bool = False,
    ):
        """
        Args:
            tau_prior: Prior for between‑pair heterogeneity. Options: 'halfcauchy', 'halfnormal', 'exp'.
            tau_prior_scale: Scale parameter for tau prior.
            mu_prior_mean: Mean of prior for overall mean (if used). Currently model has zero mean.
            mu_prior_sd: SD of prior for overall mean.
            random_seed: Random seed for sampling.
            draws: Number of posterior draws per chain.
            tune: Number of tuning steps.
            chains: Number of MCMC chains.
            target_accept: Target acceptance rate for NUTS.
            progressbar: Whether to show progress bar.
        """
        if not PYMC_AVAILABLE:
            raise ImportError("PyMC is required. Install with: pip install pymc arviz")

        self.tau_prior = tau_prior
        self.tau_prior_scale = tau_prior_scale
        self.mu_prior_mean = mu_prior_mean
        self.mu_prior_sd = mu_prior_sd
        self.random_seed = random_seed
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.progressbar = progressbar

        self._fitted = False
        self._trace = None
        self._summary = None
        self._pair_names = None
        self._data = None

    def fit(self, df: pd.DataFrame, estimate_col: str = "rd", se_col: str = "se") -> BayesianShrinkage:
        """
        Fit the hierarchical model.

        Args:
            df: DataFrame containing at least columns for pair identifier, estimate, and standard error.
            estimate_col: Name of column with point estimates.
            se_col: Name of column with standard errors.

        The DataFrame must have an index or a column 'pair' identifying each pair.
        Rows with missing estimates or SE are dropped.
        """
        if "pair" not in df.columns:
            # Use index as pair identifier
            df = df.reset_index().rename(columns={"index": "pair"})

        # Keep only necessary columns and drop missing
        data = df[["pair", estimate_col, se_col]].dropna().copy()
        if data.empty:
            raise ValueError("No valid data after dropping missing.")

        self._pair_names = data["pair"].tolist()
        y = data[estimate_col].values
        sigma = data[se_col].values
        n_pairs = len(y)

        # Build PyMC model
        with pm.Model() as model:
            # Hyperprior for between‑pair SD
            if self.tau_prior == "halfcauchy":
                tau = pm.HalfCauchy("tau", beta=self.tau_prior_scale)
            elif self.tau_prior == "halfnormal":
                tau = pm.HalfNormal("tau", sigma=self.tau_prior_scale)
            elif self.tau_prior == "exp":
                tau = pm.Exponential("tau", lam=1.0/self.tau_prior_scale)
            else:
                raise ValueError(f"Unknown tau_prior: {self.tau_prior}")

            # Prior for overall mean (optional, but we center at 0)
            # theta_i ~ Normal(0, tau)
            theta = pm.Normal("theta", mu=0, sigma=tau, shape=n_pairs)

            # Likelihood
            pm.Normal("y_obs", mu=theta, sigma=sigma, observed=y)

            # Sample
            self._trace = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                target_accept=self.target_accept,
                random_seed=self.random_seed,
                progressbar=self.progressbar,
                return_inferencedata=True,
            )

        self._fitted = True
        self._data = data
        return self

    def summary(self) -> pd.DataFrame:
        """
        Return a DataFrame with original and shrunken estimates,
        posterior SD, and posterior probability that theta > 0.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted.")

        # Extract posterior of theta
        theta_posterior = az.extract(self._trace, var_names=["theta"]).values  # shape (n_draws, n_pairs)
        theta_mean = theta_posterior.mean(axis=0)
        theta_sd = theta_posterior.std(axis=0)
        prob_pos = (theta_posterior > 0).mean(axis=0)

        # Combine with original data
        df_out = self._data.copy()
        df_out["theta_shrunken"] = theta_mean
        df_out["theta_sd"] = theta_sd
        df_out["prob_positive"] = prob_pos
        df_out["p_value"] = 2 * (1 - stats.norm.cdf(abs(df_out["rd"] / df_out["se"])))  # approximate
        return df_out

    def plot_posterior_intervals(
        self,
        ax=None,
        sort_by: str = "estimate",
        top_n: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """
        Plot forest plot with original (grey) and shrunken (colored) estimates and intervals.

        Args:
            ax: Matplotlib axes.
            sort_by: How to sort pairs: 'estimate' (original) or 'shrunken'.
            top_n: Show only top N pairs by absolute shrunken estimate.
        """
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        if not self._fitted:
            raise RuntimeError("Model not fitted.")

        df = self.summary()
        if sort_by == "estimate":
            df = df.sort_values("rd", ascending=False)
        elif sort_by == "shrunken":
            df = df.sort_values("theta_shrunken", ascending=False)
        else:
            raise ValueError("sort_by must be 'estimate' or 'shrunken'")

        if top_n is not None:
            df = df.head(top_n)

        y_pos = np.arange(len(df))
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(df))))

        # Original 95% CI (using SE)
        ci_low_orig = df["rd"] - 1.96 * df["se"]
        ci_high_orig = df["rd"] + 1.96 * df["se"]
        ax.hlines(y_pos, ci_low_orig, ci_high_orig, color="gray", alpha=0.5, linewidth=1, label="Original 95% CI")

        # Shrunken 95% credible intervals (using posterior SD)
        ci_low_shrink = df["theta_shrunken"] - 1.96 * df["theta_sd"]
        ci_high_shrink = df["theta_shrunken"] + 1.96 * df["theta_sd"]
        ax.hlines(y_pos, ci_low_shrink, ci_high_shrink, color="blue", alpha=0.7, linewidth=2, label="Shrunken 95% CrI")

        # Points
        ax.scatter(df["rd"], y_pos, color="gray", alpha=0.7, marker="o", label="Original estimate")
        ax.scatter(df["theta_shrunken"], y_pos, color="blue", marker="D", label="Shrunken estimate")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["pair"])
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Risk Difference")
        ax.set_title("Bayesian Shrinkage of Pair Estimates")
        ax.legend(loc="best")
        plt.tight_layout()
        return ax

    def plot_prob_positive(self, ax=None, top_n: Optional[int] = None, **kwargs) -> Any:
        """
        Plot posterior probability that theta > 0 for each pair.
        """
        import matplotlib.pyplot as plt
        df = self.summary()
        if top_n is not None:
            df = df.sort_values("prob_positive", ascending=False).head(top_n)
        else:
            df = df.sort_values("prob_positive", ascending=False)

        y_pos = np.arange(len(df))
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(df))))

        ax.barh(y_pos, df["prob_positive"].values, color="steelblue")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["pair"])
        ax.axvline(0.95, color="red", linestyle="--", label="95% threshold")
        ax.axvline(0.5, color="gray", linestyle=":")
        ax.set_xlabel("Posterior Probability theta > 0")
        ax.set_title("Posterior Probability of Positive Effect")
        ax.set_xlim(0, 1)
        ax.legend()
        plt.tight_layout()
        return ax