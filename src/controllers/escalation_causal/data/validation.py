"""
Data Validation Module
======================

Checks for data quality, alignment, positivity, and other assumptions.
"""

import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class DataValidator:
    """Collection of data validation methods."""

    @staticmethod
    def validate_alignment(df: pd.DataFrame, flags: pd.DataFrame) -> None:
        """Check that df and flags have identical index."""
        if not df.index.equals(flags.index):
            raise ValueError("df and flags must have identical index.")

    @staticmethod
    def check_positivity(
        df: pd.DataFrame,
        flags: pd.DataFrame,
        all_codes: List[str],
        min_prob: float = 0.01,
        warn_only: bool = True,
    ) -> None:
        """
        Check positivity of testing probabilities for each trigger.

        For each trigger code, estimate P(T=1|X) using a simple logistic regression
        and check if any tested observation has predicted probability below min_prob.
        Skips codes with no tested isolates or constant outcome.
        """
        # Use a simple set of covariates: all numeric columns in df.
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            logger.warning("No numeric columns found; skipping positivity check.")
            return

        X = df[numeric_cols].fillna(0).to_numpy()

        for code in all_codes:
            tcol = f"{code}_T"
            if tcol not in flags.columns:
                continue

            y = flags[tcol].astype(int).to_numpy()

            # Skip if no tested isolates or constant outcome
            if y.sum() == 0:
                logger.debug(f"Trigger {code}: no tested isolates, skipping positivity check.")
                continue
            if len(np.unique(y)) < 2:
                logger.debug(f"Trigger {code}: outcome constant (all {y[0]}), skipping positivity check.")
                continue

            # Fit logistic regression
            try:
                model = LogisticRegression(max_iter=1000, solver="lbfgs")
                model.fit(X, y)
                proba = model.predict_proba(X)[:, 1]
            except Exception as e:
                logger.warning(f"Trigger {code}: logistic regression failed: {e}")
                continue

            tested_mask = (y == 1)
            min_prob_tested = proba[tested_mask].min()
            if min_prob_tested < min_prob:
                msg = f"Trigger {code}: minimum predicted testing probability among tested = {min_prob_tested:.4f} < {min_prob}"
                if warn_only:
                    logger.warning(msg)
                else:
                    raise ValueError(msg)

    @staticmethod
    def check_missingness(df: pd.DataFrame, threshold: float = 0.5) -> List[str]:
        """
        Return columns with missing proportion above threshold.
        """
        missing_frac = df.isnull().mean()
        high_missing = missing_frac[missing_frac > threshold].index.tolist()
        if high_missing:
            logger.warning(f"Columns with >{threshold*100:.0f}% missing: {high_missing}")
        return high_missing