"""
Main Orchestration Pipeline
===========================

End‑to‑end pipeline for causal estimation of antibiotic escalation:
- Split data into train/test
- Fit routine policy on training data
- Compute continuous escalation scores on test data
- For each (trigger, target) pair:
    - Fit testing model P(T_trigger|X) on test data, compute IPW weights
    - Fit nuisance models (propensity, outcome) among tested isolates
    - Run TMLE to estimate risk difference
- Aggregate results and run optional diagnostics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from joblib import Parallel, delayed

from .config.settings import RunConfig
from .data.validation import DataValidator
from .policy.routine_policy import RoutinePolicy
from .nuisance.testing_model import TestingModel
from .nuisance.joint_selection import JointSelectionModel
from .nuisance.nuisance_fitter import NuisanceModelFitter
from .estimator.tmle import TMLE_RD, TMLE_Result
from .utils.weights import effective_sample_size

logger = logging.getLogger(__name__)


@dataclass
class PairResult:
    """Result for a single trigger–target pair."""
    trigger: str
    target: str
    rd: float
    ci_low: float
    ci_high: float
    p_value: float
    se: float
    ess: float                     # effective sample size among tested
    n_used: int                     # number of tested isolates used
    n_trigger_tested: int           # number tested for trigger
    n_A1: int                       # number resistant among tested
    n_A0: int                       # number susceptible among tested
    baseline_mu0: float             # weighted mean outcome under A=0
    escalation_score_mean: float    # mean escalation score (weighted)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    model_spec: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    skip_reason: str = ""


class CausalPipeline:
    """
    Main pipeline orchestrator.

    Steps:
        1. Validate input data.
        2. Split into train/test using group shuffle (e.g., by lab).
        3. Fit routine policy on training set.
        4. Compute continuous escalation scores on test set for all targets.
        5. For each trigger–target pair:
            a. Fit testing model P(T_trigger=1|X) on test data → weights.
            b. Among tested isolates, fit nuisance models (propensity, outcome) with cross‑fit.
            c. Run TMLE for risk difference.
        6. Collect results.
    """

    def __init__(self, config: RunConfig, n_jobs: int = 1):
        self.config = config
        self.n_jobs = n_jobs
        self._validator = DataValidator()
        self._fitted = False
        self._policy: Optional[RoutinePolicy] = None
        self._results: Optional[pd.DataFrame] = None

    def run(
        self,
        df: pd.DataFrame,
        flags: pd.DataFrame,
        all_codes: List[str],
        pairs: List[Tuple[str, str]],
    ) -> pd.DataFrame:
        """
        Run the full pipeline.

        Args:
            df: Main cohort DataFrame (contains covariates and metadata).
            flags: Antibiotic flags DataFrame (_T and _R columns) aligned with df.
            all_codes: List of all antibiotic codes (used to compute escalation scores for all targets).
            pairs: List of (trigger, target) pairs to estimate.

        Returns:
            DataFrame with one row per pair, containing estimates and diagnostics.
        """
        # 1. Validate
        self._validator.validate_alignment(df, flags)
        self._validator.check_positivity(df, flags, all_codes, self.config.tmle.min_prob)

        # 2. Split
        train_idx, test_idx = self._split_indices(df)
        df_train, df_test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
        flags_train, flags_test = flags.iloc[train_idx].copy(), flags.iloc[test_idx].copy()

        # 3. Fit routine policy on training data
        policy = RoutinePolicy(
            context_cols=self.config.policy.context_cols,
            method=self.config.policy.method,
            min_context_n=self.config.policy.min_context_n,
            model_type=self.config.policy.model_type,
            calibrate=self.config.policy.calibrate,
            calibration_method=self.config.policy.calibration_method,
            calibration_cv=self.config.policy.calibration_cv,
            min_prob=self.config.tmle.min_prob,
            random_state=self.config.split.random_state,
        )
        policy.fit(df_train, flags_train, all_codes)
        self._policy = policy

        # 4. Compute continuous escalation scores on test set for all target codes
        esc_scores = {}
        for code in all_codes:
            if f"{code}_T" in flags_test.columns:
                scores = policy.compute_escalation_score(
                    df_test,
                    flags_test,
                    code,
                    stabilize=self.config.tmle.stabilize_weights,
                    trim_percentile=self.config.tmle.weight_cap_percentile,
                )
                esc_scores[code] = scores

        # 5. Process each pair in parallel
        results = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self._estimate_pair_wrapper)(
                df_test, flags_test, trigger, target, esc_scores
            )
            for trigger, target in pairs
        )

        df_out = pd.DataFrame([vars(r) for r in results])
        self._results = df_out
        self._fitted = True
        return df_out

    def _estimate_pair_wrapper(self, df, flags, trigger, target, esc_scores):
        """Wrapper for _estimate_pair to catch exceptions and return PairResult."""
        try:
            return self._estimate_pair(df, flags, trigger, target, esc_scores)
        except Exception as e:
            logger.exception(f"Failed to estimate {trigger} -> {target}: {e}")
            return PairResult(
                trigger=trigger,
                target=target,
                rd=np.nan,
                ci_low=np.nan,
                ci_high=np.nan,
                p_value=np.nan,
                se=np.nan,
                ess=np.nan,
                n_used=0,
                n_trigger_tested=0,
                n_A1=0,
                n_A0=0,
                baseline_mu0=np.nan,
                escalation_score_mean=np.nan,
                status="failed",
                skip_reason=str(e),
            )

    def _split_indices(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Split indices into train/test using group shuffle if group column provided."""
        n = len(df)
        test_size = self.config.split.test_size
        group_col = self.config.split.split_group_col

        if group_col and group_col in df.columns:
            groups = df[group_col].astype("string").fillna("NA").to_numpy()
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=self.config.split.random_state)
            train_idx, test_idx = next(gss.split(np.arange(n), groups=groups))
            return train_idx, test_idx
        else:
            rng = np.random.RandomState(self.config.split.random_state)
            idx = np.arange(n)
            rng.shuffle(idx)
            cut = int(n * (1 - test_size))
            return idx[:cut], idx[cut:]

    def _estimate_pair(
        self,
        df: pd.DataFrame,
        flags: pd.DataFrame,
        trigger: str,
        target: str,
        esc_scores: Dict[str, np.ndarray],
    ) -> PairResult:
        """Estimate for a single trigger–target pair on test data."""
        # Extract relevant columns
        T_col = f"{trigger}_T"
        R_col = f"{trigger}_R"
        if T_col not in flags.columns or R_col not in flags.columns:
            return PairResult(
                trigger=trigger,
                target=target,
                rd=np.nan,
                ci_low=np.nan,
                ci_high=np.nan,
                p_value=np.nan,
                se=np.nan,
                ess=np.nan,
                n_used=0,
                n_trigger_tested=0,
                n_A1=0,
                n_A0=0,
                baseline_mu0=np.nan,
                escalation_score_mean=np.nan,
                status="failed",
                skip_reason=f"Trigger {trigger} flags missing.",
            )

        # Tested mask for trigger
        T_trigger = flags[T_col].astype(int).to_numpy()
        tested_mask = (T_trigger == 1)
        n_tested = int(tested_mask.sum())
        if n_tested < self.config.tmle.min_tested:
            return PairResult(
                trigger=trigger,
                target=target,
                rd=np.nan,
                ci_low=np.nan,
                ci_high=np.nan,
                p_value=np.nan,
                se=np.nan,
                ess=np.nan,
                n_used=0,
                n_trigger_tested=n_tested,
                n_A1=0,
                n_A0=0,
                baseline_mu0=np.nan,
                escalation_score_mean=np.nan,
                status="failed",
                skip_reason=f"Too few tested isolates for {trigger}: n={n_tested} < {self.config.tmle.min_tested}",
            )

        # Outcome: escalation score for target
        if target not in esc_scores:
            return PairResult(
                trigger=trigger,
                target=target,
                rd=np.nan,
                ci_low=np.nan,
                ci_high=np.nan,
                p_value=np.nan,
                se=np.nan,
                ess=np.nan,
                n_used=0,
                n_trigger_tested=n_tested,
                n_A1=0,
                n_A0=0,
                baseline_mu0=np.nan,
                escalation_score_mean=np.nan,
                status="failed",
                skip_reason=f"Escalation score for target {target} not computed.",
            )
        Y = esc_scores[target]

        # Build covariates
        X = self._encode_covariates(df, self.config.covariates.covariate_cols)

        # Subset to tested
        X_tested = X[tested_mask]
        A = flags[R_col].astype(int).to_numpy()[tested_mask]
        Y_tested = Y[tested_mask]

        # Check group sizes
        n1 = (A == 1).sum()
        n0 = (A == 0).sum()
        if n1 < self.config.tmle.min_group or n0 < self.config.tmle.min_group:
            return PairResult(
                trigger=trigger,
                target=target,
                rd=np.nan,
                ci_low=np.nan,
                ci_high=np.nan,
                p_value=np.nan,
                se=np.nan,
                ess=np.nan,
                n_used=0,
                n_trigger_tested=n_tested,
                n_A1=int(n1),
                n_A0=int(n0),
                baseline_mu0=np.nan,
                escalation_score_mean=np.nan,
                status="failed",
                skip_reason=f"Group sizes too small: A=1 {n1}, A=0 {n0} (min required {self.config.tmle.min_group})",
            )

        # 1. Testing model P(T_trigger=1|X) on full test set → weights for tested
        if self.config.nuisance.use_joint_selection:
            # Use joint selection model (bivariate probit) for testing probabilities
            joint_model = JointSelectionModel(min_prob=self.config.tmle.min_prob)
            joint_model.fit(X, T_trigger, flags[R_col].to_numpy())
            p_test = joint_model.predict_p_test(X)
            weights_full, test_diag = joint_model.compute_weights(p_test, tested_mask)
            model_spec_testing = "joint_selection"
        else:
            # Use standard TestingModel
            test_model = TestingModel(
                model_type=self.config.nuisance.testing_model,
                calibrate=self.config.nuisance.calibrate_testing,
                calibration_method="isotonic",
                cv_folds=3,
                random_state=self.config.nuisance.random_state,
                min_prob=self.config.tmle.min_prob,
                weight_cap_percentile=self.config.tmle.weight_cap_percentile,
                n_folds_cv=self.config.nuisance.testing_cv_folds,
            )
            test_model.fit(X, T_trigger)
            if test_model._is_cross_fitted:
                p_test = test_model.get_oof_predictions()
            else:
                p_test = test_model.predict_proba(X)
            weights_full, test_diag = test_model.compute_weights(p_test, tested_mask)
            model_spec_testing = self.config.nuisance.testing_model

        weights = weights_full[tested_mask]

        # 2. Nuisance model cross‑fit (propensity, outcome) among tested
        nuis_fitter = NuisanceModelFitter(
            propensity_model_type=self.config.nuisance.propensity_model,
            outcome_model_type=self.config.nuisance.outcome_model,
            calibrate_propensity=self.config.nuisance.calibrate_propensity,
            calibrate_outcome=self.config.nuisance.calibrate_outcome,
            n_folds=self.config.tmle.n_folds,
            random_state=self.config.nuisance.random_state,
            min_prob=self.config.tmle.min_prob,
        )
        g1_hat, Q1_hat, Q0_hat = nuis_fitter.cross_fit(X_tested, A, Y_tested, weights)

        # 3. TMLE
        tmle_estimator = TMLE_RD(
            bounds=(0.0, None),
            targeted_regularization=False,
        )
        result = tmle_estimator.compute(
            A=A,
            Y=Y_tested,
            weights=weights,
            Q1=Q1_hat,
            Q0=Q0_hat,
            g1=g1_hat,
            return_all=False,
        )

        # 4. Build PairResult
        baseline_mu0 = np.average(Q0_hat, weights=weights)
        esc_mean = np.average(Y_tested, weights=weights)

        diag = {
            "testing_model": test_diag.__dict__ if hasattr(test_diag, '__dict__') else test_diag,
            "n_tested_full": int(tested_mask.sum()),
            "n_A1": int(n1),
            "n_A0": int(n0),
            "weights_mean": float(np.mean(weights)),
            "weights_p99": float(np.percentile(weights, 99)) if len(weights) > 0 else np.nan,
        }

        pair_res = PairResult(
            trigger=trigger,
            target=target,
            rd=result.psi,
            ci_low=result.ci_low,
            ci_high=result.ci_high,
            p_value=result.p_value,
            se=result.se,
            ess=effective_sample_size(weights),
            n_used=len(weights),
            n_trigger_tested=n_tested,
            n_A1=int(n1),
            n_A0=int(n0),
            baseline_mu0=baseline_mu0,
            escalation_score_mean=esc_mean,
            diagnostics=diag,
            model_spec={
                "propensity_model": self.config.nuisance.propensity_model,
                "outcome_model": self.config.nuisance.outcome_model,
                "testing_model": model_spec_testing,
                "use_joint_selection": self.config.nuisance.use_joint_selection,
                "n_folds": self.config.tmle.n_folds,
            },
            status="ok",
        )
        return pair_res

    def _encode_covariates(self, df: pd.DataFrame, covariate_cols: List[str]) -> np.ndarray:
        """
        Quick one‑hot encoding for covariates. 
        In production, use CovariateBuilder from features module.
        """
        encoded = pd.get_dummies(df[covariate_cols].astype("string").fillna("NA"), drop_first=True)
        return encoded.to_numpy(dtype="float32")