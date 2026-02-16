"""
Phase 1 Screening for Trigger–Target Pairs
===========================================

Quick screening using crude odds ratios (with Haldane correction) and FDR correction
to select promising pairs for the full causal analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def haldane_odds_ratio(
    a: int, b: int, c: int, d: int, add: float = 0.5
) -> Tuple[float, float, float, float]:
    """
    2x2 odds ratio with Haldane correction, plus Wald CI on log(OR), and Fisher exact p.

    Table:
              D=1   D=0
        A=1    a     b
        A=0    c     d

    Returns:
        OR, ci_low, ci_high, p_value
    """
    a_adj = a + add
    b_adj = b + add
    c_adj = c + add
    d_adj = d + add

    OR = (a_adj * d_adj) / (b_adj * c_adj)
    se = np.sqrt(1 / a_adj + 1 / b_adj + 1 / c_adj + 1 / d_adj)
    log_or = np.log(OR)
    ci_low = float(np.exp(log_or - 1.96 * se))
    ci_high = float(np.exp(log_or + 1.96 * se))

    table = np.array([[a, b], [c, d]], dtype=int)
    _, p = stats.fisher_exact(table)

    return float(OR), ci_low, ci_high, float(p)


@dataclass(frozen=True)
class Phase1Config:
    """Configuration for Phase 1 screening."""
    min_group: int = 50
    min_trigger_tested: int = 100
    crude_screening_threshold: float = 0.05  # p‑value threshold BEFORE FDR
    fdr_alpha: float = 0.05
    exclude_targets_equal_trigger: bool = True


class Phase1Screener:
    """
    Phase 1 screening for (trigger A, target D) pairs.

    Uses 2x2 tables among tested isolates:
        - Group 1: A_R = 1 (resistant)
        - Group 2: A_S = 1 (susceptible, i.e., tested and not resistant)
        Outcome: D_T = 1 (target tested)
    """

    def __init__(self, cfg: Phase1Config):
        self.cfg = cfg

    def run(
        self,
        *,
        df: pd.DataFrame,
        flags: pd.DataFrame,
        all_codes: List[str],
        top_n: Optional[int] = None,
        candidate_triggers: Optional[List[str]] = None,
        candidate_targets: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if not df.index.equals(flags.index):
            raise ValueError("df and flags must have identical index.")

        # Determine which codes are actually present
        codes = [c for c in all_codes if f"{c}_T" in flags.columns and f"{c}_R" in flags.columns]

        if candidate_triggers is not None:
            triggers = [c for c in candidate_triggers if c in codes]
        else:
            triggers = codes

        if candidate_targets is not None:
            targets = [c for c in candidate_targets if c in codes]
        else:
            targets = codes

        rows = []

        for A in triggers:
            # Trigger flags
            A_T = flags[f"{A}_T"].astype(int).to_numpy()
            A_R = flags[f"{A}_R"].astype(int).to_numpy()
            tested_mask = (A_T == 1)

            n_tested = int(tested_mask.sum())
            if n_tested < self.cfg.min_trigger_tested:
                continue

            # Among tested, define susceptible: tested and not resistant
            A_S = tested_mask & (A_R == 0)

            nR = int(A_R[tested_mask].sum())
            nS = int(A_S.sum())
            if nR < self.cfg.min_group or nS < self.cfg.min_group:
                continue

            for D in targets:
                if self.cfg.exclude_targets_equal_trigger and D == A:
                    continue

                D_T = flags[f"{D}_T"].astype(int).to_numpy()

                # Restrict to rows where A was tested
                D_T_sub = D_T[tested_mask]
                A_R_sub = A_R[tested_mask]
                A_S_sub = A_S[tested_mask]

                # 2x2 counts
                a = int(((A_R_sub == 1) & (D_T_sub == 1)).sum())
                b = int(((A_R_sub == 1) & (D_T_sub == 0)).sum())
                c = int(((A_S_sub == 1) & (D_T_sub == 1)).sum())
                d = int(((A_S_sub == 1) & (D_T_sub == 0)).sum())

                # Skip if group sizes insufficient (should already be caught, but double‑check)
                if (a + b) < self.cfg.min_group or (c + d) < self.cfg.min_group:
                    continue

                OR, lo, hi, p = haldane_odds_ratio(a, b, c, d)

                # Optional crude p‑value filter
                if p > self.cfg.crude_screening_threshold:
                    continue

                # Effect size: delta risk among testedA
                pR = a / (a + b) if (a + b) else np.nan
                pS = c / (c + d) if (c + d) else np.nan
                delta = float(pR - pS) if (pR == pR and pS == pS) else np.nan

                rows.append({
                    "trigger": A,
                    "target": D,
                    "or_unadjusted": OR,
                    "or_ci_low": lo,
                    "or_ci_high": hi,
                    "p_value": p,
                    "delta": delta,
                    "p_D_tested_given_A_R": pR,
                    "p_D_tested_given_A_S": pS,
                    "n_trigger_tested": n_tested,
                    "n_A_R": int(a + b),
                    "n_A_S": int(c + d),
                })

        out = pd.DataFrame(rows)
        if out.empty:
            return out

        # FDR correction
        rej, qvals, _, _ = multipletests(
            out["p_value"].to_numpy(),
            alpha=self.cfg.fdr_alpha,
            method="fdr_bh",
        )
        out["q_value"] = qvals
        out["significant"] = rej

        # Sort by significance and absolute log OR
        out["abs_log_or"] = np.abs(np.log(out["or_unadjusted"].clip(lower=1e-12)))
        out = out.sort_values(["significant", "abs_log_or"], ascending=[False, False]).reset_index(drop=True)

        if top_n is not None:
            out = out.head(int(top_n)).reset_index(drop=True)
        return out