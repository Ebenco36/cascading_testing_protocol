"""
Subgroup Summary for Heterogeneous Effects
===========================================

Aggregates CATE estimates by clinically meaningful subgroups (e.g., ward type,
age group, pathogen) and produces summary tables with confidence intervals.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats


def subgroup_ate(
    cate: np.ndarray,
    subgroups: np.ndarray,
    group_names: Optional[List[str]] = None,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """
    Compute average treatment effect within each subgroup.

    Args:
        cate: Conditional average treatment effect estimates for each observation.
        subgroups: Array of subgroup labels (same length as cate).
        group_names: Optional list of subgroup names to include.
        ci_level: Confidence level for confidence intervals.

    Returns:
        DataFrame with columns: subgroup, ate, ci_low, ci_high, n.
    """
    df = pd.DataFrame({"subgroup": subgroups, "cate": cate})
    if group_names is not None:
        df = df[df["subgroup"].isin(group_names)]

    def _agg(x):
        n = len(x)
        mean = np.mean(x)
        se = np.std(x, ddof=1) / np.sqrt(n) if n > 1 else np.nan
        ci = stats.norm.interval(ci_level, loc=mean, scale=se) if not np.isnan(se) else (np.nan, np.nan)
        return pd.Series({
            "ate": mean,
            "ci_low": ci[0],
            "ci_high": ci[1],
            "n": n,
        })

    result = df.groupby("subgroup")["cate"].apply(_agg).reset_index()
    return result


def subgroup_difference(
    cate: np.ndarray,
    subgroups: np.ndarray,
    reference_group: str,
    group_names: Optional[List[str]] = None,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """
    Compute difference in ATE between each subgroup and a reference group.

    Args:
        cate: CATE estimates.
        subgroups: Subgroup labels.
        reference_group: Name of the reference subgroup.
        group_names: Optional list of subgroups to include.
        ci_level: Confidence level.

    Returns:
        DataFrame with columns: subgroup, diff, ci_low, ci_high, p_value.
    """
    df = pd.DataFrame({"subgroup": subgroups, "cate": cate})
    if group_names is not None:
        df = df[df["subgroup"].isin(group_names)]

    # Separate reference
    ref_mask = (df["subgroup"] == reference_group)
    ref_cate = df.loc[ref_mask, "cate"].values
    other_groups = df.loc[~ref_mask, "subgroup"].unique()

    rows = []
    for grp in other_groups:
        grp_cate = df.loc[df["subgroup"] == grp, "cate"].values
        diff = np.mean(grp_cate) - np.mean(ref_cate)
        # Standard error of difference
        n1 = len(grp_cate)
        n2 = len(ref_cate)
        var1 = np.var(grp_cate, ddof=1) if n1 > 1 else 0
        var2 = np.var(ref_cate, ddof=1) if n2 > 1 else 0
        se_diff = np.sqrt(var1/n1 + var2/n2) if (n1>1 and n2>1) else np.nan
        ci = stats.norm.interval(ci_level, loc=diff, scale=se_diff) if not np.isnan(se_diff) else (np.nan, np.nan)
        # p-value for test of difference
        if not np.isnan(se_diff) and se_diff > 0:
            z = diff / se_diff
            p = 2 * (1 - stats.norm.cdf(abs(z)))
        else:
            p = np.nan
        rows.append({
            "subgroup": grp,
            "diff": diff,
            "ci_low": ci[0],
            "ci_high": ci[1],
            "p_value": p,
        })
    return pd.DataFrame(rows)