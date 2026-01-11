"""
StatisticalTesting.py

Cascade Statistical Testing Module
=================================

Adds confirmatory statistics to the ARM-derived cascade graph.

What this module does
---------------------
1) Automatic Data Standardization:
   Detects if raw dataframe columns use long names (e.g. "DOX - Doxycycline_Outcome")
   and renames them to match rule codes (e.g. "DOX_Outcome").

2) Context heterogeneity test (chi-square / Fisher exact fallback):
   For a given cascade Source_R -> Target_T, tests whether the probability of
   Target being tested among Source-resistant episodes differs across levels
   of a context variable (e.g., ward type).

3) Multiple-testing correction:
   Bonferroni, Holm, and Benjamini–Hochberg FDR via statsmodels.

Design goals
------------
- Deterministic and reproducible (no randomness used here).
- Defensive: validates columns, handles empty/degenerate cases gracefully.
- Clean outputs: CSV-ready tables (no embedded model objects).

Dependencies
------------
pip install pandas numpy scipy statsmodels
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helpers (pure, deterministic)
# -----------------------------------------------------------------------------

def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_bool_to_int(s: pd.Series) -> pd.Series:
    """Convert {bool, 0/1, numeric strings} -> {0,1} int, missing -> 0."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def _is_resistant(value: object) -> int:
    """
    Conservative resistance parsing:
      - 1 if outcome is exactly 'R' or startswith 'R' (e.g., 'R*')
      - or contains 'R' in combined codes (e.g., 'IR') excluding 'NS'
    """
    if value is None:
        return 0
    s = str(value).strip().upper()
    if s in ("", "NA", "N/A", "NONE", "NAN"):
        return 0
    if s == "R" or s.startswith("R"):
        return 1
    if "R" in s and s not in ("NS",):
        return 1
    return 0


def _cramers_v(chi2: float, n: int, r: int, k: int) -> float:
    if n <= 0:
        return float("nan")
    denom = n * (min(r - 1, k - 1))
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(max(chi2, 0.0) / denom))


@dataclass(frozen=True)
class CascadeSpec:
    source: str
    target: str

    @property
    def name(self) -> str:
        return f"{self.source}→{self.target}"


# -----------------------------------------------------------------------------
# Context heterogeneity testing (chi-square / Fisher)
# -----------------------------------------------------------------------------

class CascadeStatisticalTesting:
    """
    Perform context heterogeneity tests for discovered cascades.
    Automatically standardizes column names if they don't match rule codes.

    Inputs
    ------
    raw_df:
      Original raw data with columns like:
        - {drug}_Outcome : outcome code (R/S/I/...)
        - {drug}_Tested  : 0/1 or bool tested indicator
        - Context variables (e.g., ARS_WardType, CareType, Year, ...)

    cascade_rules:
      Output from ARMEngine.discover_rules(), must contain Source, Target.
    """

    DEFAULT_CONTEXT_COLS = ("ARS_WardType", "CareType", "AgeGroup", "Sex", "Year")

    def __init__(
        self,
        raw_df: pd.DataFrame,
        cascade_rules: pd.DataFrame,
        *,
        context_cols: Optional[Iterable[str]] = None,
        verbose: bool = True,
    ) -> None:
        if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
            raise ValueError("raw_df must be a non-empty pandas DataFrame")
        if not isinstance(cascade_rules, pd.DataFrame) or cascade_rules.empty:
            raise ValueError("cascade_rules must be a non-empty pandas DataFrame")

        for c in ("Source", "Target"):
            if c not in cascade_rules.columns:
                raise ValueError(f"cascade_rules missing required column: {c}")

        # 1. Standardize DataFrame Columns
        self.raw_df = self._standardize_column_names(raw_df, verbose=verbose)
        
        self.cascade_rules = cascade_rules.copy()
        self.context_cols = tuple(context_cols) if context_cols is not None else self.DEFAULT_CONTEXT_COLS
        self.verbose = verbose

        self.test_results: Optional[pd.DataFrame] = None
        self.all_test_results: Dict[str, pd.DataFrame] = {}

    def _standardize_column_names(self, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        Detects if columns are 'LongName_Outcome' and renames them to 'Code_Outcome'.
        Matches logic from DataLoader._code_from_base: split by " - " and take first part.
        """
        out = df.copy()
        rename_map = {}
        
        # Regex to capture "Code - Name_Suffix" pattern
        # Suffixes: _Outcome or _Tested
        for col in out.columns:
            if " - " in col and (col.endswith("_Outcome") or col.endswith("_Tested")):
                parts = col.split(" - ")
                code = parts[0].strip()
                
                if col.endswith("_Outcome"):
                    new_name = f"{code}_Outcome"
                else:
                    new_name = f"{code}_Tested"
                
                rename_map[col] = new_name
        
        if rename_map:
            out = out.rename(columns=rename_map)
            if verbose:
                print(f"✓ Auto-standardized {len(rename_map)} raw data columns to match rule codes.")
                # print(f"  Example: {list(rename_map.keys())[0]} -> {list(rename_map.values())[0]}")
        
        return out

    # ---- internal data build ----

    def _build_cascade_frame(self, spec: CascadeSpec) -> pd.DataFrame:
        """
        Build a minimal frame containing:
          - Source_R (binary)
          - Target_T (binary)
          - context columns (intersection with available)
        """
        source_outcome = f"{spec.source}_Outcome"
        target_tested = f"{spec.target}_Tested"

        if source_outcome not in self.raw_df.columns or target_tested not in self.raw_df.columns:
            # Fallback: check if we missed something or if data is truly missing
            return pd.DataFrame()

        df = pd.DataFrame(index=self.raw_df.index)
        
        # Parse Resistance (Source)
        df["Source_R"] = self.raw_df[source_outcome].apply(_is_resistant).astype(int)
        
        # Parse Tested (Target)
        df["Target_T"] = _safe_bool_to_int(self.raw_df[target_tested])

        for c in self.context_cols:
            if c in self.raw_df.columns:
                df[c] = self.raw_df[c]

        return df.dropna(subset=["Source_R", "Target_T"])

    # ---- single cascade test ----

    def chi_square_test_cascade_by_context(
        self,
        source_drug: str,
        target_drug: str,
        *,
        context_var: str = "ARS_WardType",
        min_stratum_size: int = 10,
        fisher_if_2x2_and_small_expected: bool = True,
    ) -> Dict[str, object]:
        """
        Test whether P(Target_T=1 | Source_R=1) differs across context levels.

        Returns a flat dict suitable for a DataFrame row.
        """
        spec = CascadeSpec(str(source_drug), str(target_drug))
        df = self._build_cascade_frame(spec)
        
        if df.empty:
            return {
                "Source": spec.source,
                "Target": spec.target,
                "Cascade": spec.name,
                "Context_Variable": context_var,
                "Error": "Missing required raw columns or no data",
            }

        if context_var not in df.columns:
            return {
                "Source": spec.source,
                "Target": spec.target,
                "Cascade": spec.name,
                "Context_Variable": context_var,
                "Error": f"Context variable not found: {context_var}",
            }

        # Restrict to resistant episodes
        df_r = df[df["Source_R"] == 1].copy()
        if df_r.empty:
            return {
                "Source": spec.source,
                "Target": spec.target,
                "Cascade": spec.name,
                "Context_Variable": context_var,
                "Error": "No resistant cases for this cascade",
            }

        # Build contingency table: rows = context levels, cols = [tested_yes, tested_no]
        rows: List[List[int]] = []
        levels: List[str] = []

        # Normalize to strings for stable grouping
        ctx_series = df_r[context_var].astype(str)
        for lvl in sorted(ctx_series.dropna().unique()):
            sub = df_r[ctx_series == lvl]
            if len(sub) < min_stratum_size:
                continue
            tested_yes = int((sub["Target_T"] == 1).sum())
            tested_no = int((sub["Target_T"] == 0).sum())
            rows.append([tested_yes, tested_no])
            levels.append(lvl)

        if len(rows) < 2:
            return {
                "Source": spec.source,
                "Target": spec.target,
                "Cascade": spec.name,
                "Context_Variable": context_var,
                "Error": f"Fewer than 2 strata with ≥{min_stratum_size} resistant cases",
            }

        table = np.asarray(rows, dtype=int)

        # Default chi-square
        # chi2, pval, dof, expected = chi2_contingency(table)
                # Perform chi-square test (with graceful handling of degenerate tables)
        try:
            chi2, pval, dof, expected = chi2_contingency(table)
        except ValueError as e:
            # This happens when expected frequencies have zeros or are otherwise invalid
            return {
                "Source": spec.source,
                "Target": spec.target,
                "Cascade": spec.name,
                "Context_Variable": context_var,
                "Error": f"Chi-square failed: {e}",
            }


        method = "chi2"
        warning = ""
        low_expected = int((expected < 5).sum())

        # Fisher fallback if 2x2 and expected low
        if fisher_if_2x2_and_small_expected and table.shape == (2, 2) and low_expected > 0:
            _odds, p_f = fisher_exact(table, alternative="two-sided")
            method = "fisher_exact"
            pval = float(p_f)
            warning = "Used Fisher exact due to small expected counts"
        elif low_expected > 0:
            warning = f"{low_expected}/{expected.size} expected cells < 5; chi-square approximation may be weak"

        n_resistant = int(table.sum())
        v = _cramers_v(float(chi2), n_resistant, table.shape[0], table.shape[1])

        return {
            "Source": spec.source,
            "Target": spec.target,
            "Cascade": spec.name,
            "Context_Variable": context_var,
            "N_Strata": int(table.shape[0]),
            "N_Resistant": int(n_resistant),
            "Test": method,
            "Chi2": float(chi2),
            "DOF": int(dof),
            "Pvalue": float(pval),
            "CramersV": float(v),
            "Context_Levels": "|".join(levels),
            "Warning": warning,
        }

    # ---- multiple testing correction ----

    @staticmethod
    def apply_multiple_comparison_correction(
        test_results: pd.DataFrame,
        *,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Adds corrected p-values + boolean significance flags with stable dtypes.
        Avoids pandas FutureWarnings by creating target columns with correct dtypes.
        """
        out = test_results.copy()
        if out.empty:
            return out

        out["Pvalue"] = pd.to_numeric(out["Pvalue"], errors="coerce")
        mask = out["Pvalue"].notna() & np.isfinite(out["Pvalue"].to_numpy())
        pvals = out.loc[mask, "Pvalue"].to_numpy(dtype=float)

        # Initialize columns with correct dtypes
        out["Pvalue_Bonferroni"] = np.nan
        out["Pvalue_Holm"] = np.nan
        out["Pvalue_FDR"] = np.nan
        out["Significant_Bonferroni"] = False
        out["Significant_Holm"] = False
        out["Significant_FDR"] = False

        if len(pvals) == 0:
            out.attrs["alpha"] = alpha
            out.attrs["alpha_bonferroni"] = float("nan")
            return out

        rej_bon, p_bon, _, alpha_bon = multipletests(pvals, alpha=alpha, method="bonferroni")
        rej_holm, p_holm, _, _ = multipletests(pvals, alpha=alpha, method="holm")
        rej_fdr, p_fdr, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")

        out.loc[mask, "Pvalue_Bonferroni"] = p_bon
        out.loc[mask, "Pvalue_Holm"] = p_holm
        out.loc[mask, "Pvalue_FDR"] = p_fdr

        out.loc[mask, "Significant_Bonferroni"] = np.asarray(rej_bon, dtype=bool)
        out.loc[mask, "Significant_Holm"] = np.asarray(rej_holm, dtype=bool)
        out.loc[mask, "Significant_FDR"] = np.asarray(rej_fdr, dtype=bool)

        # Ensure boolean dtype
        out["Significant_Bonferroni"] = out["Significant_Bonferroni"].astype(bool)
        out["Significant_Holm"] = out["Significant_Holm"].astype(bool)
        out["Significant_FDR"] = out["Significant_FDR"].astype(bool)

        out.attrs["alpha"] = alpha
        out.attrs["alpha_bonferroni"] = float(alpha_bon)
        return out

    # ---- run across cascades ----

    def test_all_cascades_by_context(
        self,
        *,
        context_var: str = "ARS_WardType",
        min_stratum_size: int = 10,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Run heterogeneity tests for all unique cascades and apply correction.
        """
        pairs = self.cascade_rules[["Source", "Target"]].drop_duplicates().reset_index(drop=True)

        if self.verbose:
            print("\n" + "=" * 70)
            print("CHI-SQUARE / FISHER TESTING: Cascade Context Heterogeneity")
            print("=" * 70)
            print(f"Context variable: {context_var}")
            print(f"Min stratum size (resistant): {min_stratum_size}")
            print(f"Unique cascades: {len(pairs)}")

        rows: List[Dict[str, object]] = []
        for i, r in pairs.iterrows():
            res = self.chi_square_test_cascade_by_context(
                str(r["Source"]),
                str(r["Target"]),
                context_var=context_var,
                min_stratum_size=min_stratum_size,
            )
            if "Error" not in res:
                rows.append(res)

            if self.verbose and (i + 1) % 100 == 0:
                print(f"  Tested {i+1}/{len(pairs)} cascades...")

        out = pd.DataFrame(rows)
        if out.empty:
            self.test_results = out
            return out

        out = self.apply_multiple_comparison_correction(out, alpha=alpha)
        out = out.sort_values(["Pvalue", "CramersV"], ascending=[True, False]).reset_index(drop=True)
        self.test_results = out

        if self.verbose:
            print(f"\n✓ Completed tests for {len(out)} cascades (non-error)")

        return out

    def test_all_contexts(
        self,
        *,
        context_vars: Optional[List[str]] = None,
        min_stratum_size: int = 10,
        alpha: float = 0.05,
    ) -> Dict[str, pd.DataFrame]:
        """
        Run tests across multiple context variables. Returns dict[context_var] -> results_df.
        """
        if context_vars is None:
            context_vars = ["ARS_WardType", "CareType", "Year"]

        results: Dict[str, pd.DataFrame] = {}
        for ctx in context_vars:
            results[ctx] = self.test_all_cascades_by_context(
                context_var=ctx,
                min_stratum_size=min_stratum_size,
                alpha=alpha,
            )

        self.all_test_results = results
        return results

    # ---- outputs ----

    def save_results(self, output_dir: str | Path) -> None:
        outdir = _ensure_dir(output_dir)

        if self.test_results is not None and not self.test_results.empty:
            (outdir / "chi_square_cascades.csv").write_text(
                self.test_results.to_csv(index=False), encoding="utf-8"
            )

        if self.all_test_results:
            for ctx, df in self.all_test_results.items():
                if df is not None and not df.empty:
                    (outdir / f"chi_square_{str(ctx).lower()}.csv").write_text(
                        df.to_csv(index=False), encoding="utf-8"
                    )

    def print_summary(self, *, top_n: int = 20) -> None:
        if self.test_results is None or self.test_results.empty:
            print("! No chi-square results available.")
            return

        alpha = self.test_results.attrs.get("alpha", 0.05)
        alpha_bon = self.test_results.attrs.get("alpha_bonferroni", np.nan)

        print("\n" + "=" * 70)
        print("CHI-SQUARE TEST SUMMARY (Top cascades)")
        print("=" * 70)
        print(f"Alpha: {alpha} | Bonferroni alpha: {alpha_bon:.2e}")
        print(f"Rows: {len(self.test_results)}\n")

        show = self.test_results.head(top_n)
        for i, r in show.iterrows():
            sig = "***" if bool(r.get("Significant_Bonferroni", False)) else ""
            print(
                f"{i+1:2d}. {r['Cascade']:>9s} | {r['Context_Variable']:<12s} | "
                f"p={r['Pvalue']:.2e} p_bon={r['Pvalue_Bonferroni']:.2e} {sig} | "
                f"V={r['CramersV']:.3f} | strata={int(r['N_Strata'])} | nR={int(r['N_Resistant'])}"
            )
        print("")


# -----------------------------------------------------------------------------
# Optional orchestrator (mirrors Run.py style)
# -----------------------------------------------------------------------------

def run_statistical_testing(
    raw_df: pd.DataFrame,
    cascade_rules: pd.DataFrame,
    *,
    output_dir: str | Path = "./publication_figures/statistical_testing",
    context_vars: Optional[List[str]] = None,
    chi_min_stratum_size: int = 10,
    alpha: float = 0.05,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Run chi-square/Fisher heterogeneity tests and save CSV outputs.

    Returns a dict:
      - "chi_by_context": Dict[str, pd.DataFrame]
      - "chi_main": pd.DataFrame  (for first context var)
    """
    outdir = _ensure_dir(output_dir)

    if context_vars is None:
        context_vars = ["ARS_WardType", "CareType", "Year"]

    # Chi-square / Fisher tests
    # NOTE: The class automatically standardizes raw_df columns in __init__
    chi = CascadeStatisticalTesting(raw_df, cascade_rules, verbose=verbose)
    
    chi_by_ctx = chi.test_all_contexts(
        context_vars=context_vars,
        min_stratum_size=chi_min_stratum_size,
        alpha=alpha,
    )
    chi.save_results(outdir)

    chi_main = chi_by_ctx.get(context_vars[0], pd.DataFrame())
    if verbose and isinstance(chi_main, pd.DataFrame) and not chi_main.empty:
        chi.test_results = chi_main
        chi.print_summary(top_n=20)

    return {
        "chi_by_context": chi_by_ctx,
        "chi_main": chi_main,
    }


if __name__ == "__main__":
    print("StatisticalTesting module ready.")
    print("Import: CascadeStatisticalTesting, run_statistical_testing")
