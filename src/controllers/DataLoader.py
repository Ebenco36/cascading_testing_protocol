# DataLoader.py
"""
Enhanced DataLoader for AMR Cascade Discovery
============================================

This module provides a robust data-loading + preprocessing layer for cascade discovery
in antimicrobial susceptibility testing (AST) surveillance data.

Core responsibilities
--------------------
1) Read common file formats (parquet/feather/csv) and remove "Unnamed" columns.
2) Identify paired antibiotic columns:
      <ABX>_Tested   (0/1)
      <ABX>_Outcome  (S/I/R or similar)
3) Provide a transaction matrix builder for association-rule mining:
      <CODE>_T  (tested item)
      <CODE>_R  (resistant item; recoded by a chosen rule)
   plus optional one-hot encoded covariate items.
4) Provide *defensible* dataset reduction knobs to make mining feasible:
   - row filtering (Pathogen, ward, year, specimen, etc.)
   - exclusion flags (screening / invalid records)
   - dynamic antibiotic selection based on:
       testing prevalence/count and resistance prevalence/count

Design principles
----------------
- Keep this layer purely about data shaping; do not mine rules here.
- Make reductions transparent and reproducible (print prevalence tables if verbose).
- Default settings aim to be conservative and clinically interpretable.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

try:
    from src.mappers.top_pathogens import ALL_PATHOGENS  # type: ignore
    from src.utils.LoadClasses import LoadClasses  # type: ignore
except Exception:  # pragma: no cover
    ALL_PATHOGENS = None  # type: ignore
    LoadClasses = None  # type: ignore

_UNNAMED_RE = re.compile(r"^Unnamed(?::\s*\d+)?$")


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if _UNNAMED_RE.match(str(c))]
    return df.drop(columns=cols) if cols else df


def read_any(path: Union[str, Path]) -> pd.DataFrame:
    """
    Auto-detect reader:
      - Parquet directory  -> pd.read_parquet(dir, engine="pyarrow")
      - .parquet           -> pd.read_parquet(file, engine="pyarrow")
      - .feather/.ft       -> pd.read_feather(file)
      - otherwise          -> pd.read_csv(file)
    """
    p = Path(path)
    if p.is_dir():
        return _drop_unnamed(pd.read_parquet(p, engine="pyarrow"))
    suf = p.suffix.lower()
    if suf == ".parquet":
        return _drop_unnamed(pd.read_parquet(p, engine="pyarrow"))
    if suf in (".feather", ".ft"):
        return _drop_unnamed(pd.read_feather(p))
    try:
        return _drop_unnamed(pd.read_csv(p, low_memory=False, dtype_backend="pyarrow"))
    except TypeError:
        return _drop_unnamed(pd.read_csv(p, low_memory=False))


class DataLoader:
    """
    Enhanced data loader with cascade-discovery preprocessing.

    Attributes
    ----------
    df : pd.DataFrame
        Raw (but cleaned) dataset restricted to pathogen(s) by regex (if provided).
    paired_bases : List[str]
        Antibiotic base names that have both _Tested and _Outcome columns.
        Example base: "CIP - Ciprofloxacin"
    filtered_df : Optional[pd.DataFrame]
        The last row-filtered DataFrame used by get_transaction_matrix().
        Useful for downstream reporting/statistical testing.
    """

    TESTED_SUFFIX = "_Tested"
    OUTCOME_SUFFIX = "_Outcome"

    def __init__(self, filepath: str, pathogen_groups_regex: Sequence[str] | str = ()):
        self.filepath = filepath
        self.load = LoadClasses() if LoadClasses is not None else None

        df = read_any(filepath)

        # Harmonize a common ward label issue (if present)
        if "ARS_WardType" in df.columns:
            df = df.assign(
                ARS_WardType=df["ARS_WardType"].replace(
                    {
                        "Early Rehabilitation": "Rehabilitation",
                        "Rehabilitation": "Rehabilitation",
                    }
                )
            )

        if "Pathogen" not in df.columns:
            raise ValueError("Expected 'Pathogen' column not found in the input data.")

        # Restrict to requested pathogen(s) if provided
        if pathogen_groups_regex:
            if isinstance(pathogen_groups_regex, (list, tuple, set)):
                pattern = "|".join(f"(?:{p})" for p in pathogen_groups_regex if p)
            else:
                pattern = str(pathogen_groups_regex)
        else:
            pattern = ALL_PATHOGENS  # may be None

        if pattern:
            df = df[df["Pathogen"].astype("string").str.contains(pattern, case=False, na=False, regex=True)]

        # Drop known artifact columns if present
        artifact_cols = [c for c in df.columns if c.endswith("_Tested_Outcome")]
        if artifact_cols:
            df = df.drop(columns=artifact_cols, errors="ignore")

        self.df = df.reset_index(drop=True)
        self.all_cols = self.df.columns.to_list()

        # Identify antibiotic tested/outcome columns
        self.abx_tested_cols = sorted([c for c in self.all_cols if c.endswith(self.TESTED_SUFFIX)])
        self.abx_outcome_cols = sorted([c for c in self.all_cols if c.endswith(self.OUTCOME_SUFFIX) and " - " in c])

        self.tested_bases = [c[: -len(self.TESTED_SUFFIX)] for c in self.abx_tested_cols]
        self.outcome_bases = [c[: -len(self.OUTCOME_SUFFIX)] for c in self.abx_outcome_cols]

        self.paired_bases = sorted(set(self.tested_bases).intersection(self.outcome_bases))

        self.meta_cols = [c for c in self.all_cols if c not in (self.abx_tested_cols + self.abx_outcome_cols)]

        self._sanity_check_outcomes(raise_on_violation=False)

        # cache
        self.filtered_df: Optional[pd.DataFrame] = None
        self._transaction_matrix: Optional[pd.DataFrame] = None

    # ------------------------------ helpers ------------------------------ #
    # ---------- internals ----------
    def _present(self, cols: List[str]) -> List[str]:
        """Return only the columns that exist in dataframe."""
        return [c for c in cols if c in self.df.columns]
    
    def _tested_col(self, base: str) -> str:
        return f"{base}{self.TESTED_SUFFIX}"

    def _outcome_col(self, base: str) -> str:
        return f"{base}{self.OUTCOME_SUFFIX}"

    
    def _bases_to_cols(self, bases: List[str], return_which: str = "tested") -> List[str]:
        """
        Map a list of antibiotic base names to specific columns.
          return_which: "tested" | "outcome" | "both"
        """
        if return_which == "tested":
            return self._present([self._tested_col(b) for b in bases])
        elif return_which == "outcome":
            return self._present([self._outcome_col(b) for b in bases])
        elif return_which == "both":
            cols = []
            for b in bases:
                t = self._tested_col(b)
                y = self._outcome_col(b)
                if t in self.df.columns: cols.append(t)
                if y in self.df.columns: cols.append(y)
            return cols
        else:
            raise ValueError("return_which must be 'tested', 'outcome', or 'both'")
        
    @staticmethod
    def _code_from_base(base: str) -> str:
        """
        Extract antibiotic short code from base, e.g.:
          "CIP - Ciprofloxacin" -> "CIP"
        """
        return str(base).split(" - ")[0].strip()

    def _sanity_check_outcomes(self, raise_on_violation: bool = False) -> None:
        """
        Ensure logical consistency:
          if <ABX>_Tested == 0 then <ABX>_Outcome must be missing.
        When violated, we set Outcome to NA (or raise).
        """
        for base in self.paired_bases:
            tcol = self._tested_col(base)
            ycol = self._outcome_col(base)
            if tcol not in self.df.columns or ycol not in self.df.columns:
                continue
            t = self.df[tcol].astype("Int8").fillna(0)
            y = self.df[ycol]
            bad_mask = (t == 0) & y.notna()
            if bad_mask.any():
                if raise_on_violation:
                    n_bad = int(bad_mask.sum())
                    raise AssertionError(f"[{base}] Found {n_bad} rows where {tcol}==0 but {ycol} is not NA.")
                self.df.loc[bad_mask, ycol] = pd.NA

    # ------------------------- optional class lookups ------------------------- #

    def load_abx_classes(self):
        if self.load is None:
            raise RuntimeError("LoadClasses is not available in this environment.")
        return self.load.antibiotic_class_list

    
    def _get_bases_by_categories(self, categories: List[str], include_not_set: bool = False) -> List[str]:
        """
        Use your LoadClasses to fetch *base* antibiotic names by AWaRe categories.
        We then intersect with what exists in the dataframe (by either tested or outcome).
        """
        bases_from_categories = self.load.get_antibiotics_by_category(categories)
        if include_not_set:
            bases_from_categories += self.load.get_antibiotics_by_category(["Not Set"])

        # Keep only bases that actually exist in either tested or outcome
        present_bases = set(self.tested_bases) | set(self.outcome_bases)
        bases = [b for b in bases_from_categories if b in present_bases]
        return sorted(bases)

    def _get_bases_by_class(self, classes: List[str]) -> List[str]:
        bases_from_classes = self.load.get_antibiotics_by_class(classes)
        present_bases = set(self.tested_bases) | set(self.outcome_bases)
        bases = [b for b in bases_from_classes if b in present_bases]
        return sorted(bases)

    def get_abx_by_category(self, categories: List[str], *, return_which: str = "tested",
                            use_not_set: bool = False) -> List[str]:
        """
        Fetch *columns* (not bases) for antibiotics in certain AWaRe categories.
        return_which ∈ {"tested","outcome","both"} controls which columns you get.
        """
        bases = self._get_bases_by_categories(categories, include_not_set=use_not_set)
        return self._bases_to_cols(bases, return_which=return_which)

    def get_abx_by_class(self, classes: List[str], *, return_which: str = "tested") -> List[str]:
        """
        Fetch *columns* (not bases) for antibiotics in certain drug classes.
        return_which ∈ {"tested","outcome","both"}.
        """
        bases = self._get_bases_by_class(classes)
        return self._bases_to_cols(bases, return_which=return_which)

    def get_combined(self, *, return_which: str = "tested", use_not_set: bool = False) -> pd.DataFrame:
        """
        Return dataframe with metadata + selected antibiotic columns.

        Parameters
        ----------
        return_which : {"tested","outcome","both"}, default="tested"
            Which side to include.
        use_not_set : bool, default=False
            If True, include antibiotics categorized as "Not Set".
        """
        access  = self.get_abx_by_category(["Access"],  return_which=return_which, use_not_set=False)
        watch   = self.get_abx_by_category(["Watch"],   return_which=return_which, use_not_set=False)
        reserve = self.get_abx_by_category(["Reserve"], return_which=return_which, use_not_set=False)

        abx_selected = access + watch + reserve
        if use_not_set:
            abx_selected += self.get_abx_by_category(["Not Set"], return_which=return_which, use_not_set=True)

        return self.df[self._present(self.meta_cols + abx_selected)]

    # ------------------------- antibiotic prevalence table ------------------------- #

    def antibiotic_prevalence_table(
        self,
        df: pd.DataFrame,
        *,
        recode_mode: str = "R_vs_nonR",
    ) -> pd.DataFrame:
        """
        Compute per-antibiotic prevalence metrics for dynamic selection.

        Returns a DataFrame with:
          code, base, tested_rate, tested_count, res_rate_tested, res_count_tested
        """
        rows = []
        n = len(df)
        if n == 0:
            raise ValueError("Cannot compute prevalence table on empty dataframe.")

        for base in self.paired_bases:
            tcol = self._tested_col(base)
            ycol = self._outcome_col(base)
            if tcol not in df.columns or ycol not in df.columns:
                continue

            tested = df[tcol].astype("Int8").fillna(0).astype(int) == 1
            tested_count = int(tested.sum())
            tested_rate = tested_count / n if n else 0.0

            y = df.loc[tested, ycol].astype("string")

            if recode_mode == "R_vs_nonR":
                rmask = y.str.contains("R", na=False)
            elif recode_mode == "NS_vs_S":
                # non-susceptible vs susceptible
                rmask = ~y.str.contains("S", na=False)
            else:
                raise ValueError("recode_mode must be 'R_vs_nonR' or 'NS_vs_S'")

            res_count = int(rmask.sum())
            res_rate = (res_count / tested_count) if tested_count else 0.0

            rows.append(
                {
                    "code": self._code_from_base(base),
                    "base": base,
                    "tested_rate": tested_rate,
                    "tested_count": tested_count,
                    "res_rate_tested": res_rate,
                    "res_count_tested": res_count,
                }
            )

        out = pd.DataFrame(rows).sort_values(["tested_rate", "res_rate_tested"], ascending=False).reset_index(drop=True)
        return out

    # ------------------------- transaction matrix builder ------------------------- #

    def get_transaction_matrix(
        self,
        *,
        filters: Optional[
            Dict[str, Union[str, int, float, bool, List[Union[str, int, float, bool]]]]
        ] = None,
        recode_mode: str = "R_vs_nonR",
        verbose: bool = True,
        covariate_cols: Optional[List[str]] = None,
        include_covariates: bool = True,
        max_levels: int = 25,
        min_count: int = 50,
        apply_exclusions: bool = True,
        # Dynamic antibiotic selection knobs
        min_test_rate: float = 0.0,
        min_test_count: int = 0,
        min_res_rate: float = 0.0,
        min_res_count: int = 0,
        max_antibiotics: Optional[int] = None,
        always_keep: Optional[List[str]] = None,
        # NEW: row filter – drop isolates with no resistance in any selected antibiotic
        drop_all_susceptible_rows: bool = True,
    ) -> pd.DataFrame:
        """
        Build a binary transaction matrix with:
            - <CODE>_T items (tested)
            - <CODE>_R items (resistant per recode_mode)
            - optional covariate one-hot items (e.g., ward type)

        Filtering is applied in two stages:
        (1) Row filters + exclusion flags
        (2) Dynamic antibiotic selection based on tested/resistance prevalence

        Additional row filter (optional):
        - If drop_all_susceptible_rows=True, drop rows where all <CODE>_R items are False.
        """
        df = self.df.copy()

        # (0) Apply common exclusion flags if present (defensible row reduction)
        if apply_exclusions:
            for col in [
                "IsSpecificlyExcluded_Screening",
                "IsSpecificlyExcluded_Pathogen",
                "IsSpecificlyExcluded_PathogenevidenceNegative",
            ]:
                if col in df.columns:
                    # Column may be boolean or string; treat "True"/True as excluded
                    s = df[col]
                    bad = s.astype("string").str.lower().isin(["true", "1", "yes"])
                    df = df.loc[~bad].copy()

        # (1) Apply explicit row filters
        if filters:
            missing_cols = [col for col in filters.keys() if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Filter columns not found: {missing_cols}")

            mask = pd.Series(True, index=df.index)
            for col, values in filters.items():
                if isinstance(values, (str, int, float, bool)):
                    values = [values]
                values_str = [str(v) for v in values]
                mask &= df[col].astype("string").isin(values_str)
            df = df.loc[mask].reset_index(drop=True)

        if df.empty:
            raise ValueError(
                f"No rows left after filtering. filters={filters}, apply_exclusions={apply_exclusions}"
            )

        if not self.paired_bases:
            raise ValueError("No paired tested/outcome antibiotics detected in this dataset.")

        self.filtered_df = df

        # (2) Dynamic antibiotic selection
        prev = self.antibiotic_prevalence_table(df, recode_mode=recode_mode)

        keep_mask = (
            (prev["tested_rate"] >= float(min_test_rate))
            & (prev["tested_count"] >= int(min_test_count))
            & (prev["res_rate_tested"] >= float(min_res_rate))
            & (prev["res_count_tested"] >= int(min_res_count))
        )
        keep_codes = set(prev.loc[keep_mask, "code"].tolist())

        if always_keep:
            keep_codes |= set([str(c).strip() for c in always_keep])

        if max_antibiotics is not None:
            # rank among currently kept
            ranked = prev[prev["code"].isin(keep_codes)].copy()
            ranked = ranked.sort_values(["tested_rate", "res_rate_tested"], ascending=False)
            keep_codes = set(ranked.head(int(max_antibiotics))["code"].tolist())

        selected_bases = [
            b for b in self.paired_bases if self._code_from_base(b) in keep_codes
        ]
        if not selected_bases:
            raise ValueError(
                "Antibiotic selection removed all antibiotics. "
                "Relax thresholds (min_test_rate/min_res_rate/min_counts) or set always_keep."
            )

        # Default covariates: keep minimal (avoid combinatorial explosion)
        if covariate_cols is None:
            covariate_cols = ["ARS_WardType", "CareType", "AgeGroup", "Year"]

        # T items
        T = pd.DataFrame(index=df.index)
        for base in selected_bases:
            code = self._code_from_base(base)
            tcol = self._tested_col(base)
            T[f"{code}_T"] = (df[tcol].astype("Int8").fillna(0).astype(int) == 1)

        # R items
        R = pd.DataFrame(index=df.index)
        for base in selected_bases:
            code = self._code_from_base(base)
            ycol = self._outcome_col(base)
            y = df[ycol].astype("string")

            if recode_mode == "R_vs_nonR":
                rmask = y.str.contains("R", na=False)
            elif recode_mode == "NS_vs_S":
                rmask = ~y.str.contains("S", na=False)
            else:
                raise ValueError("recode_mode must be 'R_vs_nonR' or 'NS_vs_S'")

            R[f"{code}_R"] = rmask.fillna(False)

        # Covariate items (one-hot; filtered by min_count and max_levels)
        C = pd.DataFrame(index=df.index)
        if include_covariates and covariate_cols:
            missing = [c for c in covariate_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Covariate columns not found: {missing}")

            for c in covariate_cols:
                s = df[c].astype("string").fillna("NA")
                vc = s.value_counts()
                keep_levels = vc[vc >= min_count].index.tolist()[:max_levels]
                for level in keep_levels:
                    C[f"{c}={level}"] = (s == level)

        matrix = pd.concat([R, T, C], axis=1).astype(bool)

        # Drop empty columns (all-false)
        nonzero = matrix.columns[(matrix.sum(axis=0) > 0).to_numpy()].tolist()
        matrix = matrix[nonzero]

        # Optionally drop rows where all resistance items (_R) are False
        r_items = [c for c in matrix.columns if c.endswith("_R")]
        n_rows_before_r_filter = matrix.shape[0]
        n_rows_dropped_all_susceptible = 0
        if drop_all_susceptible_rows and r_items:
            r_block = matrix[r_items]
            keep_rows = r_block.any(axis=1)  # at least one _R == True
            n_rows_dropped_all_susceptible = int((~keep_rows).sum())
            matrix = matrix.loc[keep_rows].reset_index(drop=True)

        if verbose:
            r_items = [c for c in matrix.columns if c.endswith("_R")]
            t_items = [c for c in matrix.columns if c.endswith("_T")]
            cov_items = [
                c for c in matrix.columns if (c not in r_items and c not in t_items)
            ]

            abx_codes = sorted({c[:-2] for c in (r_items + t_items)})

            print(f"✓ Rows after filters (before R-only row drop): {n_rows_before_r_filter:,}")
            if drop_all_susceptible_rows and r_items:
                print(
                    f"✓ Dropped rows with all _R == False: {n_rows_dropped_all_susceptible:,} "
                    f"({n_rows_dropped_all_susceptible / n_rows_before_r_filter * 100:.1f}%)"
                )
            print(f"✓ Final rows in transaction matrix: {matrix.shape[0]:,}")
            print(f"✓ Selected antibiotics: {len(selected_bases)} (from {len(self.paired_bases)})")
            print(f"  • Codes: {', '.join(abx_codes)}")
            print(f"✓ Transaction matrix: {matrix.shape[0]} rows × {matrix.shape[1]} items")
            print(f"  • R items: {len(r_items)} | T items: {len(t_items)} | Covariates: {len(cov_items)}")

            if r_items:
                print("  • R items:", ", ".join(sorted(r_items)))
            if t_items:
                print("  • T items:", ", ".join(sorted(t_items)))
            if cov_items:
                print("  • Covariate items:", ", ".join(sorted(cov_items)))

        self._transaction_matrix = matrix
        return matrix

