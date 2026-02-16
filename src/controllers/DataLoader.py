# DataLoader.py
"""
DataLoader v4 for AMR Cascade Discovery
======================================

Robust data loading + preprocessing layer for cascade discovery in AST surveillance data.

What this version guarantees (if strict=True):
- Tested columns are clean 0/1
- Outcome columns are normalized to {S, I, R, NA}
- Consistency: Tested==0 -> Outcome is NA, and Outcome in {S,I,R} -> Tested==1
- Paired antibiotic columns are detected and audited
- Cohorts are reproducible (fingerprint + filter report + diagnostics)
- Transaction matrices are schema-checked and caching is safe (no stale reuse)

No mining / ML training happens here.

v4 changes (high impact):
- FIX: get_abx_flags() no longer calls missing _as_int01 (now implemented)
- Outcome col detection no longer requires " - " in name (more robust exports)
- get_transaction_matrix() enforces normalization + consistency even if normalize_on_load=False
- Cache signature uses stable serialization for filter_config / filter_pipeline when available
- Small performance improvements: reuse converted tested columns for selected antibiotics
"""

from __future__ import annotations

import re
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    from src.mappers.top_pathogens import ALL_PATHOGENS  # type: ignore
    from src.utils.LoadClasses import LoadClasses  # type: ignore
except Exception:  # pragma: no cover
    ALL_PATHOGENS = None  # type: ignore
    LoadClasses = None  # type: ignore


_UNNAMED_RE = re.compile(r"^Unnamed(?::\s*\d+)?$")
_OUTCOME_ALLOWED = {"S", "I", "R"}

# Common "missing" tokens seen in exports
_MISSING_TOKENS = {
    "", " ", "NA", "N/A", "NULL", "NONE", "NAN", "-", "--", "?", "ND", "NOT DONE", "NOTDONE"
}


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


@dataclass(frozen=True)
class CascadeSpec:
    A_code: str
    B_code: str
    recode_mode: str = "R_vs_nonR"  # "R_vs_nonR" or "NS_vs_S"
    condition_on_A_tested: bool = True
    drop_I: bool = True
    y_mode: str = "B_tested"  # "B_tested" | "B_resistant" | "num_tests"


@dataclass(frozen=True)
class CohortMeta:
    name: str
    n_rows: int
    n_labs: int
    yearmonth_min: Optional[str]
    yearmonth_max: Optional[str]
    pathogen: Optional[str]
    filter_fingerprint: str
    filter_report: List[Dict[str, Any]]
    exclusions_applied: bool
    created_at_utc: str
    notes: Dict[str, Any]


FilterDict = Dict[str, Union[str, int, float, bool, List[Union[str, int, float, bool]]]]


class DataLoader:
    TESTED_SUFFIX = "_Tested"
    OUTCOME_SUFFIX = "_Outcome"

    # If these columns exist, we can validate provided panel depth vs computed
    PANEL_DEPTH_COL = "TotalAntibioticsTested"
    LAB_COL = "Anonymized_Lab"
    YM_COL = "YearMonth"
    PATHOGEN_COL = "Pathogen"

    def __init__(
        self,
        filepath: str,
        pathogen_groups_regex: Sequence[str] | str = (),
        *,
        strict: bool = False,
        normalize_on_load: bool = True,
        validate_panel_depth: bool = False,
        max_violation_examples: int = 5,
    ):
        """
        strict:
          - if True, raise on any critical violations (bad tested values, impossible outcomes, etc.)
          - if False, coerce/clean conservatively and record diagnostics

        normalize_on_load:
          - normalize tested + outcomes once at init (recommended)

        validate_panel_depth:
          - if True and TotalAntibioticsTested exists, compute discrepancies and store diagnostics
        """
        self.filepath = filepath
        self.strict = bool(strict)
        self.normalize_on_load = bool(normalize_on_load)
        self.validate_panel_depth_flag = bool(validate_panel_depth)
        self.max_violation_examples = int(max_violation_examples)

        self.load = LoadClasses() if LoadClasses is not None else None

        df = read_any(filepath)

        # Basic schema guard
        self.ensure_unique_columns(df)

        # Harmonize ward label issue
        if "ARS_WardType" in df.columns:
            df = df.assign(
                ARS_WardType=df["ARS_WardType"].replace(
                    {
                        "Early Rehabilitation": "Rehabilitation",
                        "Rehabilitation": "Rehabilitation",
                    }
                )
            )

        if self.PATHOGEN_COL not in df.columns:
            raise ValueError(f"Expected '{self.PATHOGEN_COL}' column not found.")

        # Restrict to requested pathogen(s)
        pattern = None
        if pathogen_groups_regex:
            if isinstance(pathogen_groups_regex, (list, tuple, set)):
                pattern = "|".join(f"(?:{p})" for p in pathogen_groups_regex if p)
            else:
                pattern = str(pathogen_groups_regex)
        else:
            pattern = ALL_PATHOGENS  # may be None

        if pattern:
            df = df[df[self.PATHOGEN_COL].astype("string").str.contains(pattern, case=False, na=False, regex=True)]

        # Drop known artifact columns
        artifact_cols = [c for c in df.columns if str(c).endswith("_Tested_Outcome")]
        if artifact_cols:
            df = df.drop(columns=artifact_cols, errors="ignore")

        self.df = df.reset_index(drop=True)
        self.all_cols = self.df.columns.to_list()

        # Identify antibiotic tested/outcome columns (robust: suffix-only)
        self.abx_tested_cols = sorted([c for c in self.all_cols if str(c).endswith(self.TESTED_SUFFIX)])
        self.abx_outcome_cols = sorted([c for c in self.all_cols if str(c).endswith(self.OUTCOME_SUFFIX)])

        self.tested_bases = [c[: -len(self.TESTED_SUFFIX)] for c in self.abx_tested_cols]
        self.outcome_bases = [c[: -len(self.OUTCOME_SUFFIX)] for c in self.abx_outcome_cols]

        self.paired_bases = sorted(set(self.tested_bases).intersection(self.outcome_bases))
        self.meta_cols = [c for c in self.all_cols if c not in (self.abx_tested_cols + self.abx_outcome_cols)]

        # Report pairing diagnostics
        self.pairing_diagnostics = self._pairing_diagnostics()

        # Fast lookup map code->base
        self.code_to_base: Dict[str, str] = {self._code_from_base(b).upper(): b for b in self.paired_bases}

        # Normalization + sanity
        self.diagnostics: Dict[str, Any] = {}
        if self.normalize_on_load:
            self._normalize_tested_columns()
            self._normalize_outcome_columns()
            self._enforce_tested_outcome_consistency()

        if self.validate_panel_depth_flag:
            self.diagnostics["panel_depth_validation"] = self._validate_panel_depth()

        # Caches with keys to avoid stale reuse
        self._cache: Dict[str, Any] = {}

        # last filter reporting
        self.last_filter_report: Optional[List[Dict[str, Any]]] = None
        self.last_filter_name: Optional[str] = None

        # for convenience (debug)
        self.filtered_df: Optional[pd.DataFrame] = None

    # ------------------------------ hashing / cache keys ------------------------------ #

    def _stable_hash(self, obj: Any) -> str:
        try:
            s = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=True)
        except Exception:
            s = str(obj)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    def _cache_key(self, prefix: str, payload: Dict[str, Any]) -> str:
        return f"{prefix}:{self._stable_hash(payload)}"

    def _stable_filter_fingerprint(self, obj: Any) -> Any:
        """
        Attempt stable serialization for cache keys.
        - prefers .to_json() if available
        - then .to_dict()
        - else str(obj)
        """
        if obj is None:
            return None
        for meth in ("to_json", "to_dict"):
            if hasattr(obj, meth) and callable(getattr(obj, meth)):
                try:
                    return getattr(obj, meth)()
                except Exception:
                    pass
        return str(obj)

    # ------------------------------ base/code helpers ------------------------------ #

    @staticmethod
    def _code_from_base(base: str) -> str:
        # supports "AMC - Amoxicillin..." and also "AMC" style
        s = str(base)
        if " - " in s:
            return s.split(" - ")[0].strip()
        return s.strip()

    def base_from_code(self, code: str) -> str:
        code_u = str(code).strip().upper()
        if code_u in self.code_to_base:
            return self.code_to_base[code_u]
        raise ValueError(f"Antibiotic code {code} not found among paired antibiotics.")

    def _tested_col(self, base: str) -> str:
        return f"{base}{self.TESTED_SUFFIX}"

    def _outcome_col(self, base: str) -> str:
        return f"{base}{self.OUTCOME_SUFFIX}"

    # ------------------------------ schema guards ------------------------------ #

    @staticmethod
    def ensure_unique_columns(df: pd.DataFrame) -> None:
        if not df.columns.is_unique:
            dupes = df.columns[df.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate columns detected: {dupes[:25]}{'...' if len(dupes) > 25 else ''}")

    @staticmethod
    def validate_matrix_schema(matrix: pd.DataFrame) -> None:
        DataLoader.ensure_unique_columns(matrix)
        bad = [c for c in matrix.columns if (c is None) or (str(c).strip() == "")]
        if bad:
            raise ValueError(f"Found empty/invalid column names: {bad[:10]}")

    # ------------------------------ normalization ------------------------------ #

    @staticmethod
    def _coerce_tested_to_int01(s: pd.Series) -> pd.Series:
        """
        Coerce tested flags to strict 0/1 int.
        Handles: 0/1, 0.0/1.0, '0'/'1', True/False, 'yes'/'no', etc.
        """
        if s.dtype == bool:
            return s.astype(int)

        ss = s.astype("string")
        lowered = ss.str.strip().str.lower()

        bool_map = {
            "true": 1, "t": 1, "yes": 1, "y": 1,
            "false": 0, "f": 0, "no": 0, "n": 0,
        }
        mapped = lowered.map(bool_map)

        numeric = pd.to_numeric(ss, errors="coerce")
        out = mapped.where(mapped.notna(), numeric)

        out = out.fillna(0)
        out = (out > 0).astype(int)
        return out

    @staticmethod
    def _as_int01(s: pd.Series) -> pd.Series:
        """Public internal helper used throughout. Returns Int8 0/1 with no NA."""
        return DataLoader._coerce_tested_to_int01(s).astype("Int8")

    @staticmethod
    def _normalize_outcome_series(y: pd.Series) -> pd.Series:
        """
        Normalize outcome to {S, I, R, NA}.
        - strips whitespace/punctuation
        - uppercases
        - maps common missing tokens to NA
        - anything not in {S,I,R} becomes NA
        """
        s = y.astype("string")
        s = s.str.strip()
        s = s.str.replace(r"[,\.;:]+$", "", regex=True).str.strip()
        s = s.str.upper()
        s = s.where(~s.isin(list(_MISSING_TOKENS)), pd.NA)
        s = s.where(s.isin(list(_OUTCOME_ALLOWED)), pd.NA)
        return s

    def _normalize_tested_columns(self) -> None:
        violations = []
        for col in self.abx_tested_cols:
            coerced = self._coerce_tested_to_int01(self.df[col])
            bad = ~coerced.isin([0, 1])
            if bad.any():
                violations.append({"column": col, "n_bad": int(bad.sum())})
            self.df[col] = coerced.astype("Int8")

        if violations:
            msg = f"Tested coercion produced non-binary values in {len(violations)} columns."
            if self.strict:
                raise ValueError(msg + f" Examples: {violations[:self.max_violation_examples]}")
            self.diagnostics["tested_nonbinary_violations"] = violations

    def _normalize_outcome_columns(self) -> None:
        invalid_counts = []
        for col in self.abx_outcome_cols:
            before = self.df[col]
            normed = self._normalize_outcome_series(before)

            n_before = int(before.notna().sum())
            n_after = int(normed.notna().sum())
            n_lost = max(0, n_before - n_after)
            if n_lost > 0:
                invalid_counts.append({"column": col, "n_invalid_mapped_to_na": n_lost})

            self.df[col] = normed

        if invalid_counts:
            self.diagnostics["outcome_invalid_mapped_to_na"] = invalid_counts[:500]

    def _enforce_tested_outcome_consistency(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Enforce:
          1) Tested == 0 -> Outcome must be NA
          2) Outcome in {S,I,R} -> Tested must be 1

        If df is None: mutates self.df (legacy behavior).
        If df is provided: returns a fixed copy (does not mutate self.df).
        """
        target = self.df if df is None else df.copy()

        violations = []
        fixed_outcome_to_na = 0
        fixed_tested_to_1 = 0

        for base in self.paired_bases:
            tcol = self._tested_col(base)
            ycol = self._outcome_col(base)
            if tcol not in target.columns or ycol not in target.columns:
                continue

            t = self._as_int01(target[tcol]).fillna(0).astype("Int8")
            y = self._normalize_outcome_series(target[ycol])  # normalize defensively

            # (1) tested==0 but outcome present -> outcome -> NA
            bad1 = (t == 0) & y.notna()
            if bad1.any():
                n_bad1 = int(bad1.sum())
                fixed_outcome_to_na += n_bad1
                if self.strict and df is None:
                    examples = target.loc[bad1, [tcol, ycol]].head(self.max_violation_examples).to_dict("records")
                    raise ValueError(f"Inconsistent data: {tcol}==0 but {ycol} present. n={n_bad1}, ex={examples}")
                target.loc[bad1, ycol] = pd.NA
                violations.append({"base": base, "type": "tested0_outcome_present", "n": n_bad1})

            # (2) outcome present but tested==0 -> tested -> 1 (rare export bug)
            t2 = self._as_int01(target[tcol]).fillna(0).astype("Int8")
            y2 = self._normalize_outcome_series(target[ycol])
            bad2 = (t2 == 0) & y2.notna()
            if bad2.any():
                n_bad2 = int(bad2.sum())
                fixed_tested_to_1 += n_bad2
                if self.strict and df is None:
                    examples = target.loc[bad2, [tcol, ycol]].head(self.max_violation_examples).to_dict("records")
                    raise ValueError(f"Inconsistent data: {ycol} present but {tcol}==0. n={n_bad2}, ex={examples}")
                target.loc[bad2, tcol] = 1
                violations.append({"base": base, "type": "outcome_present_tested0", "n": n_bad2})

            # write back normalized outcome (defensive)
            target[ycol] = self._normalize_outcome_series(target[ycol])

        self.diagnostics["consistency_fixes"] = {
            "fixed_outcome_to_na": int(fixed_outcome_to_na),
            "fixed_tested_to_1": int(fixed_tested_to_1),
            "violations": violations[:500],
        }

        if df is None:
            self.df = target
        return target

    def _pairing_diagnostics(self) -> Dict[str, Any]:
        tested_only = sorted(set(self.tested_bases) - set(self.outcome_bases))
        outcome_only = sorted(set(self.outcome_bases) - set(self.tested_bases))
        paired = sorted(set(self.tested_bases).intersection(self.outcome_bases))
        # useful: how many outcome cols did not follow expected "CODE - Name" format
        outcome_no_dash = [b for b in self.outcome_bases if " - " not in str(b)]
        return {
            "n_tested_cols": len(self.abx_tested_cols),
            "n_outcome_cols": len(self.abx_outcome_cols),
            "n_paired": len(paired),
            "n_tested_only": len(tested_only),
            "n_outcome_only": len(outcome_only),
            "tested_only_examples": tested_only[:20],
            "outcome_only_examples": outcome_only[:20],
            "n_outcome_bases_without_dash": int(len(outcome_no_dash)),
            "outcome_bases_without_dash_examples": outcome_no_dash[:20],
        }

    # ------------------------------ exclusions + filtering ------------------------------ #

    @staticmethod
    def _parse_truthy(series: pd.Series) -> pd.Series:
        if series.dtype == bool:
            return series.fillna(False)
        s = series.astype("string").str.strip().str.lower()
        truthy = s.isin(["true", "1", "yes", "y", "t"])
        falsy = s.isin(["false", "0", "no", "n", "f"])
        out = pd.Series(False, index=series.index)
        out = out.where(~truthy, True)
        out = out.where(~falsy, False)
        num = pd.to_numeric(series, errors="coerce")
        out = out.where(num.isna(), num.fillna(0).astype(float) > 0)
        return out.fillna(False)

    def _apply_exclusions(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        out = df
        removed_counts: Dict[str, int] = {}
        for col in [
            "IsSpecificlyExcluded_Screening",
            "IsSpecificlyExcluded_Pathogen",
            "IsSpecificlyExcluded_PathogenevidenceNegative",
        ]:
            if col in out.columns:
                before = len(out)
                bad = self._parse_truthy(out[col])
                out = out.loc[~bad].copy()
                removed_counts[col] = before - len(out)
        return out, removed_counts

    def _apply_simple_filters(
        self, df: pd.DataFrame, filters: FilterDict
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        reports: List[Dict[str, Any]] = []
        out = df

        missing_cols = [col for col in filters.keys() if col not in out.columns]
        if missing_cols:
            raise ValueError(f"Filter columns not found: {missing_cols}")

        for col, values in filters.items():
            n_before = len(out)
            if isinstance(values, (str, int, float, bool)):
                values_list = [values]
            else:
                values_list = list(values)

            # keep it simple + predictable: string matching (as before)
            values_str = [str(v) for v in values_list]
            mask = out[col].astype("string").isin(values_str)
            out = out.loc[mask].copy()

            n_after = len(out)
            reports.append(
                {
                    "column": col,
                    "operator": "in",
                    "values": values_str,
                    "n_before": n_before,
                    "n_after": n_after,
                    "n_removed": n_before - n_after,
                    "pct_retained": (n_after / n_before * 100) if n_before else 0.0,
                }
            )

        return out.reset_index(drop=True), reports

    def apply_filters_and_exclusions(
        self,
        *,
        filters: Optional[FilterDict] = None,
        filter_config: Optional[Any] = None,
        filter_pipeline: Optional[Any] = None,
        # convenience filters
        CSQ: Optional[Union[str, Sequence[str]]] = None,
        CSY: Optional[Union[str, Sequence[str]]] = None,
        CSQMG: Optional[Union[str, Sequence[str]]] = None,
        CSYMG: Optional[Union[str, Sequence[str]]] = None,
        apply_exclusions: bool = True,
        verbose: bool = True,
        stop_on_empty: bool = True,
    ) -> pd.DataFrame:
        provided = [filters is not None, filter_config is not None, filter_pipeline is not None]
        if sum(provided) > 1:
            raise ValueError("Provide only one of: filters, filter_config, filter_pipeline.")

        df = self.df.copy()
        report: List[Dict[str, Any]] = []
        name: str = "none"

        exclusion_counts = {}
        if apply_exclusions:
            df, exclusion_counts = self._apply_exclusions(df)
            if exclusion_counts:
                report.append(
                    {
                        "column": "__exclusions__",
                        "operator": "drop_truthy",
                        "values": exclusion_counts,
                        "n_before": None,
                        "n_after": len(df),
                        "n_removed": int(sum(exclusion_counts.values())),
                        "pct_retained": None,
                    }
                )

        if filter_pipeline is not None:
            df, rep = filter_pipeline.apply(df, verbose=verbose, stop_on_empty=stop_on_empty)
            report.extend(rep)
            name = getattr(filter_pipeline, "name", "filter_pipeline")

        elif filter_config is not None:
            df, rep = filter_config.apply(df)
            report.extend(rep)
            name = getattr(filter_config, "name", "filter_config")

        elif filters is not None:
            df, rep = self._apply_simple_filters(df, filters)
            report.extend(rep)
            name = "simple_dict_filters"

        # Apply CSQ* convenience filters (if present)
        extra = {"CSQ": CSQ, "CSY": CSY, "CSQMG": CSQMG, "CSYMG": CSYMG}
        for col, vals in extra.items():
            if vals is None or col not in df.columns:
                continue
            before = len(df)
            if isinstance(vals, (str, int, float, bool)):
                vlist = [vals]
            else:
                vlist = list(vals)
            mask = df[col].astype("string").isin([str(v) for v in vlist])
            df = df.loc[mask].copy()
            after = len(df)
            report.append(
                {
                    "column": col,
                    "operator": "in",
                    "values": [str(v) for v in vlist],
                    "n_before": before,
                    "n_after": after,
                    "n_removed": before - after,
                    "pct_retained": (after / before * 100) if before else 0.0,
                }
            )
            if stop_on_empty and df.empty:
                raise ValueError(f"Filtering produced 0 rows after applying {col}={vals}.")

        if verbose and report:
            print("\n" + "=" * 80)
            print(f"APPLYING FILTERS ({name})")
            print("=" * 80)
            for r in report:
                if r["column"] == "__exclusions__":
                    print(f"[exclusions] dropped={r['n_removed']:,} detail={r['values']}")
                else:
                    print(
                        f"[{r['column']}:{r['operator']}] {r['n_before']:,} → {r['n_after']:,} "
                        f"({r['pct_retained']:.1f}% retained, {r['n_removed']:,} dropped)"
                    )
            print("=" * 80 + "\n")

        self.last_filter_report = report
        self.last_filter_name = name
        return df.reset_index(drop=True)

    # ------------------------------ cohort meta ------------------------------ #

    def get_cohort(
        self,
        *,
        filters: Optional[FilterDict] = None,
        filter_config: Optional[Any] = None,
        filter_pipeline: Optional[Any] = None,
        apply_exclusions: bool = True,
        verbose: bool = True,
        stop_on_empty: bool = True,
        notes: Optional[Dict[str, Any]] = None,
        validate_panel_depth: Optional[bool] = None,
    ) -> Tuple[pd.DataFrame, CohortMeta]:
        d = self.apply_filters_and_exclusions(
            filters=filters,
            filter_config=filter_config,
            filter_pipeline=filter_pipeline,
            apply_exclusions=apply_exclusions,
            verbose=verbose,
            stop_on_empty=stop_on_empty,
        )
        if d.empty:
            raise ValueError("Cohort is empty after filtering/exclusions.")

        name = self.last_filter_name or "unknown"
        report = self.last_filter_report or []

        # Determine fingerprint source (stable)
        if filter_config is not None:
            fp_source = self._stable_filter_fingerprint(filter_config)
        elif filters is not None:
            fp_source = filters
        elif filter_pipeline is not None:
            fp_source = self._stable_filter_fingerprint(filter_pipeline)
        else:
            fp_source = "no_filters"

        filter_fingerprint = self._stable_hash(fp_source)

        n_labs = int(d[self.LAB_COL].nunique()) if self.LAB_COL in d.columns else 0
        ym_min = str(d[self.YM_COL].min()) if self.YM_COL in d.columns else None
        ym_max = str(d[self.YM_COL].max()) if self.YM_COL in d.columns else None

        pathogen = None
        if self.PATHOGEN_COL in d.columns:
            vals = d[self.PATHOGEN_COL].astype("string").dropna().unique().tolist()
            if len(vals) == 1:
                pathogen = str(vals[0])

        extra_notes: Dict[str, Any] = {
            "pairing_diagnostics": self.pairing_diagnostics,
            "loader_diagnostics": self.diagnostics,
        }

        do_vpd = self.validate_panel_depth_flag if validate_panel_depth is None else bool(validate_panel_depth)
        if do_vpd:
            extra_notes["cohort_panel_depth_validation"] = self._validate_panel_depth(d)

        merged_notes = dict(extra_notes)
        if notes:
            merged_notes.update(notes)

        meta = CohortMeta(
            name=name,
            n_rows=int(len(d)),
            n_labs=n_labs,
            yearmonth_min=ym_min,
            yearmonth_max=ym_max,
            pathogen=pathogen,
            filter_fingerprint=filter_fingerprint,
            filter_report=report,
            exclusions_applied=apply_exclusions,
            created_at_utc=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            notes=merged_notes,
        )
        return d, meta

    # ------------------------------ prevalence table ------------------------------ #

    def antibiotic_prevalence_table(self, df: pd.DataFrame, *, recode_mode: str = "R_vs_nonR") -> pd.DataFrame:
        rows = []
        n = len(df)
        if n == 0:
            raise ValueError("Cannot compute prevalence table on empty dataframe.")

        for base in self.paired_bases:
            tcol = self._tested_col(base)
            ycol = self._outcome_col(base)
            if tcol not in df.columns or ycol not in df.columns:
                continue

            tested = (self._as_int01(df[tcol]) == 1)
            tested_count = int(tested.sum())
            tested_rate = tested_count / n if n else 0.0

            y = self._normalize_outcome_series(df.loc[tested, ycol])

            if recode_mode == "R_vs_nonR":
                rmask = (y == "R")
            elif recode_mode == "NS_vs_S":
                rmask = y.isin(["R", "I"])
            else:
                raise ValueError("recode_mode must be 'R_vs_nonR' or 'NS_vs_S'")

            res_count = int(rmask.sum())
            res_rate = (res_count / tested_count) if tested_count else 0.0

            rows.append(
                {
                    "code": self._code_from_base(base),
                    "base": base,
                    "tested_rate": float(tested_rate),
                    "tested_count": int(tested_count),
                    "res_rate_tested": float(res_rate),
                    "res_count_tested": int(res_count),
                }
            )

        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return out.sort_values(["tested_rate", "res_rate_tested"], ascending=False).reset_index(drop=True)

    # ------------------------------ recoding masks ------------------------------ #

    def recode_outcome_mask(
        self,
        y: pd.Series,
        *,
        recode_mode: str = "R_vs_nonR",
        drop_I: bool = False,
    ) -> pd.Series:
        yn = self._normalize_outcome_series(y)

        if drop_I:
            yn = yn.where(yn.isin(["S", "R"]), pd.NA)

        if recode_mode == "R_vs_nonR":
            return (yn == "R").fillna(False)
        if recode_mode == "NS_vs_S":
            return yn.isin(["R", "I"]).fillna(False)
        raise ValueError("recode_mode must be 'R_vs_nonR' or 'NS_vs_S'")

    def get_abx_flags(
        self,
        df: pd.DataFrame,
        codes: List[str],
        *,
        recode_mode: str = "R_vs_nonR",
        drop_I: bool = False,
        prefix: str = "",
    ) -> pd.DataFrame:
        """
        Efficient flag builder:
        - A_T: tested flag (0/1)
        - A_R: tested AND resistant/non-susceptible (0/1)

        Uses dict->DataFrame construction to avoid pandas fragmentation warnings.
        """
        if df is None or df.empty or not codes:
            return pd.DataFrame(index=df.index if df is not None else None)

        data: Dict[str, np.ndarray] = {}

        for code in codes:
            code_u = str(code).strip().upper()
            try:
                base = self.base_from_code(code_u)
            except Exception:
                continue

            tcol = f"{base}{self.TESTED_SUFFIX}"
            ycol = f"{base}{self.OUTCOME_SUFFIX}"
            if tcol not in df.columns or ycol not in df.columns:
                continue

            tested = (self._as_int01(df[tcol]) == 1)
            rmask = self.recode_outcome_mask(
                df[ycol],
                recode_mode=recode_mode,
                drop_I=drop_I,
            )

            data[f"{prefix}{code_u}_T"] = tested.astype(np.int8).to_numpy()
            data[f"{prefix}{code_u}_R"] = (tested & rmask).astype(np.int8).to_numpy()

        out = pd.DataFrame(data, index=df.index)
        return out.copy()

    # ------------------------------ screening (directional) ------------------------------ #

    def screen_triggers(
        self,
        df: pd.DataFrame,
        *,
        target_code: str,
        candidate_codes: List[str],
        recode_mode: str = "R_vs_nonR",
        drop_I: bool = False,
        min_group: int = 50,
        condition_on_A_tested: bool = True,
    ) -> pd.DataFrame:
        """
        Δ = P(D_tested | A_R) - P(D_tested | A_S)

        Groups are disjoint:
          A_R group: outcome=='R' (or NS if NS_vs_S)
          A_S group: outcome=='S' (strictly susceptible)
        """
        D_base = self.base_from_code(target_code)
        D_test = self._tested_col(D_base)
        if D_test not in df.columns:
            raise ValueError(f"Target tested column missing: {D_test}")

        Dy = (self._as_int01(df[D_test]) == 1)

        rows = []
        for A in candidate_codes:
            if A == target_code:
                continue

            A_base = self.base_from_code(A)
            A_test = self._tested_col(A_base)
            A_out = self._outcome_col(A_base)
            if A_test not in df.columns or A_out not in df.columns:
                continue

            testedA = (self._as_int01(df[A_test]) == 1)
            sub = df.loc[testedA].copy() if condition_on_A_tested else df.copy()
            if sub.empty:
                continue

            yA = self._normalize_outcome_series(sub[A_out])

            if drop_I:
                keep = yA.isin(["S", "R"])
                sub = sub.loc[keep].copy()
                yA = self._normalize_outcome_series(sub[A_out])
            if sub.empty:
                continue

            if recode_mode == "R_vs_nonR":
                AR = (yA == "R")
            elif recode_mode == "NS_vs_S":
                AR = yA.isin(["R", "I"])
            else:
                raise ValueError("recode_mode must be 'R_vs_nonR' or 'NS_vs_S'")

            AS = (yA == "S")

            nR = int(AR.sum())
            nS = int(AS.sum())
            if nR < min_group or nS < min_group:
                continue

            Dy_sub = Dy.loc[sub.index]
            pR = float(Dy_sub.loc[AR].mean())
            pS = float(Dy_sub.loc[AS].mean())

            rows.append(
                {
                    "A": A,
                    "target_D": target_code,
                    "delta": float(pR - pS),
                    "p_D_tested_given_A_R": pR,
                    "p_D_tested_given_A_S": pS,
                    "n_A_R": nR,
                    "n_A_S": nS,
                }
            )

        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return out.sort_values("delta", ascending=False).reset_index(drop=True)

    # ------------------------------ panel depth ------------------------------ #

    def compute_panel_depth(self, df: pd.DataFrame, *, codes: Optional[List[str]] = None) -> pd.Series:
        if codes is None:
            tested_cols = self.abx_tested_cols
        else:
            tested_cols = []
            for code in codes:
                base = self.base_from_code(code)
                tcol = self._tested_col(base)
                if tcol in df.columns:
                    tested_cols.append(tcol)

        if not tested_cols:
            return pd.Series(np.zeros(len(df), dtype=int), index=df.index)

        tested_mat = df[tested_cols].apply(self._as_int01).fillna(0).astype(int)
        return tested_mat.sum(axis=1).astype(int)

    def _validate_panel_depth(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        d = self.df if df is None else df
        if self.PANEL_DEPTH_COL not in d.columns:
            return {"available": False, "reason": f"{self.PANEL_DEPTH_COL} not in data"}

        provided = pd.to_numeric(d[self.PANEL_DEPTH_COL], errors="coerce").fillna(0).astype(int)
        computed = self.compute_panel_depth(d)

        diff = (provided - computed).astype(int)
        absdiff = diff.abs()

        summary = {
            "available": True,
            "n_rows": int(len(d)),
            "provided_mean": float(provided.mean()),
            "computed_mean": float(computed.mean()),
            "absdiff_mean": float(absdiff.mean()),
            "pct_exact_match": float((absdiff == 0).mean() * 100.0),
            "pct_absdiff_ge_1": float((absdiff >= 1).mean() * 100.0),
            "pct_absdiff_ge_3": float((absdiff >= 3).mean() * 100.0),
            "examples": diff.value_counts().head(10).to_dict(),
        }
        return summary

    def lab_panel_summary(
        self,
        df: pd.DataFrame,
        *,
        codes: Optional[List[str]] = None,
        lab_col: str = LAB_COL,
    ) -> pd.DataFrame:
        if lab_col not in df.columns:
            raise ValueError(f"lab_col not found: {lab_col}")

        depth = self.compute_panel_depth(df, codes=codes)
        tmp = df[[lab_col]].copy()
        tmp["panel_depth"] = depth.values

        g = tmp.groupby(lab_col)["panel_depth"]
        out = g.agg(["count", "mean", "median", "min", "max", "std"]).reset_index()
        out = out.rename(columns={"count": "n_isolates", "mean": "depth_mean", "std": "depth_std"})
        return out.sort_values("depth_mean", ascending=False).reset_index(drop=True)

    # ------------------------------ transaction matrix ------------------------------ #

    def get_transaction_matrix(
        self,
        *,
        filters: Optional[FilterDict] = None,
        filter_config: Optional[Any] = None,
        filter_pipeline: Optional[Any] = None,
        CSQ: Optional[Union[str, Sequence[str]]] = None,
        CSY: Optional[Union[str, Sequence[str]]] = None,
        CSQMG: Optional[Union[str, Sequence[str]]] = None,
        CSYMG: Optional[Union[str, Sequence[str]]] = None,
        recode_mode: str = "R_vs_nonR",
        verbose: bool = True,
        covariate_cols: Optional[List[str]] = None,
        include_covariates: bool = True,
        max_levels: int = 25,
        min_count: int = 50,
        apply_exclusions: bool = True,
        min_test_rate: float = 0.0,
        min_test_count: int = 0,
        min_res_rate: float = 0.0,
        min_res_count: int = 0,
        max_antibiotics: Optional[int] = None,
        always_keep: Optional[List[str]] = None,
        drop_all_susceptible_rows: bool = False,
        enforce_normalization: bool = True,
    ) -> pd.DataFrame:
        """
        Build boolean transaction matrix:
          <CODE>_T  tested
          <CODE>_R  resistant/non-susceptible (per recode_mode)
          plus covariate one-hot items (optional)

        enforce_normalization:
          - if True (default), normalizes outcomes + tested and enforces T/Y consistency
            on the cohort slice used to build the matrix (even if normalize_on_load=False).
        """

        cache_payload = {
            "filters": filters,
            "filter_config": self._stable_filter_fingerprint(filter_config),
            "filter_pipeline": self._stable_filter_fingerprint(filter_pipeline),
            "CSQ": CSQ, "CSY": CSY, "CSQMG": CSQMG, "CSYMG": CSYMG,
            "recode_mode": recode_mode,
            "covariate_cols": covariate_cols,
            "include_covariates": include_covariates,
            "max_levels": max_levels,
            "min_count": min_count,
            "apply_exclusions": apply_exclusions,
            "min_test_rate": min_test_rate,
            "min_test_count": min_test_count,
            "min_res_rate": min_res_rate,
            "min_res_count": min_res_count,
            "max_antibiotics": max_antibiotics,
            "always_keep": always_keep,
            "drop_all_susceptible_rows": drop_all_susceptible_rows,
            "enforce_normalization": enforce_normalization,
        }
        key = self._cache_key("txn", cache_payload)
        if key in self._cache:
            m = self._cache[key]
            if verbose:
                print("✓ Using cached transaction matrix")
            return m.copy()

        df = self.apply_filters_and_exclusions(
            filters=filters,
            filter_config=filter_config,
            filter_pipeline=filter_pipeline,
            CSQ=CSQ, CSY=CSY, CSQMG=CSQMG, CSYMG=CSYMG,
            apply_exclusions=apply_exclusions,
            verbose=verbose,
        )
        if df.empty:
            raise ValueError("No rows left after filtering/exclusions.")
        if not self.paired_bases:
            raise ValueError("No paired tested/outcome antibiotics detected.")

        # Enforce normalization/consistency on the cohort slice (recommended + safer)
        if enforce_normalization:
            # normalize just what's in df (cheap compared to later bugs)
            for c in self.abx_tested_cols:
                if c in df.columns:
                    df[c] = self._as_int01(df[c])
            for c in self.abx_outcome_cols:
                if c in df.columns:
                    df[c] = self._normalize_outcome_series(df[c])
            df = self._enforce_tested_outcome_consistency(df)

        self.filtered_df = df.copy()

        # Dynamic antibiotic selection
        prev = self.antibiotic_prevalence_table(df, recode_mode=recode_mode)
        keep_mask = (
            (prev["tested_rate"] >= float(min_test_rate))
            & (prev["tested_count"] >= int(min_test_count))
            & (prev["res_rate_tested"] >= float(min_res_rate))
            & (prev["res_count_tested"] >= int(min_res_count))
        )
        keep_codes = set(prev.loc[keep_mask, "code"].astype(str).str.strip().str.upper().tolist())
        if always_keep:
            keep_codes |= set([str(c).strip().upper() for c in always_keep])

        if max_antibiotics is not None and len(keep_codes) > 0:
            ranked = prev[prev["code"].isin(keep_codes)].copy()
            ranked = ranked.sort_values(["tested_rate", "res_rate_tested"], ascending=False)
            keep_codes = set(ranked.head(int(max_antibiotics))["code"].astype(str).str.upper().tolist())

        selected_bases = [b for b in self.paired_bases if self._code_from_base(b).upper() in keep_codes]
        if not selected_bases:
            raise ValueError("Antibiotic selection removed all antibiotics. Relax thresholds or set always_keep.")

        if covariate_cols is None:
            covariate_cols = ["ARS_WardType", "CareType", "AgeGroup", "Year"]

        # Precompute tested Int8 for selected antibiotics (speed)
        tested_int: Dict[str, pd.Series] = {}
        for base in selected_bases:
            tcol = self._tested_col(base)
            if tcol in df.columns:
                tested_int[tcol] = self._as_int01(df[tcol])

        # Build T items (tested)
        T = pd.DataFrame(index=df.index)
        for base in selected_bases:
            code = self._code_from_base(base).upper()
            tcol = self._tested_col(base)
            tvals = tested_int.get(tcol, self._as_int01(df[tcol]))
            T[f"{code}_T"] = (tvals == 1)

        # Build R items
        R = pd.DataFrame(index=df.index)
        for base in selected_bases:
            code = self._code_from_base(base).upper()
            ycol = self._outcome_col(base)
            y = df[ycol]
            rmask = self.recode_outcome_mask(y, recode_mode=recode_mode, drop_I=False)
            tcol = self._tested_col(base)
            tmask = (tested_int.get(tcol, self._as_int01(df[tcol])) == 1)
            R[f"{code}_R"] = (rmask & tmask).fillna(False)

        # Covariates one-hot
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
        self.validate_matrix_schema(matrix)

        # Drop all-false columns
        nonzero_cols = matrix.columns[(matrix.sum(axis=0) > 0).to_numpy()].tolist()
        matrix = matrix[nonzero_cols]

        # Optionally drop all-susceptible rows (all R==False)
        r_items = [c for c in matrix.columns if c.endswith("_R")]
        n_before = matrix.shape[0]
        dropped_all_s = 0
        if drop_all_susceptible_rows and r_items:
            keep_rows = matrix[r_items].any(axis=1)
            dropped_all_s = int((~keep_rows).sum())
            matrix = matrix.loc[keep_rows].reset_index(drop=True)

        if verbose:
            r_items = [c for c in matrix.columns if c.endswith("_R")]
            t_items = [c for c in matrix.columns if c.endswith("_T")]
            cov_items = [c for c in matrix.columns if c not in r_items and c not in t_items]
            abx_codes = sorted({c[:-2] for c in (r_items + t_items)})

            print(f"✓ Rows after filters (before R-only row drop): {n_before:,}")
            if drop_all_susceptible_rows and r_items:
                print(f"✓ Dropped rows with all _R == False: {dropped_all_s:,} ({dropped_all_s / n_before * 100:.1f}%)")
            print(f"✓ Final rows in transaction matrix: {matrix.shape[0]:,}")
            print(f"✓ Selected antibiotics: {len(selected_bases)} (from {len(self.paired_bases)})")
            print(f"  • Codes: {', '.join(abx_codes)}")
            print(f"✓ Transaction matrix: {matrix.shape[0]} rows × {matrix.shape[1]} items")
            print(f"  • R items: {len(r_items)} | T items: {len(t_items)} | Covariates: {len(cov_items)}")

        
        # Audit: R implies T
        viol = {}
        for code in sorted({c[:-2] for c in matrix.columns if isinstance(c, str) and c.endswith("_R")}):
            r = f"{code}_R"
            t = f"{code}_T"
            if t in matrix.columns:
                bad = matrix[r] & ~matrix[t]
                n_bad = int(bad.sum())
                if n_bad:
                    viol[code] = n_bad

        if viol:
            msg = f"Found R without T for {len(viol)} antibiotics. Examples: {list(viol.items())[:10]}"
            if self.strict:
                raise ValueError(msg)
            self.diagnostics["txn_R_without_T"] = viol
    
        self._cache[key] = matrix.copy()
        return matrix

    # ------------------------------ convenience helpers ------------------------------ #

    def get_selected_abx_codes(self, matrix: pd.DataFrame) -> List[str]:
        if matrix is None or matrix.empty:
            return []
        codes = set()
        for c in matrix.columns:
            if isinstance(c, str) and (c.endswith("_T") or c.endswith("_R")) and len(c) > 2:
                codes.add(c[:-2])
        return sorted(codes)

    def get_tested_items(self, matrix: Optional[pd.DataFrame] = None) -> List[str]:
        m = matrix
        if m is None:
            raise ValueError("Provide a matrix (returned by get_transaction_matrix).")
        return [c for c in m.columns if c.endswith("_T")]

    def get_resistant_items(self, matrix: Optional[pd.DataFrame] = None) -> List[str]:
        m = matrix
        if m is None:
            raise ValueError("Provide a matrix (returned by get_transaction_matrix).")
        return [c for c in m.columns if c.endswith("_R")]

    # ------------------------------ optional class lookup (unchanged) ------------------------------ #

    def load_abx_classes(self):
        if self.load is None:
            raise RuntimeError("LoadClasses is not available in this environment.")
        return self.load.antibiotic_class_list
