from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd

def _aware_palette_present(self, present_values: List[str]) -> Dict[str, str]:
    """
    Keep only categories that actually appear in the nodes.
    """
    pal = self._aware_palette_hex()
    present = set([str(x) for x in present_values if str(x).strip()])
    # drop Not set if not present
    return {k: v for k, v in pal.items() if k in present}


def _split_pipe(s: object) -> List[str]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    s = str(s).strip()
    if not s:
        return []
    return [t.strip() for t in s.split("|") if t and t.strip()]


def _uniq_sorted(items: List[str]) -> List[str]:
    return sorted(set([x for x in items if x]))


def _join_pipe(items: List[str]) -> str:
    return "|".join(_uniq_sorted(items))


def _drug_from_token(tok: str) -> str:
    # e.g., "MER_R" -> "MER"
    return tok.split("_", 1)[0] if "_" in tok else tok


def _suffix_from_token(tok: str) -> str:
    # e.g., "MER_R" -> "R"
    return tok.split("_", 1)[1] if "_" in tok else ""


def _is_antibiotic_event(tok: str) -> bool:
    # "MER_R", "MER_T", "CareType=In-Patient", etc.
    return "_" in tok and tok.split("_", 1)[1] in {"R", "I", "S", "T"}


@dataclass
class ARMRuleCleanerConfig:
    # if True, remove X_T in consequent if X_R appears in antecedent (tautology)
    drop_implied_tests: bool = True

    # If True, also drop any _T tokens that appear in the antecedent before making R-only
    # (usually you already want antecedent_R_only anyway)
    antecedent_keep_only_R: bool = True

    # Always keep only _T tokens in consequence (you requested)
    consequent_keep_only_T: bool = True

    # If True, require the cleaned rule to have both sides non-empty to be "informative"
    require_both_sides_nonempty: bool = True

    # If True, ignore non-antibiotic tokens (like "CareType=In-Patient") in both sides
    # and report what was dropped.
    drop_non_antibiotic_context: bool = True


class ARMRuleCleaner:
    """
    Cleans ARM rules into reviewer-safe "Resistance -> Testing" interpretation.

    Input columns expected:
      - Antecedents
      - Consequents
    Optional:
      - Confidence, Lift, Support (kept for collapse/scoring)

    Produces:
      - Antecedent_R_only
      - Consequent_T_only
      - Dropped_from_antecedent
      - Dropped_from_consequent
      - Dropped_implied_tests
      - Consequent_T_cross_only
      - Is_cross_informative
      - Rule_normalized
      - Rule_cross_normalized
      - Collapsed_count_key
    """

    def __init__(self, config: Optional[ARMRuleCleanerConfig] = None):
        self.cfg = config or ARMRuleCleanerConfig()

    # --------------------------
    # Core per-row parsing logic
    # --------------------------
    def _clean_side(
        self,
        tokens: List[str],
        keep_suffixes: Optional[set] = None,
        drop_non_antibiotic: bool = True,
    ) -> Tuple[List[str], List[str]]:
        """
        Returns (kept_tokens, dropped_tokens)
        """
        kept, dropped = [], []
        for t in tokens:
            if drop_non_antibiotic and (not _is_antibiotic_event(t)):
                dropped.append(t)
                continue
            if keep_suffixes is not None:
                if _suffix_from_token(t) in keep_suffixes:
                    kept.append(t)
                else:
                    dropped.append(t)
            else:
                kept.append(t)
        return _uniq_sorted(kept), _uniq_sorted(dropped)

    def _remove_implied_tests(
        self,
        antecedent_r: List[str],
        consequent_t: List[str],
    ) -> Tuple[List[str], List[str]]:
        """
        Remove X_T from consequence if X_R is in antecedent (implied/tautological tests).
        Returns (cross_only, dropped_implied)
        """
        ant_r_drugs = {_drug_from_token(t) for t in antecedent_r if t.endswith("_R")}
        cross_only, dropped = [], []
        for t in consequent_t:
            if not t.endswith("_T"):
                # Consequent should already be only _T, but just in case:
                continue
            drug = _drug_from_token(t)
            if drug in ant_r_drugs:
                dropped.append(t)
            else:
                cross_only.append(t)
        return _uniq_sorted(cross_only), _uniq_sorted(dropped)

    # --------------------------
    # Public API
    # --------------------------
    def clean_dataframe(
        self,
        df: pd.DataFrame,
        antecedent_col: str = "Antecedents",
        consequent_col: str = "Consequents",
    ) -> pd.DataFrame:
        if antecedent_col not in df.columns or consequent_col not in df.columns:
            raise KeyError(f"Expected columns '{antecedent_col}' and '{consequent_col}' in df.")

        out = df.copy()

        rows = []
        for _, r in out.iterrows():
            ant_tokens = _split_pipe(r[antecedent_col])
            con_tokens = _split_pipe(r[consequent_col])

            # 1) Antecedent: keep only _R (your request), optionally drop non-antibiotic context tokens
            if self.cfg.antecedent_keep_only_R:
                ant_kept, ant_dropped = self._clean_side(
                    ant_tokens,
                    keep_suffixes={"R"},
                    drop_non_antibiotic=self.cfg.drop_non_antibiotic_context,
                )
            else:
                ant_kept, ant_dropped = self._clean_side(
                    ant_tokens,
                    keep_suffixes=None,
                    drop_non_antibiotic=self.cfg.drop_non_antibiotic_context,
                )

            # 2) Consequent: keep only _T (your request)
            if self.cfg.consequent_keep_only_T:
                con_kept, con_dropped = self._clean_side(
                    con_tokens,
                    keep_suffixes={"T"},
                    drop_non_antibiotic=self.cfg.drop_non_antibiotic_context,
                )
            else:
                con_kept, con_dropped = self._clean_side(
                    con_tokens,
                    keep_suffixes=None,
                    drop_non_antibiotic=self.cfg.drop_non_antibiotic_context,
                )

            # 3) Remove implied tests (X_R in antecedent implies X_T)
            dropped_implied = []
            con_cross = con_kept
            if self.cfg.drop_implied_tests:
                con_cross, dropped_implied = self._remove_implied_tests(ant_kept, con_kept)

            # 4) Informative flags and normalized strings
            ant_r_only = _join_pipe(ant_kept)
            con_t_only = _join_pipe(con_kept)
            con_t_cross = _join_pipe(con_cross)

            if self.cfg.require_both_sides_nonempty:
                is_informative = bool(ant_r_only) and bool(con_t_only)
                is_cross_informative = bool(ant_r_only) and bool(con_t_cross)
            else:
                is_informative = bool(ant_r_only) or bool(con_t_only)
                is_cross_informative = bool(ant_r_only) or bool(con_t_cross)

            rule_norm = f"{ant_r_only} -> {con_t_only}" if is_informative else ""
            rule_cross_norm = f"{ant_r_only} -> {con_t_cross}" if is_cross_informative else ""

            rows.append(
                {
                    "Antecedent_R_only": ant_r_only,
                    "Consequent_T_only": con_t_only,
                    "Dropped_from_antecedent": _join_pipe(ant_dropped),
                    "Dropped_from_consequent": _join_pipe(con_dropped),
                    "Dropped_implied_tests": _join_pipe(dropped_implied),
                    "Consequent_T_cross_only": con_t_cross,
                    "Is_informative": is_informative,
                    "Is_cross_informative": is_cross_informative,
                    "Rule_normalized": rule_norm,
                    "Rule_cross_normalized": rule_cross_norm,
                    # Use cross-normalized key for chord plots / networks (best signal)
                    "Collapsed_count_key": rule_cross_norm if rule_cross_norm else rule_norm,
                }
            )

        feats = pd.DataFrame(rows, index=out.index)
        out = pd.concat([out, feats], axis=1)
        return out

    def collapse_rules(
        self,
        df: pd.DataFrame,
        key_col: str = "Rule_cross_normalized",
        support_col: str = "Support",
        lift_col: str = "Lift",
        conf_col: str = "Confidence",
        keep_only_informative: bool = True,
    ) -> pd.DataFrame:
        """
        Collapse duplicates (multiple raw rules mapping to same cleaned rule).
        Keeps max support/lift/confidence; also returns count of collapsed rows.
        """
        if key_col not in df.columns:
            raise KeyError(f"'{key_col}' not found. Run clean_dataframe() first.")

        d = df.copy()
        if keep_only_informative:
            # prefer cross-informative; fall back to informative if key_col is empty
            if "Is_cross_informative" in d.columns:
                d = d[d["Is_cross_informative"] == True].copy()
            else:
                d = d[d[key_col].astype(str).str.len() > 0].copy()

        # Remove empties
        d = d[d[key_col].astype(str).str.strip().str.len() > 0].copy()
        if d.empty:
            return pd.DataFrame(columns=[key_col, "n_rules", "max_support", "max_lift", "max_confidence"])

        agg = (
            d.groupby(key_col, as_index=False)
            .agg(
                n_rules=(key_col, "size"),
                max_support=(support_col, "max") if support_col in d.columns else (key_col, "size"),
                max_lift=(lift_col, "max") if lift_col in d.columns else (key_col, "size"),
                max_confidence=(conf_col, "max") if conf_col in d.columns else (key_col, "size"),
                antecedent=("Antecedent_R_only", "first") if "Antecedent_R_only" in d.columns else (key_col, "first"),
                consequent=("Consequent_T_cross_only", "first") if "Consequent_T_cross_only" in d.columns else (key_col, "first"),
            )
        )

        # rank for reporting (support primary, then lift)
        agg = agg.sort_values(["max_support", "max_lift"], ascending=[False, False]).reset_index(drop=True)
        return agg


# if __name__ == "__main__":
#     # Suppose you loaded your rules CSV:
#     rules = pd.read_csv("./publication_figures/cascade_rules.csv")
#     # columns must include "Antecedents" and "Consequents"

#     cleaner = ARMRuleCleaner(
#         ARMRuleCleanerConfig(
#             drop_implied_tests=True,            # removes PIT_T if PIT_R in antecedent
#             drop_non_antibiotic_context=True,   # drops CareType=... from both sides and records it
#             require_both_sides_nonempty=True
#         )
#     )

#     cleaned = cleaner.clean_dataframe(rules)
#     collapsed = cleaner.collapse_rules(cleaned)
#     cleaned.to_csv("testRules_normalized.csv", index=False)
#     collapsed.to_csv("testRules_collapsed.csv", index=False)
#     # pass
