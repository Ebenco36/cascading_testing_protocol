"""
CASCADE DISCOVERY & VISUALIZATION PIPELINE
===========================================

Layer 2: ARMEngine
  - Discovers testing cascades of the form: (contains *_R) -> (contains *_T)
  - Supports multiple mining backends:
      * apriori   : mlxtend Apriori + association rule mining
      * fpgrowth  : mlxtend FP-Growth + association rule mining
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Literal

import logging
import numpy as np
import pandas as pd
import networkx as nx

from src.controllers.ARMRuleCleaner import ARMRuleCleaner, ARMRuleCleanerConfig

LOGGER = logging.getLogger(__name__)

Algorithm = Literal["apriori", "fpgrowth"]
EdgeMode = Literal["pairwise", "conditional"]


# ============================================================================
# LAYER 2: ARM ENGINE
# ============================================================================


class ARMEngine:
    """Association Rule Mining engine specialized for AST cascade discovery."""

    def __init__(self, transaction_matrix: pd.DataFrame):
        """
        Parameters
        ----------
        transaction_matrix:
            Boolean/binary indicator matrix (rows = isolates/episodes, cols = items).
            Items should include:
              - drug resistance items:  {DRUG}_R
              - drug tested items:      {DRUG}_T
              - optional covariate items: e.g. WardType=ICU, CareType=Inpatient, ...
        """
        if not isinstance(transaction_matrix, pd.DataFrame):
            raise TypeError("transaction_matrix must be a pandas DataFrame")

        self.matrix = self._validate_transaction_matrix(transaction_matrix)

        # Fast access structures
        self._cols: List[str] = list(self.matrix.columns)
        self._col_index: Dict[str, int] = {c: i for i, c in enumerate(self._cols)}
        self._X = self.matrix.to_numpy(dtype=bool, copy=False)  # shape: (n, p)
        self._n = int(self._X.shape[0])

        # Cache for support computations
        self._support_cache: Dict[Tuple[str, ...], float] = {}

        # Discovered (expanded) cascade edges
        self.rules: Optional[pd.DataFrame] = None

    # ----------------------------- validation ----------------------------- #

    def _validate_transaction_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that df is a proper binary/boolean transaction matrix."""
        if df.empty:
            raise ValueError("transaction_matrix is empty")

        if df.columns.isnull().any():
            raise ValueError("transaction_matrix has null column names")

        if df.columns.duplicated().any():
            dups = df.columns[df.columns.duplicated()].tolist()
            raise ValueError(f"transaction_matrix has duplicated columns: {dups[:10]}")

        if df.isna().any().any():
            nan_cols = df.columns[df.isna().any()].tolist()
            raise ValueError(f"transaction_matrix contains NaNs in columns: {nan_cols[:10]}")

        out = df.astype(bool)

        # Require at least one resistance and one tested column
        has_r = any(str(c).endswith("_R") for c in out.columns)
        has_t = any(str(c).endswith("_T") for c in out.columns)
        if not has_r or not has_t:
            raise ValueError(
                "transaction_matrix must contain at least one *_R column and one *_T column."
            )

        # Remove all-zero columns
        zero_cols = out.columns[(out.sum(axis=0) == 0).to_numpy()].tolist()
        if zero_cols:
            LOGGER.warning("Dropping %d all-zero columns from transaction matrix", len(zero_cols))
            out = out.drop(columns=zero_cols)

        if out.shape[1] == 0:
            raise ValueError("transaction_matrix has no non-zero columns after cleanup")

        return out

    # ------------------------------ metrics ------------------------------ #

    def _support(self, items: Iterable[str]) -> float:
        """Support of an itemset: P(all items present)."""
        key = tuple(sorted(map(str, items)))
        if not key:
            return 1.0
        cached = self._support_cache.get(key)
        if cached is not None:
            return cached

        idx = []
        for it in key:
            j = self._col_index.get(it)
            if j is None:
                self._support_cache[key] = 0.0
                return 0.0
            idx.append(j)

        M = self._X[:, idx]
        sup = float(np.mean(np.all(M, axis=1))) if self._n else 0.0
        self._support_cache[key] = sup
        return sup

    def _edge_metrics(self, antecedent_items: List[str], consequent_item: str) -> Tuple[float, float, float]:
        """Compute (support, confidence, lift) for antecedent -> {consequent_item}."""
        ante_sup = self._support(antecedent_items)
        if ante_sup <= 0.0:
            return 0.0, 0.0, 0.0
        cons_sup = self._support([consequent_item])
        both_sup = self._support(list(antecedent_items) + [consequent_item])
        conf = both_sup / ante_sup if ante_sup else 0.0
        lift = (conf / cons_sup) if cons_sup else 0.0
        return both_sup, conf, lift

    # ------------------------------ public ------------------------------- #

    def discover_rules(
        self,
        *,
        algorithm: Algorithm = "fpgrowth",
        edge_mode: EdgeMode = "pairwise",
        min_support: float = 0.01,
        min_confidence: float = 0.30,
        min_lift: float = 1.10,
        max_len: Optional[int] = 4,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Discover cascade edges.

        Parameters
        ----------
        algorithm:
            "apriori" or "fpgrowth" (mlxtend backends)
        edge_mode:
            "pairwise" : metrics computed for [r] -> t  (recommended for cascade edges)
            "conditional": metrics computed for ([r] + context_items) -> t
        min_support, min_confidence, min_lift:
            Standard ARM thresholds.
        max_len:
            Maximum itemset length (for apriori/fpgrowth)
        verbose:
            Print progress information
        """

        if algorithm not in ("apriori", "fpgrowth"):
            raise ValueError("algorithm must be 'apriori' or 'fpgrowth'")

        if edge_mode not in ("pairwise", "conditional"):
            raise ValueError("edge_mode must be 'pairwise' or 'conditional'")

        if not (0 < min_support <= 1):
            raise ValueError("min_support must be in (0, 1]")
        if not (0 < min_confidence <= 1):
            raise ValueError("min_confidence must be in (0, 1]")
        if min_lift <= 0:
            raise ValueError("min_lift must be > 0")

        if verbose:
            print(f"\n{'='*70}")
            print("CASCADE RULE DISCOVERY")
            print(f"{'='*70}")
            print(f"Algorithm:      {algorithm}")
            print(f"Edge Mode:      {edge_mode}")
            print(
                f"Thresholds:     min_support={min_support}, min_confidence={min_confidence}, "
                f"min_lift={min_lift}, max_len={max_len}\n"
            )

        if algorithm == "fpgrowth":
            rules = self._discover_via_mlxtend_fpgrowth(
                edge_mode=edge_mode,
                min_support=min_support,
                min_confidence=min_confidence,
                min_lift=min_lift,
                max_len=max_len,
                verbose=verbose,
            )
        elif algorithm == "apriori":
            rules = self._discover_via_mlxtend_apriori(
                edge_mode=edge_mode,
                min_support=min_support,
                min_confidence=min_confidence,
                min_lift=min_lift,
                max_len=max_len,
                verbose=verbose,
            )

        self.rules = rules
        return rules

    # ------------------------------------------------------------------ #
    # FP-Growth (mlxtend)
    # ------------------------------------------------------------------ #

    def _discover_via_mlxtend_fpgrowth(
        self,
        *,
        edge_mode: EdgeMode,
        min_support: float,
        min_confidence: float,
        min_lift: float,
        max_len: Optional[int],
        verbose: bool,
    ) -> pd.DataFrame:
        try:
            from mlxtend.frequent_patterns import fpgrowth, association_rules
        except ImportError as e:
            raise ImportError("Install mlxtend for fpgrowth backend: pip install mlxtend") from e

        if verbose:
            print("Step 1: Mining frequent itemsets (FP-Growth)...")
        frequent_itemsets = fpgrowth(
            self.matrix,
            min_support=min_support,
            use_colnames=True,
            max_len=max_len,
        )
        if verbose:
            print(f"  ✓ Found {len(frequent_itemsets)} frequent itemsets")

        if len(frequent_itemsets) == 0:
            return self._empty_rules_frame()

        if verbose:
            print("Step 2: Generating association rules...")
        all_rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence,
        )
        if verbose:
            print(f"  ✓ Generated {len(all_rules)} candidate rules")

        if len(all_rules) == 0:
            return self._empty_rules_frame()

        if verbose:
            print("Step 3: Filtering to cascade pattern (contains *_R → contains *_T) and expanding...")
        cascade_rules = self._filter_cascade_pattern(all_rules, edge_mode=edge_mode)
        if verbose:
            print(f"  ✓ Cascade edges (expanded): {len(cascade_rules)}")

        if len(cascade_rules) == 0:
            return self._empty_rules_frame()

        # Lift filter
        cascade_rules = cascade_rules[cascade_rules["Lift"] >= float(min_lift)].copy()
        if verbose:
            print(f"Step 4: Applying lift >= {min_lift} ... ✓ {len(cascade_rules)} remain")

        if len(cascade_rules) == 0:
            return self._empty_rules_frame()

        cascade_rules = self._metric_sanity_checks(cascade_rules)

        out_cols = [
            "Antecedents",
            "Consequents",
            "Support",
            "Confidence",
            "Lift",
        ]
        return cascade_rules[out_cols].sort_values(
            ["Lift", "Confidence", "Support"], ascending=False
        ).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Apriori (mlxtend)
    # ------------------------------------------------------------------ #

    def _discover_via_mlxtend_apriori(
        self,
        *,
        edge_mode: EdgeMode,
        min_support: float,
        min_confidence: float,
        min_lift: float,
        max_len: Optional[int],
        verbose: bool,
    ) -> pd.DataFrame:
        try:
            from mlxtend.frequent_patterns import apriori, association_rules
        except ImportError as e:
            raise ImportError("Install mlxtend for apriori backend: pip install mlxtend") from e

        if verbose:
            print("Step 1: Mining frequent itemsets (Apriori)...")
        frequent_itemsets = apriori(
            self.matrix,
            min_support=min_support,
            use_colnames=True,
            max_len=max_len,
        )
        if verbose:
            print(f"  ✓ Found {len(frequent_itemsets)} frequent itemsets")

        if len(frequent_itemsets) == 0:
            return self._empty_rules_frame()

        if verbose:
            print("Step 2: Generating association rules...")
        all_rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence,
        )
        if verbose:
            print(f"  ✓ Generated {len(all_rules)} candidate rules")

        if len(all_rules) == 0:
            return self._empty_rules_frame()

        if verbose:
            print("Step 3: Filtering to cascade pattern (contains *_R → contains *_T) and expanding...")
        cascade_rules = self._filter_cascade_pattern(all_rules, edge_mode=edge_mode)
        if verbose:
            print(f"  ✓ Cascade edges (expanded): {len(cascade_rules)}")

        if len(cascade_rules) == 0:
            return self._empty_rules_frame()

        # Lift filter
        cascade_rules = cascade_rules[cascade_rules["Lift"] >= float(min_lift)].copy()
        if verbose:
            print(f"Step 4: Applying lift >= {min_lift} ... ✓ {len(cascade_rules)} remain")

        if len(cascade_rules) == 0:
            return self._empty_rules_frame()

        cascade_rules = self._metric_sanity_checks(cascade_rules)

        out_cols = [
            "Antecedents",
            "Consequents",
            "Support",
            "Confidence",
            "Lift",
        ]
        return cascade_rules[out_cols].sort_values(
            ["Lift", "Confidence", "Support"], ascending=False
        ).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Common helpers
    # ------------------------------------------------------------------ #

    def _empty_rules_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "Antecedents",
                "Consequents",
                "Support",
                "Confidence",
                "Lift",
            ]
        )

    def _metric_sanity_checks(self, df: pd.DataFrame) -> pd.DataFrame:
        bad_conf = (~df["Confidence"].between(0, 1)).sum()
        bad_sup = (~df["Support"].between(0, 1)).sum()
        bad_lift = (df["Lift"] < 0).sum()
        if bad_conf or bad_sup or bad_lift:
            LOGGER.warning(
                "Metric sanity: clamping values (bad_conf=%d, bad_support=%d, bad_lift=%d)",
                int(bad_conf), int(bad_sup), int(bad_lift),
            )
        df["Confidence"] = df["Confidence"].clip(0, 1)
        df["Support"] = df["Support"].clip(0, 1)
        df["Lift"] = df["Lift"].clip(lower=0)
        return df

    def _filter_cascade_pattern(self, rules: pd.DataFrame, *, edge_mode: EdgeMode) -> pd.DataFrame:
        """
        Filter rules to cascade pattern and expand into pairwise edges, recomputing metrics.

        For each mlxtend rule:
          antecedents: items in antecedent (may include covariates and *_R)
          consequents: items in consequent (may include multiple items)

        We keep rules where:
          antecedent contains >=1 *_R
          consequent contains >=1 *_T

        We then expand to every (r_item, t_item) pair and recompute:
          - Support/Confidence/Lift for either:
              [r] -> t                    (edge_mode="pairwise")
              ([r] + ctx_items) -> t      (edge_mode="conditional")
        """
        cascade_rows: List[dict] = []

        for _, row in rules.iterrows():
            antecedents = sorted(map(str, list(row["antecedents"])))
            consequents = sorted(map(str, list(row["consequents"])))

            r_items = [a for a in antecedents if a.endswith("_R")]
            t_items = [c for c in consequents if c.endswith("_T")]
            if not r_items or not t_items:
                continue

            # Context items (covariates, excluding *_R and *_T)
            ctx_items = [
                a for a in antecedents
                if (not a.endswith("_R")) and (not a.endswith("_T"))
            ]

            for r in r_items:
                for t in t_items:
                    if edge_mode == "pairwise":
                        ante_for_metrics = [r]
                    else:
                        ante_for_metrics = [r] + ctx_items

                    edge_support, edge_conf, edge_lift = self._edge_metrics(ante_for_metrics, t)

                    cascade_rows.append(
                        {
                            "Antecedents": "|".join(antecedents),
                            "Consequents": "|".join(consequents),
                            "Confidence": float(edge_conf),
                            "Lift": float(edge_lift),
                            "Support": float(edge_support),
                        }
                    )

        return pd.DataFrame(cascade_rows)

    def save_rules(self, filepath: str) -> None:
        """Save discovered rules to CSV (creates parent directory if needed)."""
        if self.rules is None:
            raise ValueError("No rules discovered yet. Run discover_rules() first.")
        path = Path(filepath)
        if path.parent and str(path.parent) not in (".", ""):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.rules.to_csv(path, index=False)
        print(f"✓ Rules saved to: {path}")


# ============================================================================
# LAYER 3: CASCADE ANALYSIS
# ============================================================================


class CascadeAnalyzer:
    """Analyze cascade patterns and compute network metrics."""

    def __init__(
        self,
        rules: pd.DataFrame,
        *,
        edge_weight: str = "Lift",
        aggregate: str = "max",
    ):
        self.rules = rules if rules is not None else pd.DataFrame()
        self.edge_weight = edge_weight
        self.aggregate = aggregate

        if len(self.rules) == 0:
            self.graph = nx.DiGraph()
        else:
            self.graph = self._build_graph()

    def _build_graph(self) -> nx.DiGraph:
        """Build directed graph from Antecedents -> Consequents."""
        G = nx.DiGraph()

        for _, r in self.rules.iterrows():
            # Parse antecedents and consequents (pipe-separated strings)
            antecedents_str = str(r.get("Antecedents", ""))
            consequents_str = str(r.get("Consequents", ""))

            ante_items = [x.strip() for x in antecedents_str.split("|") if x.strip()]
            cons_items = [x.strip() for x in consequents_str.split("|") if x.strip()]

            if not ante_items or not cons_items:
                continue

            # Extract *_R items from antecedents
            r_items = [a for a in ante_items if a.endswith("_R")]
            # Extract *_T items from consequents
            t_items = [c for c in cons_items if c.endswith("_T")]

            if not r_items or not t_items:
                continue

            # For each (r, t) pair, create an edge Source -> Target
            for r_item in r_items:
                for t_item in t_items:
                    src = r_item[:-2]  # strip _R
                    tgt = t_item[:-2]  # strip _T

                    if src == tgt:
                        continue

                    w = float(r.get(self.edge_weight, 0.0))
                    lift = float(r.get("Lift", np.nan))
                    conf = float(r.get("Confidence", np.nan))
                    sup = float(r.get("Support", np.nan))

                    if G.has_edge(src, tgt):
                        if self.aggregate == "sum":
                            G[src][tgt]["Support_sum"] = float(
                                G[src][tgt].get("Support_sum", 0.0) + sup
                            )
                            G[src][tgt]["Lift"] = max(float(G[src][tgt].get("Lift", 0.0)), lift)
                            G[src][tgt]["Confidence"] = max(
                                float(G[src][tgt].get("Confidence", 0.0)), conf
                            )
                            G[src][tgt]["weight"] = max(float(G[src][tgt].get("weight", 0.0)), w)
                        else:
                            if w > float(G[src][tgt].get("weight", -np.inf)):
                                G[src][tgt].update(
                                    weight=w,
                                    Lift=lift,
                                    Confidence=conf,
                                    Support=sup,
                                    Support_sum=float(G[src][tgt].get("Support_sum", sup)),
                                )
                    else:
                        G.add_edge(
                            src,
                            tgt,
                            weight=w,
                            Lift=lift,
                            Confidence=conf,
                            Support=sup,
                            Support_sum=sup,
                        )

        return G

    def compute_network_stats(self) -> dict:
        """Compute simple network statistics."""
        G = self.graph
        n = G.number_of_nodes()
        m = G.number_of_edges()
        density = nx.density(G) if n > 1 else 0.0

        unique_drugs = 0
        mean_lift = np.nan
        mean_conf = np.nan
        if len(self.rules):
            unique_drugs = len(self._get_all_drugs())
            mean_lift = float(np.nanmean(self.rules["Lift"]))
            mean_conf = float(np.nanmean(self.rules["Confidence"]))

        return {
            "nodes": int(n),
            "edges": int(m),
            "network_density": float(density),
            "total_rules": int(len(self.rules)),
            "unique_drugs": unique_drugs,
            "mean_lift": mean_lift,
            "mean_confidence": mean_conf,
            "edge_weight": self.edge_weight,
            "aggregate": self.aggregate,
        }

    def _get_all_drugs(self) -> set:
        """Extract all unique drug names from rules."""
        drugs = set()
        for _, r in self.rules.iterrows():
            ante_str = str(r.get("Antecedents", ""))
            cons_str = str(r.get("Consequents", ""))

            for item in ante_str.split("|"):
                item = item.strip()
                if item.endswith("_R"):
                    drugs.add(item[:-2])

            for item in cons_str.split("|"):
                item = item.strip()
                if item.endswith("_T"):
                    drugs.add(item[:-2])

        return drugs

    def top_gatekeepers(self, k: int = 10, *, weighted: bool = False) -> List[Tuple[str, float]]:
        if not weighted:
            out_deg = sorted(self.graph.out_degree(), key=lambda x: x[1], reverse=True)
            return [(n, float(d)) for n, d in out_deg[:k]]

        out_w = sorted(
            (
                (
                    n,
                    float(
                        sum(d.get("weight", 0.0) for _, _, d in self.graph.out_edges(n, data=True))
                    ),
                )
                for n in self.graph.nodes()
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        return out_w[:k]

    def top_reserves(self, k: int = 10, *, weighted: bool = False) -> List[Tuple[str, float]]:
        if not weighted:
            in_deg = sorted(self.graph.in_degree(), key=lambda x: x[1], reverse=True)
            return [(n, float(d)) for n, d in in_deg[:k]]

        in_w = sorted(
            (
                (
                    n,
                    float(
                        sum(d.get("weight", 0.0) for _, _, d in self.graph.in_edges(n, data=True))
                    ),
                )
                for n in self.graph.nodes()
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        return in_w[:k]

    def centrality(self) -> Dict[str, float]:
        """Weighted PageRank as a robust global importance score."""
        if self.graph.number_of_nodes() == 0:
            return {}
        return nx.pagerank(self.graph, weight="weight")

    def print_summary(self, k: int = 10) -> None:
        stats = self.compute_network_stats()
        gatekeepers = self.top_gatekeepers(k=k, weighted=False)
        reserves = self.top_reserves(k=k, weighted=False)

        print(f"\n{'='*70}")
        print("CASCADE NETWORK SUMMARY")
        print(f"{'='*70}")
        print(f"Rules Discovered:        {stats['total_rules']}")
        print(f"Unique Drugs:            {stats['unique_drugs']}")
        print(
            f"Mean Lift:               {stats['mean_lift']:.2f}"
            if not np.isnan(stats["mean_lift"])
            else "Mean Lift:               NA"
        )
        print(
            f"Mean Confidence:         {stats['mean_confidence']:.2f}"
            if not np.isnan(stats["mean_confidence"])
            else "Mean Confidence:         NA"
        )
        print(f"Network Density:         {stats['network_density']:.3f}\n")

        print("Top Gatekeeper Drugs (by out-degree):")
        for drug, degree in gatekeepers:
            print(f"  • {drug}: triggers {int(degree)} downstream tests")

        print("\nTop Reserve Drugs (by in-degree):")
        for drug, degree in reserves:
            print(f"  • {drug}: tested after {int(degree)} upstream failures")

        print("")



# ============================================================================
# LAYER 4: ARM RULE CLEANER INTEGRATION
# ============================================================================

def clean_and_collapse_rules(
    rules_filepath: str,
    output_normalized_filepath: str = None,
    output_collapsed_filepath: str = None,
    config: Optional[ARMRuleCleanerConfig] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load saved rules CSV, clean, and collapse them.

    Parameters
    ----------
    rules_filepath : str
        Path to saved rules CSV (from ARMEngine.save_rules())
    output_normalized_filepath : str, optional
        Path to save cleaned rules. If None, uses input_filepath + "_normalized.csv"
    output_collapsed_filepath : str, optional
        Path to save collapsed rules. If None, uses input_filepath + "_collapsed.csv"
    config : ARMRuleCleanerConfig, optional
        Cleaning configuration. Defaults to sensible ARM cascade defaults.
    verbose : bool
        Print progress information

    Returns
    -------
    cleaned_df : pd.DataFrame
        Cleaned rules with all intermediate columns
    collapsed_df : pd.DataFrame
        Collapsed/deduplicated rules with aggregated metrics
    """

    # Load rules
    if verbose:
        print(f"\n{'='*70}")
        print("RULE CLEANING & COLLAPSING")
        print(f"{'='*70}")
        print(f"Loading rules from: {rules_filepath}")

    rules = pd.read_csv(rules_filepath)

    if verbose:
        print(f"  ✓ Loaded {len(rules)} rules")

    # Set defaults for filepaths
    if output_normalized_filepath is None:
        base = Path(rules_filepath).stem
        parent = Path(rules_filepath).parent
        output_normalized_filepath = parent / f"{base}_normalized.csv"

    if output_collapsed_filepath is None:
        base = Path(rules_filepath).stem
        parent = Path(rules_filepath).parent
        output_collapsed_filepath = parent / f"{base}_collapsed.csv"

    # Initialize cleaner with config
    if config is None:
        config = ARMRuleCleanerConfig(
            drop_implied_tests=True,
            antecedent_keep_only_R=True,
            consequent_keep_only_T=True,
            require_both_sides_nonempty=True,
            drop_non_antibiotic_context=True,
        )

    cleaner = ARMRuleCleaner(config)

    # Clean
    if verbose:
        print("\nStep 1: Cleaning rules...")
    cleaned = cleaner.clean_dataframe(rules)

    if verbose:
        n_inform = (cleaned["Is_informative"] == True).sum()
        n_cross = (cleaned["Is_cross_informative"] == True).sum()
        print(f"  ✓ {n_inform} informative rules")
        print(f"  ✓ {n_cross} cross-informative rules")

    # Save cleaned
    cleaned.to_csv(output_normalized_filepath, index=False)
    if verbose:
        print(f"  ✓ Saved cleaned rules to: {output_normalized_filepath}")

    # Collapse
    if verbose:
        print("\nStep 2: Collapsing duplicate rules...")
    collapsed = cleaner.collapse_rules(cleaned, keep_only_informative=True)

    if verbose:
        print(f"  ✓ Collapsed to {len(collapsed)} unique rules")
        if len(collapsed) > 0:
            avg_collapse = collapsed["n_rules"].mean()
            print(f"  ✓ Average rules per collapsed group: {avg_collapse:.1f}")

    # Save collapsed
    collapsed.to_csv(output_collapsed_filepath, index=False)
    if verbose:
        print(f"  ✓ Saved collapsed rules to: {output_collapsed_filepath}")

    return cleaned, collapsed


# Add method to ARMEngine for convenience
def ARMEngine_clean_and_collapse(
    self,
    output_normalized_filepath: str = None,
    output_collapsed_filepath: str = None,
    config: Optional[ARMRuleCleanerConfig] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean and collapse the discovered rules in-place.

    Convenience method for ARMEngine; calls clean_and_collapse_rules on self.rules.
    """
    if self.rules is None:
        raise ValueError("No rules to clean. Run discover_rules() first.")

    # Use default output names based on a temporary save location
    if output_normalized_filepath is None:
        output_normalized_filepath = "rules_normalized.csv"
    if output_collapsed_filepath is None:
        output_collapsed_filepath = "rules_collapsed.csv"

    # Save rules temporarily
    temp_path = "temp_rules.csv"
    self.save_rules(temp_path)

    # Clean and collapse
    cleaned, collapsed = clean_and_collapse_rules(
        temp_path,
        output_normalized_filepath=output_normalized_filepath,
        output_collapsed_filepath=output_collapsed_filepath,
        config=config,
        verbose=verbose,
    )

    # Clean up temp file
    Path(temp_path).unlink(missing_ok=True)

    return cleaned, collapsed
