"""
SensitivityAnalysis.py - PRODUCTION READY
==========================================

Standalone sensitivity analysis runner for the cascade discovery project.

✓ ALL ECLAT REFERENCES REMOVED
✓ APRIORI & FPGROWTH READY
✓ TESTED & PRODUCTION-GRADE

Why a separate file?
- Keeps Run.py focused on the main end-to-end pipeline
- Lets you run long grid/bootstraps independently (e.g., on HPC)
- Reuses the *same* ARMEngine interface (algorithm="pairwise"|"apriori"|"fpgrowth")

What it includes
================
1) Grid sensitivity:
   Mines rules across (support, confidence, lift) settings and compares the
   Top-K edges to a baseline via Jaccard similarity.

2) Bootstrap stability:
   Resamples rows with replacement, re-mines rules, and estimates how often
   each edge appears in the Top-K list.

Outputs
=======
- sensitivity_summary.csv
- sensitivity_top_edges.json
- edge_stability.csv

Usage (Python)
==============
from SensitivityAnalysis import sensitivity_analysis_grid, sensitivity_analysis_bootstrap
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import itertools
import json
import random

import numpy as np
import pandas as pd

try:
    from src.controllers.ARMEngine import ARMEngine
    from src.controllers.DataLoader import DataLoader
except Exception:
    from controllers.ARMEngine import ARMEngine
    from controllers.DataLoader import DataLoader


def set_reproducibility(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def _top_edges(df: pd.DataFrame, k: int) -> List[Tuple[str, str]]:
    if df is None or len(df) == 0:
        return []
    if "Source" not in df.columns or "Target" not in df.columns:
        return []
    dd = df.sort_values(["Lift", "Confidence", "Support"], ascending=False)
    return list(zip(dd["Source"].astype(str), dd["Target"].astype(str)))[:k]



def _jaccard(a: set, b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    return float(len(a & b) / len(a | b))


def sensitivity_analysis_grid(
    matrix: pd.DataFrame,
    output_dir: str | Path,
    *,
    support_grid: List[float],
    confidence_grid: List[float],
    lift_grid: List[float],
    top_k: int = 30,
    seed: int = 42,
    verbose: bool = True,
    algorithm: str = "pairwise",
    max_len: Optional[int] = None,
    edge_mode: str = "pairwise",
) -> pd.DataFrame:
    """
    Grid sensitivity: mine rules across parameter grid and compare Top-K edge overlap
    against a baseline setting via Jaccard similarity.

    Parameters
    ----------
    matrix : pd.DataFrame
        Boolean transaction matrix (rows=episodes, cols=items)
    output_dir : str | Path
        Directory to write sensitivity_summary.csv and sensitivity_top_edges.json
    support_grid : List[float]
        Support thresholds to test
    confidence_grid : List[float]
        Confidence thresholds to test
    lift_grid : List[float]
        Lift thresholds to test
    top_k : int, default=30
        Number of top edges to track
    seed : int, default=42
        Random seed
    verbose : bool, default=True
        Print progress
    algorithm : str, default="pairwise"
        Mining algorithm ("pairwise", "apriori", or "fpgrowth")
    max_len : Optional[int]
        Max itemset size (apriori/fpgrowth only)
    edge_mode : str, default="pairwise"
        Edge extraction mode

    Returns
    -------
    pd.DataFrame
        Grid sensitivity summary with Jaccard similarity scores
    """
    set_reproducibility(seed)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if algorithm not in ["pairwise", "apriori", "fpgrowth"]:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from: pairwise, apriori, fpgrowth")

    settings = list(itertools.product(support_grid, confidence_grid, lift_grid))
    baseline = (
        support_grid[len(support_grid) // 2],
        confidence_grid[len(confidence_grid) // 2],
        lift_grid[len(lift_grid) // 2],
    )

    arm0 = ARMEngine(matrix)
    rules0 = arm0.discover_rules(
        algorithm=algorithm,
        edge_mode=edge_mode,
        min_support=baseline[0],
        min_confidence=baseline[1],
        min_lift=baseline[2],
        max_len=max_len,
        verbose=False,
    )
    base_top = set(_top_edges(rules0, top_k))

    rows = []
    edges_dump: Dict[str, List[Tuple[str, str]]] = {}

    for ms, mc, ml in settings:
        arm = ARMEngine(matrix)
        rules = arm.discover_rules(
            algorithm=algorithm,
            edge_mode=edge_mode,
            min_support=ms,
            min_confidence=mc,
            min_lift=ml,
            max_len=max_len,
            verbose=False,
        )

        te = _top_edges(rules, top_k)
        key = f"ms={ms:.4f}_mc={mc:.2f}_ml={ml:.2f}"
        edges_dump[key] = te

        te_set = set(te)
        jacc = _jaccard(te_set, base_top)

        rows.append(
            {
                "min_support": ms,
                "min_confidence": mc,
                "min_lift": ml,
                "n_rules": int(len(rules)),
                "topk_jaccard_vs_baseline": float(jacc),
                "baseline": (ms, mc, ml) == baseline,
                "algorithm": algorithm,
                "edge_mode": edge_mode,
                "max_len": int(max_len) if max_len is not None else None,
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        ["baseline", "topk_jaccard_vs_baseline"], ascending=[False, True]
    )

    summary.to_csv(outdir / "sensitivity_summary.csv", index=False)
    with open(outdir / "sensitivity_top_edges.json", "w", encoding="utf-8") as f:
        json.dump(edges_dump, f, indent=2)

    if verbose:
        print(f"✓ Grid sensitivity [{algorithm}]: {len(settings)} settings tested")
        print(f"  → {outdir / 'sensitivity_summary.csv'}")

    return summary

def _expand_rules_to_edges_df(rules: pd.DataFrame) -> pd.DataFrame:
    """
    Expand (possibly multi-item) rules into edge-level rules with Source/Target.

    For each rule, generate all R×T edges:
      - Source: antibiotic code from Antecedents items ending with "_R"
      - Target: antibiotic code from Consequents items ending with "_T"
    """
    if rules is None or rules.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    for _, row in rules.iterrows():
        ant = row.get("Antecedents")
        cons = row.get("Consequents")

        # Normalize Antecedents to list of strings
        if isinstance(ant, (set, frozenset)):
            ant_list = list(ant)
        elif isinstance(ant, str):
            ant_list = [
                x.strip().strip("{}")
                for x in ant.replace("{", "").replace("}", "").split(",")
                if x.strip()
            ]
        else:
            ant_list = []

        # Normalize Consequents to list of strings
        if isinstance(cons, (set, frozenset)):
            cons_list = list(cons)
        elif isinstance(cons, str):
            cons_list = [
                x.strip().strip("{}")
                for x in cons.replace("{", "").replace("}", "").split(",")
                if x.strip()
            ]
        else:
            cons_list = []

        # Extract all R and T antibiotics
        sources = [x.replace("_R", "") for x in ant_list if isinstance(x, str) and x.endswith("_R")]
        targets = [x.replace("_T", "") for x in cons_list if isinstance(x, str) and x.endswith("_T")]

        if not sources or not targets:
            continue

        for s in sources:
            for t in targets:
                rows.append(
                    {
                        "Source": s,
                        "Target": t,
                        "Antecedents": row.get("Antecedents"),
                        "Consequents": row.get("Consequents"),
                        "Support": row.get("Support", np.nan),
                        "Confidence": row.get("Confidence", np.nan),
                        "Lift": row.get("Lift", np.nan),
                    }
                )

    return pd.DataFrame(rows)

def sensitivity_analysis_bootstrap(
    matrix: pd.DataFrame,
    output_dir: str | Path,
    *,
    min_support: float = 0.01,
    min_confidence: float = 0.30,
    min_lift: float = 1.10,
    max_len: Optional[int] = None,
    n_bootstrap: int = 200,
    top_k: int = 30,
    seed: int = 42,
    verbose: bool = True,
    algorithm: str = "pairwise",
    edge_mode: str = "pairwise",
) -> pd.DataFrame:
    """
    Bootstrap stability: Resample rows with replacement, re-mine rules, and 
    estimate how often each edge appears in the Top-K list across bootstraps.

    Parameters
    ----------
    matrix : pd.DataFrame
        Boolean transaction matrix
    output_dir : str | Path
        Directory to write edge_stability.csv
    min_support : float, default=0.01
        Support threshold
    min_confidence : float, default=0.30
        Confidence threshold
    min_lift : float, default=1.10
        Lift threshold
    max_len : Optional[int]
        Max itemset size (apriori/fpgrowth only)
    n_bootstrap : int, default=200
        Number of bootstrap resamples
    top_k : int, default=30
        Track top-k edges per resample
    seed : int, default=42
        Random seed
    verbose : bool, default=True
        Print progress
    algorithm : str, default="pairwise"
        Mining algorithm ("pairwise", "apriori", or "fpgrowth")
    edge_mode : str, default="pairwise"
        Edge extraction mode

    Returns
    -------
    pd.DataFrame
        Edge stability summary with frequency and metrics
    """
    print(f"edge mode = {edge_mode}, algorithm = {algorithm}, n_bootstrap = {n_bootstrap}, top_k = {top_k}, min_support = {min_support}, min_confidence = {min_confidence}, min_lift = {min_lift}, max_len = {max_len}")
    set_reproducibility(seed)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if matrix is None or matrix.empty:
        raise ValueError("matrix must be a non-empty transaction matrix")

    if algorithm not in ["pairwise", "apriori", "fpgrowth"]:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from: pairwise, apriori, fpgrowth")

    rng = np.random.default_rng(seed)
    n = matrix.shape[0]

    counts: Dict[Tuple[str, str], int] = {}
    accum: Dict[Tuple[str, str], Dict[str, List[float]]] = {}

    for b in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        boot = matrix.iloc[idx].reset_index(drop=True)

        arm = ARMEngine(boot)
        rules_raw = arm.discover_rules(
            algorithm=algorithm,
            edge_mode=edge_mode,
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
            max_len=max_len,
            verbose=False,
        )

        # Ensure we have edge-level rules with Source/Target
        if "Source" in rules_raw.columns and "Target" in rules_raw.columns:
            rules = rules_raw
        else:
            rules = _expand_rules_to_edges_df(rules_raw)

        if rules is None or rules.empty:
            if verbose:
                print(f"  bootstrap {b + 1}/{n_bootstrap}: no edges, skipping")
            continue

        te = _top_edges(rules, top_k)
        if not te:
            if verbose:
                print(f"  bootstrap {b + 1}/{n_bootstrap}: top_k edges empty, skipping")
            continue

        rule_map = {(str(r["Source"]), str(r["Target"])): r for _, r in rules.iterrows()}

        for (s, t) in te:
            key = (s, t)
            counts[key] = counts.get(key, 0) + 1
            if key not in accum:
                accum[key] = {"Lift": [], "Confidence": [], "Support": []}

            rr = rule_map.get((s, t))
            if rr is None:
                continue

            for field in ["Lift", "Confidence", "Support"]:
                try:
                    v = float(rr[field])
                    if not np.isnan(v):
                        accum[key][field].append(v)
                except (ValueError, TypeError):
                    pass

        if verbose and (b + 1) % 50 == 0:
            print(f"  bootstrap {b + 1}/{n_bootstrap}")

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else np.nan

    def _std(xs: List[float]) -> float:
        return float(np.std(xs, ddof=1)) if len(xs) > 1 else np.nan

    rows = []
    for (s, t), c in counts.items():
        a = accum.get((s, t), {"Lift": [], "Confidence": [], "Support": []})
        rows.append(
            {
                "Source": s,
                "Target": t,
                "freq_in_topk": float(c) / float(n_bootstrap),
                "n_topk": int(c),
                "mean_lift": _mean(a["Lift"]),
                "sd_lift": _std(a["Lift"]),
                "mean_confidence": _mean(a["Confidence"]),
                "sd_confidence": _std(a["Confidence"]),
                "mean_support": _mean(a["Support"]),
                "sd_support": _std(a["Support"]),
                "algorithm": algorithm,
                "edge_mode": edge_mode,
                "max_len": int(max_len) if max_len is not None else None,
                "n_bootstrap": int(n_bootstrap),
                "top_k": int(top_k),
            }
        )

    out = pd.DataFrame(rows).sort_values(["freq_in_topk", "mean_lift"], ascending=False)
    out.to_csv(outdir / "edge_stability.csv", index=False)

    if verbose:
        print(f"✓ Bootstrap stability [{algorithm}]: {n_bootstrap} resamples")
        print(f"  → {outdir / 'edge_stability.csv'}")

    return out


def run_sensitivity_from_data(
    data_path: str,
    output_dir: str,
    *,
    filters: Optional[dict] = None,
    recode_mode: str = "R_vs_nonR",
    include_covariates: bool = False,
    covariate_cols: Optional[List[str]] = None,
    min_test_rate: float = 0.10,
    min_test_count: int = 2000,
    min_res_rate: float = 0.01,
    min_res_count: int = 100,
    max_antibiotics: int = 35,
    always_keep: Optional[List[str]] = None,
    support_grid: Optional[List[float]] = None,
    confidence_grid: Optional[List[float]] = None,
    lift_grid: Optional[List[float]] = None,
    n_bootstrap: int = 200,
    top_k: int = 30,
    seed: int = 42,
    verbose: bool = True,
    algorithm: str = "pairwise",
    max_len: Optional[int] = None,
    edge_mode: str = "pairwise",
) -> Dict[str, pd.DataFrame]:
    """End-to-end sensitivity runner: Load data → build matrix → analyze."""
    set_reproducibility(seed)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(data_path)
    matrix = loader.get_transaction_matrix(
        filters=filters,
        recode_mode=recode_mode,
        verbose=verbose,
        covariate_cols=(covariate_cols if include_covariates else []),
        min_test_rate=min_test_rate,
        min_test_count=min_test_count,
        min_res_rate=min_res_rate,
        min_res_count=min_res_count,
        max_antibiotics=max_antibiotics,
        always_keep=always_keep,
    )

    if support_grid is None:
        support_grid = [0.005, 0.01, 0.02]
    if confidence_grid is None:
        confidence_grid = [0.25, 0.30, 0.40]
    if lift_grid is None:
        lift_grid = [1.05, 1.10, 1.20]

    grid_df = sensitivity_analysis_grid(
        matrix=matrix,
        output_dir=outdir / "grid",
        support_grid=support_grid,
        confidence_grid=confidence_grid,
        lift_grid=lift_grid,
        top_k=top_k,
        seed=seed,
        verbose=verbose,
        algorithm=algorithm,
        max_len=max_len,
        edge_mode=edge_mode,
    )

    boot_df = sensitivity_analysis_bootstrap(
        matrix=matrix,
        output_dir=outdir / "bootstrap",
        min_support=float(support_grid[len(support_grid) // 2]),
        min_confidence=float(confidence_grid[len(confidence_grid) // 2]),
        min_lift=float(lift_grid[len(lift_grid) // 2]),
        max_len=max_len,
        n_bootstrap=n_bootstrap,
        top_k=top_k,
        seed=seed,
        verbose=verbose,
        algorithm=algorithm,
        edge_mode=edge_mode,
    )

    if verbose:
        print(f"\n✓ Sensitivity analysis complete")
        print(f"  Output: {outdir}")

    return {"grid": grid_df, "bootstrap": boot_df}


if __name__ == "__main__":
    print("✓ SensitivityAnalysis module loaded")
    print("\nSUPPORTED ALGORITHMS:")
    print("  • pairwise (default, fast)")
    print("  • apriori (classic, supports max_len)")
    print("  • fpgrowth (efficient, supports max_len)")
    print("\nUSAGE (Python):")
    print("  from SensitivityAnalysis import run_sensitivity_from_data")
    print("  results = run_sensitivity_from_data(")
    print("      data_path='path/to/data.parquet',")
    print("      output_dir='./sensitivity_output',")
    print("      algorithm='apriori',")
    print("      max_len=3,")
    print("  )")
    print("\n✓ VERSION: Production Ready (Eclat Removed)")