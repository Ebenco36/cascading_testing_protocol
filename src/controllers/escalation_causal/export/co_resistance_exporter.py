"""
Escalation to Co-Resistance Exporter
=====================================

Exports results from the escalation causal pipeline to a format suitable
for the Bayesian joint testing‑resistance model. Produces:

- cascade_dependencies.json: target -> list of triggers (for model building)
- escalation_summary.csv: filtered escalation effects with metadata
- Optional: prior parameters for cascade strengths

Author: Your Name
Version: 1.0.0
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class EscalationExport:
    """Container for exported escalation data."""
    dependencies: Dict[str, List[str]]
    summary_df: pd.DataFrame
    effect_dict: Dict[Tuple[str, str], float]  # (trigger, target) -> risk difference
    metadata: dict


class EscalationExporter:
    """
    Export escalation results for use in co-resistance joint modeling.

    Reads the results from the escalation pipeline (typically a CSV with columns
    'trigger', 'target', 'rd', 'p_value', etc.), applies filtering, and produces
    structured outputs that can be loaded by the joint model.
    """

    def __init__(
        self,
        escalation_file: Optional[str] = None,
        escalation_df: Optional[pd.DataFrame] = None,
    ):
        """
        Provide either a file path or a DataFrame with escalation results.
        The DataFrame must contain at least 'trigger', 'target', 'rd' columns.
        """
        if escalation_file is None and escalation_df is None:
            raise ValueError("Must provide either escalation_file or escalation_df")
        if escalation_file is not None:
            self.df = pd.read_csv(escalation_file)
        else:
            self.df = escalation_df.copy()

        # Validate required columns
        required = {"trigger", "target", "rd"}
        if not required.issubset(self.df.columns):
            raise ValueError(f"DataFrame must contain columns: {required}")

        # Optionally, ensure 'p_value' column exists; if not, set to NaN.
        if "p_value" not in self.df.columns:
            self.df["p_value"] = float("nan")

        # Ensure trigger and target are strings
        self.df["trigger"] = self.df["trigger"].astype(str)
        self.df["target"] = self.df["target"].astype(str)

    def filter(
        self,
        p_threshold: Optional[float] = 0.05,
        rd_min: float = 0.0,
        effect_direction: str = "positive",  # "positive", "negative", "both"
        min_ess: Optional[float] = None,
        keep_failed: bool = False,
    ) -> EscalationExporter:
        """
        Apply filters to the escalation results.

        Parameters
        ----------
        p_threshold : float, optional
            Keep only pairs with p_value < p_threshold (if p_value column exists).
            If None, ignore p-value.
        rd_min : float
            Minimum absolute risk difference to keep.
        effect_direction : "positive", "negative", or "both"
            Keep only positive effects (rd > 0), negative (rd < 0), or both.
        min_ess : float, optional
            If 'ess' column exists, keep only pairs with effective sample size >= min_ess.
        keep_failed : bool
            If False, drop rows where status is 'failed' (if 'status' column exists).
        """
        df = self.df.copy()

        # Filter by status
        if not keep_failed and "status" in df.columns:
            df = df[df["status"] == "ok"]

        # Filter by p-value
        if p_threshold is not None and "p_value" in df.columns:
            df = df[df["p_value"] < p_threshold]

        # Filter by effect direction
        if effect_direction == "positive":
            df = df[df["rd"] > 0]
        elif effect_direction == "negative":
            df = df[df["rd"] < 0]
        # else 'both' keep all

        # Filter by absolute RD
        df = df[df["rd"].abs() >= rd_min]

        # Filter by effective sample size
        if min_ess is not None and "ess" in df.columns:
            df = df[df["ess"] >= min_ess]

        # Update internal DataFrame
        self.df = df.reset_index(drop=True)

        # Store filter parameters for metadata
        self._last_p_threshold = p_threshold
        self._last_rd_min = rd_min
        self._last_direction = effect_direction
        return self

    def build_dependencies(self, use_positive: bool = True) -> Dict[str, List[str]]:
        """
        Build cascade dependencies dictionary: target -> list of triggers.
        By default, uses only positive escalation effects (rd > 0).
        If use_positive=False, uses all effects (including negative).
        """
        deps: Dict[str, List[str]] = {}
        df = self.df if use_positive else self.df[self.df["rd"] > 0]
        for _, row in df.iterrows():
            tgt = row["target"]
            trig = row["trigger"]
            if tgt not in deps:
                deps[tgt] = []
            if trig not in deps[tgt]:
                deps[tgt].append(trig)
        return deps

    def effect_size_dict(self) -> Dict[Tuple[str, str], float]:
        """Return dictionary mapping (trigger, target) to risk difference."""
        return {(row["trigger"], row["target"]): row["rd"] for _, row in self.df.iterrows()}

    def summary_statistics(self) -> dict:
        """Compute summary statistics of filtered escalation effects."""
        if self.df.empty:
            return {"n_pairs": 0, "avg_rd": float("nan"), "std_rd": float("nan")}
        return {
            "n_pairs": len(self.df),
            "avg_rd": float(self.df["rd"].mean()),
            "std_rd": float(self.df["rd"].std()),
            "min_rd": float(self.df["rd"].min()),
            "max_rd": float(self.df["rd"].max()),
            "median_rd": float(self.df["rd"].median()),
        }

    def export(
        self,
        output_dir: str,
        prefix: str = "escalation",
        save_summary_csv: bool = True,
        save_dependencies_json: bool = True,
        save_metadata_json: bool = True,
    ) -> EscalationExport:
        """
        Export filtered escalation results to files.

        Returns
        -------
        EscalationExport container with dependencies and DataFrame.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save filtered summary CSV
        summary_path = None
        if save_summary_csv:
            summary_path = os.path.join(output_dir, f"{prefix}_summary.csv")
            self.df.to_csv(summary_path, index=False)

        # Save dependencies JSON
        deps = self.build_dependencies()
        deps_path = None
        if save_dependencies_json:
            deps_path = os.path.join(output_dir, f"{prefix}_dependencies.json")
            with open(deps_path, "w") as f:
                json.dump(deps, f, indent=2)

        # Save metadata
        metadata = {
            "n_pairs": len(self.df),
            "summary_stats": self.summary_statistics(),
            "filter_params": {
                "p_threshold": getattr(self, "_last_p_threshold", None),
                "rd_min": getattr(self, "_last_rd_min", 0.0),
                "effect_direction": getattr(self, "_last_direction", "positive"),
            },
        }
        meta_path = None
        if save_metadata_json:
            meta_path = os.path.join(output_dir, f"{prefix}_metadata.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

        return EscalationExport(
            dependencies=deps,
            summary_df=self.df,
            effect_dict=self.effect_size_dict(),
            metadata={
                "files": {
                    "summary": summary_path,
                    "dependencies": deps_path,
                    "metadata": meta_path,
                },
                "stats": metadata,
            },
        )