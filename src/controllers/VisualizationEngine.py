"""
VisualizationEngine.py - PUBLICATION-GRADE SENSITIVITY & BOOTSTRAP VISUALIZATION

High-Impact Journal-Ready Visualization System
Designed for Nature Methods, Nature Microbiology, Lancet Infectious Diseases standards

Features:
  ✓ 300-600 DPI publication quality
  ✓ CMYK-compatible color schemes
  ✓ Accessible color palettes (colorblind-friendly)
  ✓ Professional typography (Arial/Helvetica)
  ✓ Consistent visual hierarchy
  ✓ High information density
  ✓ Clear statistical annotations
  ✓ Publication-ready legends and labels

Generates:
  1. Grid Sensitivity Heatmap (parameter impact analysis)
  2. Bootstrap Stability Analysis (edge robustness)
  3. Waterfall Plot (ranked stability scores)
  4. Combined Multi-Panel Figure (main text figure)

SAVE AS: src/controllers/VisualizationEngine.py
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import seaborn as sns

LOGGER = logging.getLogger(__name__)

# ========================= PUBLICATION CONSTANTS =============================

# High-impact journal color schemes (colorblind-safe)
PUBLICATION_COLORS = {
    "primary": "#0077BB",      # Strong blue
    "secondary": "#CC3311",    # Strong red
    "tertiary": "#009988",     # Teal
    "quaternary": "#EE7733",   # Orange
    "accent": "#EE3377",       # Magenta
    "neutral": "#BBBBBB",      # Gray
    "background": "#FFFFFF",   # White
    "grid": "#E0E0E0",        # Light gray
}

# Colorblind-friendly sequential palettes
COLORBLIND_PALETTES = {
    "blue_sequential": ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"],
    "orange_sequential": ["#fff5eb", "#fee6ce", "#fdd0a2", "#fdae6b", "#fd8d3c", "#f16913", "#d94801", "#a63603", "#7f2704"],
    "green_sequential": ["#f7fcf5", "#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476", "#41ab5d", "#238b45", "#006d2c", "#00441b"],
    "diverging": ["#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#f7f7f7", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061"],
    "categorical": ["#0077BB", "#CC3311", "#009988", "#EE7733", "#33BBEE", "#EE3377", "#BBBBBB"],
}

# Typography (Nature/Science standards)
PUBLICATION_FONTS = {
    "primary": "Arial",
    "fallback": "Helvetica",
    "monospace": "Courier New",
}

# Figure dimensions (matching Nature/Science guidelines)
FIGURE_SIZES = {
    "single_column": (3.5, 3),       # 89 mm width
    "1.5_column": (5.5, 4),          # 140 mm width
    "double_column": (7, 5),         # 178 mm width
    "full_page": (7, 9),             # 178 mm × 229 mm
}

# DPI standards
DPI_SETTINGS = {
    "draft": 150,
    "submission": 300,
    "print": 600,
}


# ================================ UTILITIES ==================================

def _ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _scale_array(arr: np.ndarray, vmin: float = 0, vmax: float = 1) -> np.ndarray:
    """Min-max normalize array to [vmin, vmax]."""
    if arr.size == 0:
        return arr
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max == arr_min:
        return np.full_like(arr, (vmin + vmax) / 2, dtype=float)
    return vmin + (arr - arr_min) / (arr_max - arr_min) * (vmax - vmin)


def _get_colorblind_cmap(name: str = "blue_sequential") -> LinearSegmentedColormap:
    """Get colorblind-friendly colormap."""
    colors = COLORBLIND_PALETTES.get(name, COLORBLIND_PALETTES["blue_sequential"])
    return LinearSegmentedColormap.from_list(name, colors)


def _set_publication_style():
    """Set matplotlib to publication-ready style."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [PUBLICATION_FONTS["primary"], PUBLICATION_FONTS["fallback"]],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "legend.title_fontsize": 8,
        "figure.titlesize": 11,
        "figure.titleweight": "bold",
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "patch.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,  # TrueType fonts for PDF
        "ps.fonttype": 42,   # TrueType fonts for PS
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


# ============================== BASE VISUALIZER ==============================

class BaseVisualizer(ABC):
    """Abstract base class for publication-grade visualizers."""

    def __init__(
        self,
        output_dir: str | Path = "./publication_figures",
        seed: int = 42,
        dpi: int = 300,
        verbose: bool = True,
    ):
        self.output_dir = _ensure_dir(output_dir)
        self.seed = seed
        self.dpi = dpi
        self.verbose = verbose

        np.random.seed(seed)
        _set_publication_style()

    @abstractmethod
    def plot(self, *args, **kwargs) -> plt.Figure:
        """Generate the visualization."""
        pass

    def save_figure(self, fig: plt.Figure, filename: str, dpi: Optional[int] = None) -> Path:
        """Save figure in publication-ready formats."""
        if dpi is None:
            dpi = self.dpi

        filepath_png = self.output_dir / filename
        filepath_pdf = self.output_dir / filename.replace(".png", ".pdf")
        filepath_svg = self.output_dir / filename.replace(".png", ".svg")

        # Save PNG (for quick preview)
        fig.savefig(filepath_png, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")

        # Save PDF (for publication)
        fig.savefig(filepath_pdf, format="pdf", bbox_inches="tight")

        # Save SVG (for editing in Illustrator)
        fig.savefig(filepath_svg, format="svg", bbox_inches="tight")

        if self.verbose:
            print(f"  ✓ Saved: {filepath_png.name} (PNG, PDF, SVG)")

        return filepath_png


# ==================== 1. GRID SENSITIVITY VISUALIZER ==========================

class GridSensitivityVisualizer(BaseVisualizer):
    """Publication-grade parameter sensitivity heatmaps."""

    def plot(
        self,
        grid_summary_df: pd.DataFrame,
        figsize: Tuple[float, float] = FIGURE_SIZES["double_column"],
        filename: Optional[str] = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Generate publication-quality grid sensitivity heatmap.

        Parameters
        ----------
        grid_summary_df : pd.DataFrame
            Columns: [support, confidence, lift, n_rules, avg_lift, avg_confidence]
        figsize : tuple
            Figure dimensions in inches
        filename : str
            Output filename

        Returns
        -------
        plt.Figure
        """
        if grid_summary_df.empty:
            raise ValueError("grid_summary_df is empty")

        # Validate required columns
        required = ["support", "n_rules"]
        missing = [c for c in required if c not in grid_summary_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Check if we have 2D grid (support × confidence) or 1D (support only)
        has_confidence = "confidence" in grid_summary_df.columns and grid_summary_df["confidence"].nunique() > 1

        if has_confidence:
            return self._plot_2d_heatmap(grid_summary_df, figsize, filename)
        else:
            return self._plot_1d_lineplot(grid_summary_df, figsize, filename)

    def _plot_2d_heatmap(
        self,
        df: pd.DataFrame,
        figsize: Tuple[float, float],
        filename: Optional[str],
    ) -> plt.Figure:
        """Generate 2D heatmap for support × confidence grid."""
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=self.dpi)

        # Pivot 1: Number of rules
        pivot_n_rules = df.pivot_table(
            index="support",
            columns="confidence",
            values="n_rules",
            aggfunc="first",
        )

        # Pivot 2: Average lift
        pivot_avg_lift = df.pivot_table(
            index="support",
            columns="confidence",
            values="avg_lift" if "avg_lift" in df.columns else "n_rules",
            aggfunc="first",
        )

        # Heatmap 1: Rule count
        cmap1 = _get_colorblind_cmap("blue_sequential")
        sns.heatmap(
            pivot_n_rules,
            annot=True,
            fmt="d",
            cmap=cmap1,
            ax=axes[0],
            cbar_kws={"label": "Rule Count", "shrink": 0.8},
            linewidths=0.5,
            linecolor="white",
            vmin=0,
        )
        axes[0].set_title("A. Discovery Rate", fontweight="bold", pad=10)
        axes[0].set_xlabel("Confidence Threshold", fontweight="bold")
        axes[0].set_ylabel("Support Threshold", fontweight="bold")

        # Heatmap 2: Average lift
        cmap2 = _get_colorblind_cmap("orange_sequential")
        sns.heatmap(
            pivot_avg_lift,
            annot=True,
            fmt=".2f",
            cmap=cmap2,
            ax=axes[1],
            cbar_kws={"label": "Average Lift", "shrink": 0.8},
            linewidths=0.5,
            linecolor="white",
        )
        axes[1].set_title("B. Average Effect Size", fontweight="bold", pad=10)
        axes[1].set_xlabel("Confidence Threshold", fontweight="bold")
        axes[1].set_ylabel("Support Threshold", fontweight="bold")

        # Add figure title
        fig.suptitle(
            "Parameter Sensitivity Analysis: Impact on Cascade Discovery",
            fontsize=11,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout()

        if filename:
            self.save_figure(fig, filename)

        return fig

    def _plot_1d_lineplot(
        self,
        df: pd.DataFrame,
        figsize: Tuple[float, float],
        filename: Optional[str],
    ) -> plt.Figure:
        """Generate 1D line plot for single-parameter sensitivity."""
        fig, ax1 = plt.subplots(figsize=figsize, dpi=self.dpi)

        df_sorted = df.sort_values("support")

        # Primary axis: Rule count
        color1 = PUBLICATION_COLORS["primary"]
        ax1.plot(
            df_sorted["support"],
            df_sorted["n_rules"],
            marker="o",
            linewidth=2.5,
            markersize=7,
            color=color1,
            markerfacecolor=color1,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="Rule Count",
            zorder=3,
        )
        ax1.set_xlabel("Support Threshold", fontweight="bold", fontsize=10)
        ax1.set_ylabel("Number of Discovered Rules", fontweight="bold", fontsize=10, color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.grid(True, alpha=0.3, linewidth=0.5, zorder=1)
        ax1.set_axisbelow(True)

        # Secondary axis: Average lift
        if "avg_lift" in df_sorted.columns:
            ax2 = ax1.twinx()
            color2 = PUBLICATION_COLORS["secondary"]
            ax2.plot(
                df_sorted["support"],
                df_sorted["avg_lift"],
                marker="s",
                linewidth=2.5,
                markersize=7,
                color=color2,
                markerfacecolor=color2,
                markeredgecolor="white",
                markeredgewidth=1.5,
                label="Average Lift",
                zorder=3,
            )
            ax2.set_ylabel("Average Lift (Effect Size)", fontweight="bold", fontsize=10, color=color2)
            ax2.tick_params(axis="y", labelcolor=color2)

            # Combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.95)

        ax1.set_title(
            "Parameter Sensitivity: Support Threshold Impact",
            fontweight="bold",
            fontsize=11,
            pad=15,
        )

        plt.tight_layout()

        if filename:
            self.save_figure(fig, filename)

        return fig


# =================== 2. BOOTSTRAP STABILITY VISUALIZER =======================

class BootstrapStabilityVisualizer(BaseVisualizer):
    """Publication-grade bootstrap stability analysis."""

    def plot_stability_waterfall(
        self,
        edge_stability_df: pd.DataFrame,
        figsize: Tuple[float, float] = FIGURE_SIZES["double_column"],
        filename: Optional[str] = None,
        top_k: int = 30,
        **kwargs,
    ) -> plt.Figure:
        """
        Generate waterfall plot of edge stability scores.

        Parameters
        ----------
        edge_stability_df : pd.DataFrame
            Columns: [edge, n_occurrences, stability_score, rank]
        figsize : tuple
            Figure dimensions
        filename : str
            Output filename
        top_k : int
            Number of top edges to display

        Returns
        -------
        plt.Figure
        """
        if edge_stability_df.empty:
            raise ValueError("edge_stability_df is empty")

        if "stability_score" not in edge_stability_df.columns:
            raise ValueError("'stability_score' column required")

        # Get top K edges
        top_edges = edge_stability_df.nlargest(top_k, "stability_score").copy()
        top_edges = top_edges.sort_values("stability_score", ascending=True)

        # Prepare labels
        if "edge" in top_edges.columns:
            labels = top_edges["edge"].astype(str).tolist()
        else:
            labels = [f"Edge {i+1}" for i in range(len(top_edges))]

        # Color gradient based on stability
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(top_edges)))

        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)

        # Horizontal bars
        y_pos = np.arange(len(top_edges))
        bars = ax.barh(
            y_pos,
            top_edges["stability_score"],
            color=colors,
            edgecolor=PUBLICATION_COLORS["primary"],
            linewidth=1.2,
        )

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_edges["stability_score"])):
            ax.text(
                val + 0.01,
                i,
                f"{val:.1%}",
                va="center",
                ha="left",
                fontsize=7,
                fontweight="bold",
            )

        # Styling
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Stability Score (Bootstrap Frequency)", fontweight="bold", fontsize=10)
        ax.set_title(
            f"Bootstrap Stability Analysis: Top {top_k} Most Robust Testing Cascades",
            fontweight="bold",
            fontsize=11,
            pad=15,
        )
        ax.set_xlim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="x", linewidth=0.5)
        ax.set_axisbelow(True)

        # Add threshold line (e.g., 80% stability)
        ax.axvline(x=0.8, color=PUBLICATION_COLORS["secondary"], linestyle="--", linewidth=1.5, alpha=0.7, label="80% Threshold")
        ax.legend(loc="lower right", framealpha=0.95)

        plt.tight_layout()

        if filename:
            self.save_figure(fig, filename)

        return fig

    def plot_stability_heatmap(
        self,
        edge_stability_matrix: pd.DataFrame,
        figsize: Tuple[float, float] = FIGURE_SIZES["full_page"],
        filename: Optional[str] = None,
        max_edges: int = 40,
        **kwargs,
    ) -> plt.Figure:
        """
        Generate heatmap of edge presence across bootstrap resamples.

        Parameters
        ----------
        edge_stability_matrix : pd.DataFrame
            Rows: edges, Columns: bootstrap resamples, Values: 0/1
        figsize : tuple
            Figure dimensions
        filename : str
            Output filename
        max_edges : int
            Maximum edges to display

        Returns
        -------
        plt.Figure
        """
        if edge_stability_matrix.empty:
            raise ValueError("edge_stability_matrix is empty")

        # Limit to top edges for clarity
        if len(edge_stability_matrix) > max_edges:
            stability_scores = edge_stability_matrix.sum(axis=1) / edge_stability_matrix.shape[1]
            top_edges = stability_scores.nlargest(max_edges).index
            matrix_subset = edge_stability_matrix.loc[top_edges]
        else:
            matrix_subset = edge_stability_matrix

        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)

        # Custom colormap: white (absent) → blue (present)
        cmap = ListedColormap(["white", PUBLICATION_COLORS["primary"]])

        sns.heatmap(
            matrix_subset,
            cmap=cmap,
            cbar_kws={"label": "Edge Present", "shrink": 0.6, "ticks": [0, 1]},
            ax=ax,
            linewidths=0.05,
            linecolor=PUBLICATION_COLORS["grid"],
        )

        ax.set_title(
            f"Bootstrap Stability Matrix: Edge Persistence Across {matrix_subset.shape[1]} Resamples",
            fontweight="bold",
            fontsize=11,
            pad=15,
        )
        ax.set_xlabel("Bootstrap Resample Index", fontweight="bold", fontsize=10)
        ax.set_ylabel("Testing Cascade (Ranked by Stability)", fontweight="bold", fontsize=10)

        # Customize colorbar
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0.25, 0.75])
        cbar.set_ticklabels(["Absent", "Present"])

        plt.tight_layout()

        if filename:
            self.save_figure(fig, filename)

        return fig


# ==================== 3. COMBINED MULTI-PANEL VISUALIZER =======================

class CombinedMultiPanelVisualizer(BaseVisualizer):
    """Publication-grade multi-panel figure (main text figure)."""

    def plot(
        self,
        grid_summary_df: Optional[pd.DataFrame] = None,
        edge_stability_df: Optional[pd.DataFrame] = None,
        figsize: Tuple[float, float] = FIGURE_SIZES["double_column"],
        filename: Optional[str] = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Generate combined 2-panel figure for main text.

        Panel A: Grid sensitivity (line plot with dual y-axes)
        Panel B: Bootstrap stability (waterfall plot)

        Parameters
        ----------
        grid_summary_df : pd.DataFrame
            Grid sensitivity data
        edge_stability_df : pd.DataFrame
            Bootstrap stability data
        figsize : tuple
            Figure dimensions
        filename : str
            Output filename

        Returns
        -------
        plt.Figure
        """
        n_panels = sum([
            grid_summary_df is not None and not grid_summary_df.empty,
            edge_stability_df is not None and not edge_stability_df.empty,
        ])

        if n_panels == 0:
            raise ValueError("At least one dataset required")

        fig = plt.figure(figsize=figsize, dpi=self.dpi)
        gs = fig.add_gridspec(1, n_panels, hspace=0.3, wspace=0.35)

        panel_idx = 0

        # Panel A: Grid Sensitivity
        if grid_summary_df is not None and not grid_summary_df.empty:
            ax1 = fig.add_subplot(gs[0, panel_idx])
            df_sorted = grid_summary_df.sort_values("support")

            # Primary axis
            color1 = PUBLICATION_COLORS["primary"]
            ax1.plot(
                df_sorted["support"],
                df_sorted["n_rules"],
                marker="o",
                linewidth=2.5,
                markersize=7,
                color=color1,
                markerfacecolor=color1,
                markeredgecolor="white",
                markeredgewidth=1.5,
                zorder=3,
            )
            ax1.set_xlabel("Support Threshold", fontweight="bold", fontsize=9)
            ax1.set_ylabel("Rule Count", fontweight="bold", fontsize=9, color=color1)
            ax1.tick_params(axis="y", labelcolor=color1)
            ax1.grid(True, alpha=0.25, linewidth=0.5, zorder=1)
            ax1.set_axisbelow(True)
            ax1.set_title("A. Parameter Sensitivity", fontweight="bold", fontsize=10, pad=10)

            # Secondary axis
            if "avg_lift" in df_sorted.columns:
                ax2 = ax1.twinx()
                color2 = PUBLICATION_COLORS["secondary"]
                ax2.plot(
                    df_sorted["support"],
                    df_sorted["avg_lift"],
                    marker="s",
                    linewidth=2.5,
                    markersize=7,
                    color=color2,
                    markerfacecolor=color2,
                    markeredgecolor="white",
                    markeredgewidth=1.5,
                    zorder=3,
                )
                ax2.set_ylabel("Average Lift", fontweight="bold", fontsize=9, color=color2)
                ax2.tick_params(axis="y", labelcolor=color2)

            panel_idx += 1

        # Panel B: Bootstrap Stability
        if edge_stability_df is not None and not edge_stability_df.empty:
            ax3 = fig.add_subplot(gs[0, panel_idx])
            top_edges = edge_stability_df.nlargest(20, "stability_score").copy()
            top_edges = top_edges.sort_values("stability_score", ascending=True)

            if "edge" in top_edges.columns:
                labels = top_edges["edge"].astype(str).tolist()
            else:
                labels = [f"E{i+1}" for i in range(len(top_edges))]

            colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(top_edges)))
            y_pos = np.arange(len(top_edges))

            bars = ax3.barh(
                y_pos,
                top_edges["stability_score"],
                color=colors,
                edgecolor=PUBLICATION_COLORS["primary"],
                linewidth=1.0,
            )

            for i, (bar, val) in enumerate(zip(bars, top_edges["stability_score"])):
                ax3.text(
                    val + 0.01,
                    i,
                    f"{val:.0%}",
                    va="center",
                    ha="left",
                    fontsize=6.5,
                    fontweight="bold",
                )

            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(labels, fontsize=6.5)
            ax3.set_xlabel("Stability Score", fontweight="bold", fontsize=9)
            ax3.set_title("B. Bootstrap Stability", fontweight="bold", fontsize=10, pad=10)
            ax3.set_xlim(0, 1.05)
            ax3.grid(True, alpha=0.25, axis="x", linewidth=0.5)
            ax3.set_axisbelow(True)
            ax3.axvline(x=0.8, color=PUBLICATION_COLORS["secondary"], linestyle="--", linewidth=1.2, alpha=0.7)

        fig.suptitle(
            "Sensitivity & Stability Analysis of Testing Cascade Discovery",
            fontsize=11,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout()

        if filename:
            self.save_figure(fig, filename)

        return fig


# ===================== MAIN ORCHESTRATOR CLASS ===============================

class CascadeVisualizationEngine:
    """
    Publication-grade orchestrator for sensitivity & bootstrap visualizations.

    Designed for high-impact journal submissions (Nature, Science, Lancet, PNAS).

    Features:
      ✓ 300-600 DPI multi-format output (PNG, PDF, SVG)
      ✓ Colorblind-safe palettes
      ✓ Professional typography
      ✓ Statistical annotations
      ✓ Publication-ready legends
    """

    def __init__(
        self,
        output_dir: str | Path = "./publication_figures",
        seed: int = 42,
        dpi: int = 300,
        verbose: bool = True,
        rules_df: Optional[pd.DataFrame] = None,  # For API compatibility
    ):
        self.output_dir = _ensure_dir(output_dir)
        self.seed = seed
        self.dpi = dpi
        self.verbose = verbose
        self.rules_df = rules_df

        # Initialize visualizers
        self.grid_sensitivity = GridSensitivityVisualizer(output_dir, seed, dpi, verbose)
        self.bootstrap_stability = BootstrapStabilityVisualizer(output_dir, seed, dpi, verbose)
        self.combined = CombinedMultiPanelVisualizer(output_dir, seed, dpi, verbose)

        if self.verbose:
            print("\n" + "=" * 80)
            print("CASCADE VISUALIZATION ENGINE - PUBLICATION GRADE")
            print("=" * 80)
            print(f"Output: {output_dir} | DPI: {dpi} | Formats: PNG, PDF, SVG")

    def plot_grid_sensitivity(
        self,
        grid_summary_df: pd.DataFrame,
        filename: str = "Fig_S1_grid_sensitivity.png",
        **kwargs,
    ) -> plt.Figure:
        """Generate grid sensitivity figure (supplementary)."""
        if self.verbose:
            print(f"\n[1/4] Grid Sensitivity Analysis...")
        return self.grid_sensitivity.plot(
            grid_summary_df=grid_summary_df,
            filename=filename,
            **kwargs,
        )

    def plot_bootstrap_stability_waterfall(
        self,
        edge_stability_df: pd.DataFrame,
        top_k: int = 30,
        filename: str = "Fig_S2_bootstrap_stability.png",
        **kwargs,
    ) -> plt.Figure:
        """Generate bootstrap stability waterfall plot (supplementary)."""
        if self.verbose:
            print(f"[2/4] Bootstrap Stability Waterfall...")
        return self.bootstrap_stability.plot_stability_waterfall(
            edge_stability_df=edge_stability_df,
            top_k=top_k,
            filename=filename,
            **kwargs,
        )

    def plot_bootstrap_stability_heatmap(
        self,
        edge_stability_matrix: pd.DataFrame,
        max_edges: int = 40,
        filename: str = "Fig_S3_bootstrap_heatmap.png",
        **kwargs,
    ) -> plt.Figure:
        """Generate bootstrap stability heatmap (supplementary)."""
        if self.verbose:
            print(f"[3/4] Bootstrap Stability Heatmap...")
        return self.bootstrap_stability.plot_stability_heatmap(
            edge_stability_matrix=edge_stability_matrix,
            max_edges=max_edges,
            filename=filename,
            **kwargs,
        )

    def plot_combined_main_figure(
        self,
        grid_summary_df: Optional[pd.DataFrame] = None,
        edge_stability_df: Optional[pd.DataFrame] = None,
        filename: str = "Fig_3_sensitivity_stability.png",
        **kwargs,
    ) -> plt.Figure:
        """Generate combined multi-panel figure (main text)."""
        if self.verbose:
            print(f"[4/4] Combined Main Figure...")
        return self.combined.plot(
            grid_summary_df=grid_summary_df,
            edge_stability_df=edge_stability_df,
            filename=filename,
            **kwargs,
        )

    def generate_manuscript_figures(
        self,
        grid_summary_df: Optional[pd.DataFrame] = None,
        edge_stability_df: Optional[pd.DataFrame] = None,
        edge_stability_matrix: Optional[pd.DataFrame] = None,
        include_supplementary: bool = True,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """
        Generate complete figure set for manuscript submission.

        Main Text:
          - Fig 3: Combined sensitivity + stability (2-panel)

        Supplementary:
          - Fig S1: Grid sensitivity (full)
          - Fig S2: Bootstrap stability waterfall
          - Fig S3: Bootstrap stability heatmap

        Parameters
        ----------
        grid_summary_df : pd.DataFrame
            Grid sensitivity results
        edge_stability_df : pd.DataFrame
            Edge stability summary
        edge_stability_matrix : pd.DataFrame
            Edge × bootstrap matrix
        include_supplementary : bool
            Generate supplementary figures

        Returns
        -------
        dict
            {figure_name: plt.Figure}
        """
        results = {}

        # MAIN TEXT FIGURE
        if (grid_summary_df is not None and not grid_summary_df.empty) or (
            edge_stability_df is not None and not edge_stability_df.empty
        ):
            fig = self.plot_combined_main_figure(
                grid_summary_df=grid_summary_df,
                edge_stability_df=edge_stability_df,
                filename="Fig_3_sensitivity_stability.png",
            )
            results["main_figure"] = fig

        # SUPPLEMENTARY FIGURES
        if include_supplementary:
            if grid_summary_df is not None and not grid_summary_df.empty:
                fig = self.plot_grid_sensitivity(
                    grid_summary_df=grid_summary_df,
                    filename="Fig_S1_grid_sensitivity.png",
                )
                results["supp_grid"] = fig

            if edge_stability_df is not None and not edge_stability_df.empty:
                fig = self.plot_bootstrap_stability_waterfall(
                    edge_stability_df=edge_stability_df,
                    top_k=30,
                    filename="Fig_S2_bootstrap_stability.png",
                )
                results["supp_bootstrap_waterfall"] = fig

            if edge_stability_matrix is not None and not edge_stability_matrix.empty:
                fig = self.plot_bootstrap_stability_heatmap(
                    edge_stability_matrix=edge_stability_matrix,
                    max_edges=40,
                    filename="Fig_S3_bootstrap_heatmap.png",
                )
                results["supp_bootstrap_heatmap"] = fig

        if self.verbose:
            print("\n" + "=" * 80)
            print("✓ PUBLICATION FIGURES COMPLETE")
            print(f"  Main Text: {1 if 'main_figure' in results else 0} figure")
            print(f"  Supplementary: {len(results) - 1} figures")
            print(f"  Total: {len(results)} figures")
            print(f"  Location: {self.output_dir}")
            print(f"  Formats: PNG ({self.dpi} DPI), PDF (vector), SVG (editable)")
            print("=" * 80)

        return results


if __name__ == "__main__":
    print("✓ Publication-Grade CascadeVisualizationEngine loaded!")
    print("\nFEATURES:")
    print("  • 300-600 DPI multi-format output (PNG, PDF, SVG)")
    print("  • Colorblind-safe palettes")
    print("  • Nature/Science typography standards")
    print("  • Statistical annotations")
    print("  • Professional legends and labels")
    print("\nUSAGE:")
    print("  from VisualizationEngine import CascadeVisualizationEngine")
    print("  engine = CascadeVisualizationEngine(output_dir='./figures', dpi=300)")
    print("  figs = engine.generate_manuscript_figures(")
    print("      grid_summary_df=grid_df,")
    print("      edge_stability_df=stability_df,")
    print("      edge_stability_matrix=matrix_df,")
    print("  )")