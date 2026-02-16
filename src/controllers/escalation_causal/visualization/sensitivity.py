"""
Sensitivity contour plots (Cinelli‑Hazlett style).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .base import PlotlyFigure


class SensitivityContour(PlotlyFigure):
    """
    Contour plot of adjusted estimate as a function of confounder strength.
    """

    def __init__(
        self,
        contour_data: pd.DataFrame,
        x_col: str = "r2_y_z",
        y_col: str = "r2_d_z",
        z_col: str = "estimate_adj",
        title: str = "Sensitivity Contour: Adjusted Estimate",
        xlabel: str = r"Partial $R^2$ of confounder with outcome",
        ylabel: str = r"Partial $R^2$ of confounder with treatment",
        levels: int = 20,
        show_rv: bool = True,
        rv_q: float = None,
        rv_alpha: float = None,
        alpha: float = 0.05,  # significance level for labeling
        height: int = 600,
        width: int = 700,
    ):
        """
        Args:
            contour_data: DataFrame from `cinelli_hazlett.contour_data()`.
            x_col, y_col, z_col: Column names.
            show_rv: Mark robustness value points.
            rv_q, rv_alpha: Robustness values (if None, not shown).
            alpha: Significance level (for labeling).
        """
        # Pivot for contour
        pivot = contour_data.pivot(index=y_col, columns=x_col, values=z_col)
        x_vals = pivot.columns.values
        y_vals = pivot.index.values
        z_vals = pivot.values

        fig = go.Figure()

        # Contour trace
        fig.add_trace(go.Contour(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            colorscale="Viridis",
            contours=dict(
                coloring="heatmap",
                showlabels=True,
                labelfont=dict(size=10, color="white"),
            ),
            colorbar=dict(title="Adjusted RD"),
            ncontours=levels,
        ))

        if show_rv and rv_q is not None:
            fig.add_trace(go.Scatter(
                x=[rv_q],
                y=[rv_q],
                mode="markers",
                marker=dict(size=12, color="red", symbol="star"),
                name="RV (zero)",
            ))
        if show_rv and rv_alpha is not None:
            fig.add_trace(go.Scatter(
                x=[rv_alpha],
                y=[rv_alpha],
                mode="markers",
                marker=dict(size=12, color="orange", symbol="star"),
                name=f"RV (α={alpha})",
            ))

        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            height=int(height),
            width=int(width),
            template="plotly_white",
        )

        super().__init__(fig)