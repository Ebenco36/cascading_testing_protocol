"""
Scatter plot comparing co‑resistance odds ratio and escalation risk difference.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .base import PlotlyFigure


class CorrelationScatter(PlotlyFigure):
    """
    Scatter plot of two metrics (e.g., co‑resistance OR vs. escalation RD).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        label_col: str = "pair",
        title: str = "Correlation Scatter",
        xlabel: str = None,
        ylabel: str = None,
        add_trendline: bool = True,
        highlight: list = None,
        height: int = 600,
        width: int = 800,
    ):
        """
        Args:
            data: DataFrame containing columns for x, y, and labels.
            x_col, y_col: Column names.
            label_col: Column used for point labels.
            add_trendline: Add ordinary least squares trendline.
            highlight: List of indices or labels to highlight.
        """
        df = data.copy()
        if label_col not in df.columns and "trigger" in df.columns and "target" in df.columns:
            df["pair"] = df["trigger"] + " → " + df["target"]
            label_col = "pair"

        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            text=label_col,
            trendline="ols" if add_trendline else None,
            labels={x_col: xlabel or x_col, y_col: ylabel or y_col},
            title=title,
            height=height,
            width=width,
        )

        if highlight is not None:
            # Highlight specific points (e.g., significant pairs)
            highlight_df = df.loc[highlight] if isinstance(highlight, (list, pd.Index)) else df[df[label_col].isin(highlight)]
            fig.add_trace(go.Scatter(
                x=highlight_df[x_col],
                y=highlight_df[y_col],
                mode="markers+text",
                marker=dict(size=12, color="red", symbol="circle"),
                text=highlight_df[label_col],
                textposition="top center",
                name="Highlighted",
            ))

        super().__init__(fig)