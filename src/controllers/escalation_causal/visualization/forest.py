"""
Forest plot for average risk differences.
"""

from typing import Optional, List

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from .base import PlotlyFigure


class ForestPlot(PlotlyFigure):
    """
    Forest plot of risk differences with 95% confidence intervals.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        estimate_col: str = "rd",
        ci_low_col: str = "ci_low",
        ci_high_col: str = "ci_high",
        label_col: str = "pair",  # or combine trigger + target
        title: str = "Risk Difference Forest Plot",
        xlabel: str = "Risk Difference (RD)",
        sort_by: Optional[str] = None,
        colors: Optional[List[str]] = None,
        height: int = 600,
        width: int = 800,
    ):
        """
        Args:
            data: DataFrame with one row per pair.
            estimate_col: Column with point estimates.
            ci_low_col: Column with lower confidence bound.
            ci_high_col: Column with upper confidence bound.
            label_col: Column with pair labels (or create from trigger/target).
            title: Plot title.
            xlabel: X‑axis label.
            sort_by: Column to sort rows by (e.g., 'rd').
            colors: Optional list of colors for each row.
        """
        df = data.copy()
        if label_col not in df.columns and "trigger" in df.columns and "target" in df.columns:
            df["pair"] = df["trigger"] + " → " + df["target"]
            label_col = "pair"
        elif label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found and cannot be created.")

        if sort_by is not None:
            df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)

        # Create y‑axis categories (reverse order so top row is first)
        y_cats = df[label_col].tolist()[::-1]
        y_pos = list(range(len(df)))

        # Create figure
        fig = go.Figure()

        # Add confidence intervals as horizontal lines
        fig.add_trace(go.Scatter(
            x=[None], y=[None],  # dummy for legend
            mode="lines",
            line=dict(color="gray", width=2),
            name="95% CI",
            showlegend=True,
        ))
        for i, (idx, row) in enumerate(df.iterrows()):
            y = y_pos[len(df) - 1 - i]  # reverse to match y_cats
            fig.add_shape(
                type="line",
                x0=row[ci_low_col],
                x1=row[ci_high_col],
                y0=y,
                y1=y,
                line=dict(color="gray", width=2),
                layer="below",
            )

        # Add point estimates as markers
        fig.add_trace(go.Scatter(
            x=df[estimate_col].tolist()[::-1],
            y=y_pos,
            mode="markers",
            marker=dict(
                size=10,
                color=colors if colors else px.colors.qualitative.Plotly[0],
                line=dict(width=1, color="black"),
            ),
            name="Estimate",
            showlegend=True,
        ))

        # Add vertical line at zero
        fig.add_vline(x=0, line=dict(color="black", width=1, dash="dash"))

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis=dict(
                tickmode="array",
                tickvals=y_pos,
                ticktext=y_cats,
                autorange="reversed",  # already reversed, but ensure
            ),
            height=height,
            width=width,
            margin=dict(l=150, r=50, t=80, b=50),
            template="plotly_white",
            hovermode="y",
        )

        # Add hover info
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>RD: %{x:.3f}<extra></extra>",
            selector=dict(mode="markers"),
        )

        super().__init__(fig)