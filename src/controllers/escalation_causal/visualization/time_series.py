"""
Time series of escalation RD over years or months.
"""

import pandas as pd
import plotly.graph_objects as go

from .base import PlotlyFigure


class TimeTrendPlot(PlotlyFigure):
    """
    Line plot of risk difference over time with confidence bands.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        time_col: str,
        estimate_col: str,
        se_col: str = None,
        ci_low_col: str = None,
        ci_high_col: str = None,
        title: str = "Escalation RD Over Time",
        height: int = 500,
        width: int = 800,
    ):
        df = data.sort_values(time_col)

        fig = go.Figure()

        if se_col is not None:
            ci_low = df[estimate_col] - 1.96 * df[se_col]
            ci_high = df[estimate_col] + 1.96 * df[se_col]
            fig.add_trace(go.Scatter(
                x=df[time_col],
                y=ci_low,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="none",
            ))
            fig.add_trace(go.Scatter(
                x=df[time_col],
                y=ci_high,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(0,100,80,0.2)",
                line=dict(width=0),
                name="95% CI",
            ))
        elif ci_low_col is not None and ci_high_col is not None:
            fig.add_trace(go.Scatter(
                x=df[time_col],
                y=df[ci_low_col],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="none",
            ))
            fig.add_trace(go.Scatter(
                x=df[time_col],
                y=df[ci_high_col],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(0,100,80,0.2)",
                line=dict(width=0),
                name="95% CI",
            ))

        fig.add_trace(go.Scatter(
            x=df[time_col],
            y=df[estimate_col],
            mode="lines+markers",
            name="RD",
            line=dict(color="black"),
            marker=dict(size=6),
        ))

        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title=title,
            xaxis_title=time_col,
            yaxis_title="Risk Difference",
            height=height,
            width=width,
            template="plotly_white",
        )
        super().__init__(fig)