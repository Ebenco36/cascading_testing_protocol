"""
Posterior probability bar chart and shrinkage forest plot.
"""

import pandas as pd
import plotly.graph_objects as go

from .base import PlotlyFigure
from .forest import ForestPlot


class PosteriorProbabilityPlot(PlotlyFigure):
    """
    Bar chart of posterior probabilities that RD > 0.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        pair_col: str = "pair",
        prob_col: str = "prob_positive",
        title: str = "Posterior Probability of Positive Effect",
        threshold: float = 0.95,
        height: int = 600,
        width: int = 800,
    ):
        df = data.sort_values(prob_col, ascending=False).reset_index(drop=True)
        colors = ["red" if p >= threshold else "steelblue" for p in df[prob_col]]

        fig = go.Figure(data=go.Bar(
            x=df[prob_col],
            y=df[pair_col],
            orientation="h",
            marker=dict(color=colors),
            text=df[prob_col].round(3),
            textposition="outside",
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Posterior Probability",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(autorange="reversed"),
            height=height,
            width=width,
            template="plotly_white",
        )
        super().__init__(fig)


class ShrinkageForestPlot(ForestPlot):
    """
    Forest plot comparing original and shrunken estimates with credible intervals.
    """
    def __init__(
        self,
        data: pd.DataFrame,
        original_est_col: str = "rd",
        original_se_col: str = "se",
        shrunken_est_col: str = "theta_shrunken",
        shrunken_sd_col: str = "theta_sd",
        pair_col: str = "pair",
        title: str = "Bayesian Shrinkage Forest Plot",
        height: int = 700,
        width: int = 900,
    ):
        # Prepare data in the format expected by ForestPlot (multiple traces)
        df = data.sort_values(original_est_col, ascending=False).reset_index(drop=True)
        y_cats = df[pair_col].tolist()[::-1]
        y_pos = list(range(len(df)))

        fig = go.Figure()

        # Original 95% CI (gray)
        for i, (idx, row) in enumerate(df.iterrows()):
            y = y_pos[len(df) - 1 - i]
            ci_low = row[original_est_col] - 1.96 * row[original_se_col]
            ci_high = row[original_est_col] + 1.96 * row[original_se_col]
            fig.add_shape(
                type="line",
                x0=ci_low,
                x1=ci_high,
                y0=y,
                y1=y,
                line=dict(color="lightgray", width=3),
                layer="below",
            )

        # Shrunken 95% CrI (blue)
        for i, (idx, row) in enumerate(df.iterrows()):
            y = y_pos[len(df) - 1 - i]
            ci_low = row[shrunken_est_col] - 1.96 * row[shrunken_sd_col]
            ci_high = row[shrunken_est_col] + 1.96 * row[shrunken_sd_col]
            fig.add_shape(
                type="line",
                x0=ci_low,
                x1=ci_high,
                y0=y,
                y1=y,
                line=dict(color="blue", width=4),
                layer="below",
            )

        # Original estimates (gray dots)
        fig.add_trace(go.Scatter(
            x=df[original_est_col].tolist()[::-1],
            y=y_pos,
            mode="markers",
            marker=dict(size=8, color="gray", symbol="circle"),
            name="Original",
        ))

        # Shrunken estimates (blue diamonds)
        fig.add_trace(go.Scatter(
            x=df[shrunken_est_col].tolist()[::-1],
            y=y_pos,
            mode="markers",
            marker=dict(size=10, color="blue", symbol="diamond"),
            name="Shrunken",
        ))

        fig.add_vline(x=0, line=dict(color="black", width=1, dash="dash"))
        fig.update_layout(
            title=title,
            xaxis_title="Risk Difference",
            yaxis=dict(
                tickmode="array",
                tickvals=y_pos,
                ticktext=y_cats,
                autorange="reversed",
            ),
            height=height,
            width=width,
            template="plotly_white",
        )
        super().__init__(fig)  # Note: this calls the base with fig, not ForestPlot.__init__