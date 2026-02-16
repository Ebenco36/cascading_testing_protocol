"""
Forest plot stratified by clinically relevant subgroups.
"""

import pandas as pd
import plotly.graph_objects as go

from .forest import ForestPlot


class SubgroupForestPlot(ForestPlot):
    """
    Forest plot showing estimates for each subgroup, possibly with overall.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        subgroup_col: str,
        estimate_col: str = "ate",
        ci_low_col: str = "ci_low",
        ci_high_col: str = "ci_high",
        title: str = "Subgroup Forest Plot",
        height: int = 600,
        width: int = 700,
    ):
        """
        data should contain one row per subgroup with estimate and CI.
        """
        df = data.sort_values(estimate_col, ascending=False).reset_index(drop=True)
        y_cats = df[subgroup_col].tolist()[::-1]
        y_pos = list(range(len(df)))

        fig = go.Figure()

        # Confidence intervals as horizontal lines
        for i, (idx, row) in enumerate(df.iterrows()):
            y = y_pos[len(df) - 1 - i]
            fig.add_shape(
                type="line",
                x0=row[ci_low_col],
                x1=row[ci_high_col],
                y0=y,
                y1=y,
                line=dict(color="gray", width=3),
                layer="below",
            )

        # Point estimates
        fig.add_trace(go.Scatter(
            x=df[estimate_col].tolist()[::-1],
            y=y_pos,
            mode="markers",
            marker=dict(size=10, color="blue"),
            name="Estimate",
        ))

        fig.add_vline(x=0, line_dash="dash", line_color="black")
        fig.update_layout(
            title=title,
            xaxis_title="Estimate",
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
        super().__init__(fig)  # passes fig to PlotlyFigure