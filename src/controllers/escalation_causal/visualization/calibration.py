"""
Calibration plot for routine policy probabilities.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve

from .base import PlotlyFigure


class CalibrationPlot(PlotlyFigure):
    """
    Plot calibration curve (fraction of positives vs. mean predicted probability).
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Calibration Plot",
        n_bins: int = 10,
        height: int = 500,
        width: int = 600,
    ):
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy="uniform")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=prob_pred,
            y=prob_true,
            mode="lines+markers",
            name="Model",
            line=dict(color="blue"),
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line=dict(dash="dash", color="gray"),
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Mean predicted probability",
            yaxis_title="Fraction of positives",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            height=height,
            width=width,
            template="plotly_white",
        )
        super().__init__(fig)