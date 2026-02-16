"""
Combine multiple subplots into one figure.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .base import PlotlyFigure


class CombinedFigure(PlotlyFigure):
    """
    Create a figure with multiple subplots arranged in a grid.
    """

    def __init__(
        self,
        subplots: list,
        rows: int,
        cols: int,
        titles: list = None,
        height: int = 800,
        width: int = 1200,
        **subplot_kwargs,
    ):
        """
        Args:
            subplots: List of PlotlyFigure objects or go.Figure objects.
            rows, cols: Grid dimensions.
            titles: List of subplot titles.
        """
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, **subplot_kwargs)

        for idx, sp in enumerate(subplots):
            if sp is None:
                continue
            r = idx // cols + 1
            c = idx % cols + 1
            for trace in sp.figure.data:
                fig.add_trace(trace, row=r, col=c)
            # Copy layout properties if needed? Usually not.

        fig.update_layout(height=height, width=width)
        super().__init__(fig)