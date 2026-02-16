"""
Base classes for Plotly visualizations.
"""

from pathlib import Path
from typing import Optional, Union

import plotly.graph_objects as go
import plotly.io as pio


class PlotlyFigure:
    """
    Wrapper for a Plotly figure with a convenient save method.
    """

    def __init__(self, figure: go.Figure):
        self.figure = figure

    def save(
        self,
        filename: Union[str, Path],
        width: int = 1200,
        height: int = 800,
        scale: float = 2,
        engine: str = "kaleido",
        **kwargs,
    ) -> None:
        """
        Save the figure to a file.

        Args:
            filename: Output file path (extension determines format: .html, .png, .pdf, .svg).
            width: Width in pixels (for static images).
            height: Height in pixels.
            scale: Scale factor for resolution (e.g., 2 for 2x).
            engine: Engine for static export ("kaleido" or "orca").
        """
        filename = Path(filename)
        suffix = filename.suffix.lower()

        if suffix == ".html":
            # HTML export
            self.figure.write_html(
                filename,
                include_plotlyjs="cdn",
                config={"responsive": True},
            )
        else:
            # Static image export
            pio.kaleido.scope.default_width = width
            pio.kaleido.scope.default_height = height
            pio.kaleido.scope.default_scale = scale
            self.figure.write_image(filename, engine=engine, **kwargs)

    def show(self) -> None:
        """Display the figure in a Jupyter notebook or browser."""
        self.figure.show()


class PlotFactory:
    """Factory to create common plot types."""

    @staticmethod
    def forest_plot(data, **kwargs):
        from .forest import ForestPlot
        return ForestPlot(data, **kwargs).figure