"""
Clustered heatmap for risk difference or co‑resistance matrices.
"""

import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

from .base import PlotlyFigure


class ClusteredHeatmap(PlotlyFigure):
    """
    Heatmap with optional hierarchical clustering of rows and columns.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        title: str = "Heatmap",
        colorscale: str = "RdBu",
        zmid: float = 0.0,
        cluster_rows: bool = True,
        cluster_cols: bool = True,
        show_values: bool = False,
        height: int = 800,
        width: int = 800,
    ):
        """
        Args:
            data: DataFrame with values to plot (index = rows, columns = columns).
            cluster_rows: Whether to reorder rows by hierarchical clustering.
            cluster_cols: Whether to reorder columns by hierarchical clustering.
            show_values: Annotate cells with values.
        """
        df = data.copy()

        if cluster_rows:
            # Compute linkage for rows
            row_linkage = linkage(pdist(df.values, metric='euclidean'), method='average')
            row_order = dendrogram(row_linkage, no_plot=True)['leaves']
            df = df.iloc[row_order]

        if cluster_cols:
            col_linkage = linkage(pdist(df.T.values, metric='euclidean'), method='average')
            col_order = dendrogram(col_linkage, no_plot=True)['leaves']
            df = df.iloc[:, col_order]

        z = df.values
        x = df.columns.tolist()
        y = df.index.tolist()

        if show_values:
            # Annotated heatmap
            fig = ff.create_annotated_heatmap(
                z,
                x=x,
                y=y,
                colorscale=colorscale,
                showscale=True,
                zmid=zmid,
            )
        else:
            # Simple heatmap
            fig = go.Figure(data=go.Heatmap(
                z=z,
                x=x,
                y=y,
                colorscale=colorscale,
                zmid=zmid,
                colorbar=dict(title="Value"),
            ))
            fig.update_layout(
                xaxis=dict(tickangle=-45),
                yaxis=dict(tickangle=0),
            )

        fig.update_layout(
            title=title,
            height=height,
            width=width,
            xaxis_title="Target",
            yaxis_title="Trigger",
        )
        super().__init__(fig)