"""
Network graph of trigger–target effects.
"""

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from .base import PlotlyFigure


class NetworkPlot(PlotlyFigure):
    """
    Directed graph where nodes are antibiotics, edges represent significant effects.
    Edge thickness proportional to effect size, color by direction.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        threshold: float = 0.05,
        estimate_col: str = "rd",
        p_col: str = "p_value",
        min_effect: float = 0.0,
        height: int = 800,
        width: int = 800,
    ):
        """
        Args:
            data: DataFrame with trigger, target, rd, p_value.
            threshold: p‑value threshold for significance.
            estimate_col: Column with effect sizes.
            p_col: Column with p‑values.
            min_effect: Minimum absolute effect size to show.
        """
        # Ensure integer dimensions
        height = int(height)
        width = int(width)

        # Filter significant pairs
        sig = data[data[p_col] < threshold].copy()
        sig = sig[abs(sig[estimate_col]) >= min_effect]

        if sig.empty:
            # Create empty figure
            fig = go.Figure()
            fig.add_annotation(text="No significant edges to display", showarrow=False)
            super().__init__(fig)
            return

        # Build graph
        G = nx.DiGraph()
        for _, row in sig.iterrows():
            G.add_edge(
                row["trigger"],
                row["target"],
                weight=abs(row[estimate_col]),
                effect=row[estimate_col],
            )

        pos = nx.spring_layout(G, seed=42)

        # Create edges traces
        edge_traces = []
        for u, v, d in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            color = "red" if d["effect"] > 0 else "blue"
            line_width = max(1.0, d["weight"] * 10)  # scale thickness

            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=line_width, color=color),
                hoverinfo="none",
                showlegend=False,
            )
            edge_traces.append(edge_trace)

        # Node trace
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = list(G.nodes())

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            marker=dict(
                size=20,
                color="lightblue",
                line=dict(width=2, color="black"),
            ),
            hoverinfo="text",
        )

        # Combine traces
        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title="Trigger–Target Network of Escalation Effects",
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            height=height,
            width=width,
            template="plotly_white",
        )

        super().__init__(fig)