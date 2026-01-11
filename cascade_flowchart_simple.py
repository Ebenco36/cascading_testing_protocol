from __future__ import annotations
from dataclasses import dataclass
from graphviz import Digraph


@dataclass
class StyleCfg:
    """Visual style configuration for publication-ready flowcharts."""
    rankdir: str = "TB"
    splines: str = "ortho"
    nodesep: str = "0.30"
    ranksep: str = "0.55"
    pad: str = "0.20"

    fontname: str = "Helvetica"
    fontsize_graph: str = "12"
    fontsize_node: str = "8.5"
    fontsize_edge: str = "7"
    fontsize_cluster: str = "10.5"
    fontsize_detail: str = "7.5"

    box_w: str = "2.0"
    box_h: str = "0.65"
    dia_w: str = "2.2"
    dia_h: str = "0.90"
    detail_w: str = "4.5"

    show_detail_panel: bool = True   # <-- turn OFF for the cleanest journal figure


def build_arm_flowchart(format: str = "pdf", s: StyleCfg = StyleCfg()) -> Digraph:
    dot = Digraph("ARM_PostProcessing_Publication", format=format, engine="dot")

    dot.attr(
        rankdir=s.rankdir,
        splines=s.splines,
        bgcolor="white",
        nodesep=s.nodesep,
        ranksep=s.ranksep,
        pad=s.pad,
        concentrate="true",
        newrank="true",
        compound="true",
        fontname=s.fontname,
        fontsize=s.fontsize_graph,
    )

    # Helps Graphviz keep clusters compact and avoid weird stretching
    dot.attr(overlap="false")
    dot.attr(outputorder="edgesfirst")

    dot.attr(
        "node",
        fontname=s.fontname,
        fontsize=s.fontsize_node,
        color="#7A7A7A",
        penwidth="1.0",
    )
    dot.attr(
        "edge",
        fontname=s.fontname,
        fontsize=s.fontsize_edge,
        color="#4A4A4A",
        penwidth="1.0",
        arrowsize="0.70",
    )

    # Helpers
    def box(node_id: str, text: str, fill: str):
        dot.node(
            node_id, text,
            shape="box", style="filled,rounded",
            fillcolor=fill,
            width=s.box_w, height=s.box_h, fixedsize="true"
        )

    def diamond(node_id: str, text: str, fill: str):
        dot.node(
            node_id, text,
            shape="diamond", style="filled",
            fillcolor=fill,
            width=s.dia_w, height=s.dia_h, fixedsize="true"
        )

    def detail_box(node_id: str, text: str, fill: str):
        dot.node(
            node_id, text,
            shape="box", style="filled,rounded",
            fillcolor=fill,
            width=s.detail_w,
            fontsize=s.fontsize_detail
        )

    # Ortho-friendly decision labels
    def yes_edge(a: str, b: str, label: str = "Yes"):
        dot.edge(a, b, xlabel=label, fontcolor="#2E7D32", color="#2E7D32", penwidth="1.1", minlen="1")

    def no_edge(a: str, b: str, label: str = "No"):
        dot.edge(a, b, xlabel=label, fontcolor="#B71C1C", color="#9E9E9E", style="dashed", penwidth="0.9", minlen="1")

    # Palette
    C_INPUT = "#C8E6C9"
    C_CRITICAL = "#FFCCBC"
    C_DECISION = "#B2DFDB"
    C_PROCESS = "#E1F5FE"
    C_META = "#F3E5F5"
    C_VIZ = "#A5D6A7"
    C_EXPORT = "#EEEEEE"
    C_REJECT = "#F5F5F5"
    C_DETAIL = "#FFFEF0"

    # =======================
    # START
    # =======================
    dot.node(
        "START",
        "ARM\nRules",
        shape="circle",
        width="0.85", height="0.85", fixedsize="true",
        style="filled", fillcolor="#4CAF50",
        fontcolor="white", fontsize="9.5"
    )

    box("RAW_RULES", "Raw rules CSV\n(Ant. | Con. + metrics)", C_INPUT)
    dot.edge("START", "RAW_RULES", penwidth="1.1")

    # =======================
    # STAGE 1: CLEANING
    # =======================
    with dot.subgraph(name="cluster_stage1") as c:
        c.attr(
            label="Stage 1: Clean rules (Resistance → Cross-testing)",
            color="#90A4AE", style="rounded,bold",
            fontsize=s.fontsize_cluster, labeljust="l"
        )

        box("LOAD_CLEAN", "ARMRuleCleaner\nclean_dataframe()", C_CRITICAL)

        # Force the two filters to sit side-by-side (prevents long orthogonal runs)
        with c.subgraph(name="cluster_parallel") as p:
            p.attr(rank="same")
            box("ANT_FILT", "Keep *_R only\n(antecedent)", C_PROCESS)
            box("CON_FILT", "Keep *_T only\n(consequent)", C_PROCESS)

        box("TAUT_REM", "Drop implied tests\n(remove X_T if X_R)", C_PROCESS)

        diamond("CHK_INFO", "Any *_R\nleft?", C_DECISION)
        diamond("CHK_CROSS", "Any cross *_T\nleft?", C_DECISION)

        box("FINAL", "Cleaned rules\n(R_only, T_cross)", C_PROCESS)

        box("REJ_NOANT", "Reject\n(no resistance)", C_REJECT)
        box("REJ_NOCROSS", "Reject\n(no cross-test)", C_REJECT)

        # Inside-cluster connections
        c.edge("LOAD_CLEAN", "ANT_FILT")
        c.edge("LOAD_CLEAN", "CON_FILT")
        c.edge("ANT_FILT", "TAUT_REM")
        c.edge("CON_FILT", "TAUT_REM")
        c.edge("TAUT_REM", "CHK_INFO")
        c.edge("CHK_INFO", "CHK_CROSS")

    dot.edge("RAW_RULES", "LOAD_CLEAN")

    yes_edge("CHK_INFO", "CHK_CROSS")
    no_edge("CHK_INFO", "REJ_NOANT")

    yes_edge("CHK_CROSS", "FINAL")
    no_edge("CHK_CROSS", "REJ_NOCROSS")

    # =======================
    # DETAIL PANEL (optional, anchored so it DOES NOT stretch the layout)
    # =======================
    if s.show_detail_panel:
        with dot.subgraph(name="cluster_detail_clean") as c:
            c.attr(
                label="Cleaner details (caption-style summary)",
                color="#FFC107", style="rounded,bold,filled",
                fillcolor="#FFFDE7",
                fontsize=s.fontsize_cluster, labeljust="l"
            )

            detail_box(
                "DETAIL_CLEAN",
                "Token meaning: DRUG_R=resistant, DRUG_T=tested\n"
                "1) Split tokens by '|'\n"
                "2) Keep *_R in antecedent; keep *_T in consequent\n"
                "3) Remove implied tests: if X_R then drop X_T\n"
                "4) Cross-informative = (R_only non-empty) AND (T_cross non-empty)",
                C_DETAIL
            )

        # Key fix: keep LOAD_CLEAN and DETAIL_CLEAN on the SAME RANK
        # so the arrow is short and horizontal.
        with dot.subgraph() as same_rank:
            same_rank.attr(rank="same")
            same_rank.node("LOAD_CLEAN")
            same_rank.node("DETAIL_CLEAN")

        # Key fix: constraint=false so this edge does NOT warp the main pipeline.
        dot.edge(
            "LOAD_CLEAN", "DETAIL_CLEAN",
            style="dashed", color="#F57C00", penwidth="1.2",
            xlabel="explains",
            arrowhead="vee",
            constraint="false",
            minlen="1",
        )

    # =======================
    # STAGE 2: EDGES
    # =======================
    with dot.subgraph(name="cluster_stage2") as c:
        c.attr(
            label="Stage 2: Build cross edges (R → T)",
            color="#90A4AE", style="rounded,bold",
            fontsize=s.fontsize_cluster, labeljust="l"
        )

        c.node(
            "METADATA",
            "Drug metadata\n(AWaRe + class)",
            shape="tab", style="filled", fillcolor=C_META
        )
        box("EDGE_BUILD", "Build edges\n(src, dst, weight)", C_PROCESS)
        box("EDGE_FILTER", "Select important edges\n(top-N)", C_PROCESS)

        c.edge("EDGE_BUILD", "EDGE_FILTER")

    dot.edge("FINAL", "EDGE_BUILD", xlabel="R_only → T_cross")
    dot.edge("METADATA", "EDGE_BUILD")

    # =======================
    # STAGE 3: VISUALS
    # =======================
    with dot.subgraph(name="cluster_stage3") as c:
        c.attr(
            label="Stage 3: Visualizations & export",
            color="#90A4AE", style="rounded,bold",
            fontsize=s.fontsize_cluster, labeljust="l"
        )
        box("PLOT_GEN", "Plots\n(Sankey / Heatmap /\nChord / Network)", C_VIZ)
        box("EXPORT_ALL", "Export\n(HTML / PNG / SVG / PDF)", C_EXPORT)
        c.edge("PLOT_GEN", "EXPORT_ALL")

    dot.edge("EDGE_FILTER", "PLOT_GEN")

    # =======================
    # END
    # =======================
    dot.node(
        "END",
        "Final\nFigures",
        shape="doublecircle",
        width="0.90", height="0.90", fixedsize="true",
        style="filled", fillcolor="#4CAF50",
        fontcolor="white", fontsize="9.5"
    )
    dot.edge("EXPORT_ALL", "END", penwidth="1.1")

    # Rejected paths: do NOT affect layout
    dot.edge("REJ_NOANT", "END", style="dotted", color="#BDBDBD", constraint="false")
    dot.edge("REJ_NOCROSS", "END", style="dotted", color="#BDBDBD", constraint="false")

    return dot


if __name__ == "__main__":
    # Tip: for the *cleanest* publication figure, set show_detail_panel=False.
    # That keeps the main pipeline compact and places details in caption/supplement.
    cfg = StyleCfg(show_detail_panel=True)

    for fmt in ["pdf", "png", "svg"]:
        print(f"Generating {fmt.upper()}...")
        g = build_arm_flowchart(format=fmt, s=cfg)
        g.render(filename="arm_flowchart_publication", cleanup=True)
        print(f"  ✓ arm_flowchart_publication.{fmt}")

    print("\nDone!")
