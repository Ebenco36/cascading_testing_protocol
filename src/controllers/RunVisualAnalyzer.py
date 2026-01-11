import json
import os
import pandas as pd
from src.controllers.ARMRuleCrossVisualizer import ARMRuleCrossVisualizer, CrossVizConfig, _ensure_dir


if __name__ == "__main__":
    rules = pd.read_csv("testRules_normalized.csv")
    # rules = pd.read_csv("rules_2019.0.csv")

    with open("./datasets/antibiotic_class_grouping.json", "r", encoding="utf-8") as f:
        drugclass_json = json.load(f)

    with open("./datasets/antibiotic_class.json", "r", encoding="utf-8") as f:
        aware_json = json.load(f)

    cfg = CrossVizConfig(
        weight_col="Support",
        min_weight=0.0, 
        top_edges=250,
        chord_key_top_n_sectors=18,
        chord_panel_bottom=-0.25,  # increase to -0.25 if you want more room
        chord_panel_height=0.28,  # increase to 0.32 if you want even more room
        chord_figsize=(12.5, 12.5),
    )

    viz = ARMRuleCrossVisualizer(rules, drugclass_json, aware_json=aware_json, cfg=cfg)
    viz.build_cross_edges()

    out_dir = "./outputs_crossviz_publication"
    _ensure_dir(out_dir)

    sankey = viz.plot_sankey("Cross-informative flow: Resistance → Cross-testing (Sankey)")
    viz.save_plotly_all(sankey, out_dir, "01_sankey_cross")

    heat = viz.plot_heatmap("Cross-informative matrix: Resistance → Cross-testing (Heatmap)")
    viz.save_plotly_all(heat, out_dir, "02_heatmap_cross")

    viz.chord_pycirclize_bipartite(
        out_dir=out_dir,
        base_name="03_chord_bipartite_AWaRe_pycirclize_pub_FIXED",
        title="Chord (bipartite): Cross-informative flow (R → T) — colored by AWaRe",
        color_by="aware",
        legend_loc="upper right",
    )

    viz.circos_radar_rt(
        out_dir=out_dir,
        base_name="04_circos_radar_R_outflow_vs_T_inflow_all",
        title="Circos radar: R outflow vs T inflow (all antibiotics) — colored by AWaRe",
        color_by="aware",
        top_n=100000,
        legend_loc="upper right",
    )

    viz.network_publication(
        out_dir=out_dir,
        base_name="05_network_cross_colorAware_bipartite",
        color_by="aware",
        bipartite_rt=True,
        r_label="(R)",
        t_label="(T)",
    )

    print(f"Done. Outputs saved to: {os.path.abspath(out_dir)}")