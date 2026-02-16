#!/usr/bin/env python3
"""
Comparison of causal estimates with and without joint selection model.

Runs the full pipeline twice:
    - use_joint_selection = False (standard TestingModel)
    - use_joint_selection = True  (JointSelectionModel)
Then compares the risk difference estimates and saves a joint forest plot.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add the directory containing this script to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.controllers.DataLoader import DataLoader
from src.controllers.filters.FilteringStrategy import FilterConfig
from src.controllers.escalation_causal.config.settings import (
    RunConfig,
    SplitConfig,
    CovariateConfig,
    PolicyConfig,
    NuisanceConfig,
    TMLEConfig,
)
from src.controllers.escalation_causal.pipeline import CausalPipeline
from src.controllers.escalation_causal.screening.phase1_screener import Phase1Screener, Phase1Config
from src.controllers.escalation_causal.utils.io import save_results

# Visualization
try:
    from src.controllers.escalation_causal.visualization.forest import ForestPlot
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Visualization module not available – plots will be skipped.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_with_flag(use_joint: bool, config_template: RunConfig, df, flags, all_codes, pairs, output_subdir: str) -> pd.DataFrame:
    """Run pipeline with given joint selection flag."""
    # Create a copy of config and modify the flag
    config_dict = config_template.model_dump()
    config_dict["nuisance"]["use_joint_selection"] = use_joint
    modified_config = RunConfig(**config_dict)

    pipeline = CausalPipeline(modified_config, n_jobs=4)
    results = pipeline.run(df, flags, all_codes, pairs)

    # Save intermediate results
    out_dir = Path(output_subdir)
    save_results(
        out_dir,
        results,
        modified_config.model_dump(),
        cohort_meta={"note": f"use_joint_selection={use_joint}"},
    )
    logger.info(f"Results for use_joint_selection={use_joint} saved to {out_dir}")
    return results


def compare_results(res_false: pd.DataFrame, res_true: pd.DataFrame) -> pd.DataFrame:
    """Merge results from two runs for comparison."""
    # Keep only successful estimates
    false_ok = res_false[res_false["status"] == "ok"][["trigger", "target", "rd", "ci_low", "ci_high"]].copy()
    true_ok = res_true[res_true["status"] == "ok"][["trigger", "target", "rd", "ci_low", "ci_high"]].copy()

    false_ok.rename(columns={"rd": "rd_false", "ci_low": "ci_low_false", "ci_high": "ci_high_false"}, inplace=True)
    true_ok.rename(columns={"rd": "rd_true", "ci_low": "ci_low_true", "ci_high": "ci_high_true"}, inplace=True)

    merged = pd.merge(false_ok, true_ok, on=["trigger", "target"], how="inner")
    merged["rd_diff"] = merged["rd_true"] - merged["rd_false"]
    merged["pair"] = merged["trigger"] + " → " + merged["target"]
    return merged


def plot_comparison(merged: pd.DataFrame, output_dir: Path):
    """Generate comparison plot."""
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Visualization not available – skipping plot.")
        return

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Sort by absolute difference
    merged = merged.sort_values("rd_diff", ascending=False).reset_index(drop=True)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Risk Difference Estimates", "Difference (Joint – Standard)"),
        shared_xaxes=True,
        vertical_spacing=0.15,
    )

    # Forest plot style for the two estimates
    y_pos = list(range(len(merged)))
    fig.add_trace(go.Scatter(
        x=merged["rd_false"],
        y=y_pos,
        mode="markers",
        marker=dict(color="blue", size=8),
        name="Standard",
        error_x=dict(
            type="data",
            symmetric=False,
            array=merged["rd_false"] - merged["ci_low_false"],
            arrayminus=merged["ci_high_false"] - merged["rd_false"],
            color="blue",
            thickness=1,
        ),
        showlegend=True,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=merged["rd_true"],
        y=y_pos,
        mode="markers",
        marker=dict(color="red", size=8),
        name="Joint",
        error_x=dict(
            type="data",
            symmetric=False,
            array=merged["rd_true"] - merged["ci_low_true"],
            arrayminus=merged["ci_high_true"] - merged["rd_true"],
            color="red",
            thickness=1,
        ),
        showlegend=True,
    ), row=1, col=1)

    # Difference plot
    fig.add_trace(go.Bar(
        x=merged["rd_diff"],
        y=y_pos,
        orientation="h",
        marker_color=["red" if d > 0 else "blue" for d in merged["rd_diff"]],
        name="Difference",
        showlegend=False,
    ), row=2, col=1)

    # Add vertical line at zero
    fig.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_yaxes(tickvals=y_pos, ticktext=merged["pair"], row=1, col=1)
    fig.update_yaxes(tickvals=y_pos, ticktext=merged["pair"], row=2, col=1)

    fig.update_layout(
        height=800,
        width=1000,
        title_text="Comparison: Standard vs. Joint Selection Model",
        showlegend=True,
    )

    # Save
    fig.write_html(output_dir / "comparison_plot.html")
    fig.write_image(output_dir / "comparison_plot.png", width=1000, height=800, scale=2)
    fig.write_image(output_dir / "comparison_plot.pdf")
    logger.info(f"Comparison plot saved to {output_dir}")


def main():
    # ------------------------------------------------------------------
    # 1. Load and filter data (same as in run_analysis.py)
    # ------------------------------------------------------------------
    data_path = "./datasets/structured/dataset_parquet"
    filter_config_path = "./src/controllers/filters/config_all_klebsiella.json"

    loader = DataLoader(data_path, strict=False, normalize_on_load=True)
    filter_config = FilterConfig.from_json(filter_config_path)

    df, meta = loader.get_cohort(
        filter_config=filter_config,
        apply_exclusions=True,
        verbose=True,
    )
    logger.info(f"Cohort loaded: {meta.n_rows} rows from {meta.n_labs} labs")

    all_codes = sorted(loader.code_to_base.keys())
    logger.info(f"Total antibiotic codes: {len(all_codes)}")

    flags = loader.get_abx_flags(
        df,
        codes=all_codes,
        recode_mode="R_vs_nonR",
        drop_I=True,
    )

    # ------------------------------------------------------------------
    # 2. Phase 1 screening (same as before)
    # ------------------------------------------------------------------
    phase1_cfg = Phase1Config(
        min_group=50,
        min_trigger_tested=100,
        crude_screening_threshold=0.05,
        fdr_alpha=0.05,
        exclude_targets_equal_trigger=True,
    )
    screener = Phase1Screener(phase1_cfg)

    phase1_df = screener.run(
        df=df,
        flags=flags,
        all_codes=all_codes,
        top_n=100,
    )

    if phase1_df.empty:
        logger.error("No pairs passed Phase 1 screening. Exiting.")
        return

    pairs = list(zip(phase1_df["trigger"], phase1_df["target"]))
    logger.info(f"Selected {len(pairs)} pairs from Phase 1 screening")

    # ------------------------------------------------------------------
    # 3. Base configuration (without joint selection flag)
    # ------------------------------------------------------------------
    base_config = RunConfig(
        split=SplitConfig(
            test_size=0.3,
            split_group_col="Anonymized_Lab",
            random_state=42,
        ),
        covariates=CovariateConfig(
            covariate_cols=[
                "Anonymized_Lab",
                "ARS_WardType",
                "AgeGroup",
                "Year",
            ],
            min_count=200,
            max_levels=25,
            drop_first=True,
        ),
        policy=PolicyConfig(
            context_cols=["Anonymized_Lab", "PathogengroupL1", "Year"],
            method="empirical",
            min_context_n=100,
            model_type="xgb",
            calibrate=True,
            calibration_method="isotonic",
            calibration_cv=5,
        ),
        nuisance=NuisanceConfig(
            testing_model="xgb",
            propensity_model="xgb",
            outcome_model="xgb",
            calibrate_testing=True,
            calibrate_propensity=False,
            calibrate_outcome=False,
            testing_cv_folds=5,
            random_state=42,
            use_joint_selection=False,  # will be overridden in runs
        ),
        tmle=TMLEConfig(
            n_folds=5,
            min_prob=0.01,
            weight_cap_percentile=99.0,
            min_tested=200,
            min_group=50,
            stabilize_weights=True,
            n_bootstrap=None,
            alpha=0.05,
        ),
    )

    # ------------------------------------------------------------------
    # 4. Run both configurations
    # ------------------------------------------------------------------
    output_base = Path("./comparison_output")
    output_base.mkdir(exist_ok=True)

    # Without joint selection
    logger.info("=== Running WITHOUT joint selection ===")
    res_false = run_with_flag(False, base_config, df, flags, all_codes, pairs, output_base / "run_false")

    # With joint selection
    logger.info("=== Running WITH joint selection ===")
    res_true = run_with_flag(True, base_config, df, flags, all_codes, pairs, output_base / "run_true")

    # ------------------------------------------------------------------
    # 5. Compare and plot
    # ------------------------------------------------------------------
    merged = compare_results(res_false, res_true)
    merged.to_csv(output_base / "comparison_merged.csv", index=False)
    logger.info(f"Merged comparison saved with {len(merged)} pairs.")

    plot_comparison(merged, output_base)
    logger.info("Comparison complete.")


if __name__ == "__main__":
    main()