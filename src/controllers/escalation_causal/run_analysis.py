#!/usr/bin/env python3
"""
Complete analysis script for antibiotic escalation using the causal pipeline.
Generates publication‑ready plots using Plotly, including advanced visualizations.
"""

import logging
import sys
import re
from pathlib import Path

import pandas as pd
import numpy as np

from src.controllers.escalation_causal.export.co_resistance_exporter import EscalationExporter

# Add the directory containing this script to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.controllers.DataLoader import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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
from src.controllers.escalation_causal.multiple_comparison.bayesian_shrinkage import BayesianShrinkage
from src.controllers.escalation_causal.heterogeneity.causal_forest import CausalForestWrapper
from src.controllers.escalation_causal.sensitivity.cinelli_hazlett import CinelliHazlettSensitivity

# Visualization imports
from src.controllers.escalation_causal.visualization.forest import ForestPlot
from src.controllers.escalation_causal.visualization.sensitivity import SensitivityContour
from src.controllers.escalation_causal.visualization.network import NetworkPlot
from src.controllers.escalation_causal.visualization.heatmap import ClusteredHeatmap
from src.controllers.escalation_causal.visualization.scatter import CorrelationScatter
from src.controllers.escalation_causal.visualization.calibration import CalibrationPlot
from src.controllers.escalation_causal.visualization.posterior import PosteriorProbabilityPlot, ShrinkageForestPlot
from src.controllers.escalation_causal.visualization.time_series import TimeTrendPlot
from src.controllers.escalation_causal.visualization.subgroup import SubgroupForestPlot
from src.controllers.escalation_causal.visualization.combined import CombinedFigure


def generate_failure_summary(results: pd.DataFrame, output_dir: Path):
    """Extract and save summary of failed pairs with reasons and group sizes."""
    failed = results[results["status"] == "failed"].copy()
    if failed.empty:
        logger.info("No failed pairs.")
        return

    # Extract group sizes from skip_reason using regex
    def extract_sizes(reason):
        if pd.isna(reason):
            return (np.nan, np.nan)
        match = re.search(r"A=1 (\d+), A=0 (\d+)", str(reason))
        if match:
            return int(match.group(1)), int(match.group(2))
        return (np.nan, np.nan)

    failed[["n1_failed", "n0_failed"]] = failed["skip_reason"].apply(
        lambda x: pd.Series(extract_sizes(x))
    )
    summary = failed[["trigger", "target", "skip_reason", "n1_failed", "n0_failed"]]
    summary.to_csv(output_dir / "failed_pairs_summary.csv", index=False)
    logger.info(f"Failure summary saved with {len(summary)} pairs.")


def generate_plots(results: pd.DataFrame, output_dir: Path, phase1_df: pd.DataFrame = None, policy=None, flags_val=None, df_val=None):
    """Generate and save all publication plots."""
    if results.empty:
        logger.warning("No results to plot.")
        return

    # Check if there are any valid estimates
    valid = results.dropna(subset=["rd", "ci_low", "ci_high"])
    if valid.empty:
        logger.warning("No valid estimates to plot.")
    else:
        logger.info(f"Generating plots for {len(valid)} valid pairs.")

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # 1. Forest plot of all pairs
    try:
        fp = ForestPlot(
            results,
            estimate_col="rd",
            ci_low_col="ci_low",
            ci_high_col="ci_high",
            title="Risk Difference for All Trigger–Target Pairs",
        )
        fp.save(figures_dir / "forest_plot.png", width=1200, height=800, scale=2)
        fp.save(figures_dir / "forest_plot.pdf")
        fp.save(figures_dir / "forest_plot.html")
        logger.info("Forest plot saved.")
    except Exception as e:
        logger.error(f"Forest plot failed: {e}")

    # 2. Sensitivity contour for the top pair (largest absolute RD)
    if not valid.empty:
        try:
            top_idx = valid["rd"].abs().idxmax()
            row = valid.loc[top_idx]
            dof = row["n_used"] - 2  # approximate degrees of freedom
            sens = CinelliHazlettSensitivity(estimate=row["rd"], se=row["se"], dof=dof)
            contour_data = sens.contour_data(num_points=30)
            rv = sens.robustness_value()

            sc = SensitivityContour(
                contour_data,
                rv_q=rv["rv_q"],
                rv_alpha=rv["rv_alpha"],
                alpha=sens.alpha,
                title=f"Sensitivity: {row['trigger']} → {row['target']}",
            )
            sc.save(figures_dir / "sensitivity_contour.png")
            sc.save(figures_dir / "sensitivity_contour.pdf")
            sc.save(figures_dir / "sensitivity_contour.html")
            logger.info("Sensitivity contour saved.")
        except Exception as e:
            logger.error(f"Sensitivity contour failed: {e}")

    # 3. Network plot of significant pairs (p < 0.05)
    try:
        net = NetworkPlot(
            results,
            threshold=0.05,
            estimate_col="rd",
            p_col="p_value",
            min_effect=0.01,
            height=800,
            width=800,
        )
        net.save(figures_dir / "network.png")
        net.save(figures_dir / "network.pdf")
        net.save(figures_dir / "network.html")
        logger.info("Network plot saved.")
    except Exception as e:
        logger.error(f"Network plot failed: {e}")

    # 4. Clustered heatmap of RD
    try:
        rd_pivot = results.pivot(index="trigger", columns="target", values="rd").fillna(0)
        hm = ClusteredHeatmap(
            rd_pivot,
            title="Escalation Risk Difference Heatmap",
            cluster_rows=True,
            cluster_cols=True,
            show_values=False,
        )
        hm.save(figures_dir / "rd_heatmap.png")
        hm.save(figures_dir / "rd_heatmap.pdf")
        hm.save(figures_dir / "rd_heatmap.html")
        logger.info("Heatmap saved.")
    except Exception as e:
        logger.error(f"Heatmap failed: {e}")

    # 5. Scatter: co‑resistance OR vs RD (if phase1_df provided)
    if phase1_df is not None and not phase1_df.empty:
        try:
            merged = phase1_df.merge(
                results[["trigger", "target", "rd", "se", "p_value"]],
                on=["trigger", "target"],
                how="inner"
            )
            if not merged.empty:
                scat = CorrelationScatter(
                    merged,
                    x_col="or_unadjusted",
                    y_col="rd",
                    label_col="pair",
                    title="Co‑resistance OR vs. Escalation RD",
                    add_trendline=True,
                )
                scat.save(figures_dir / "or_vs_rd_scatter.png")
                scat.save(figures_dir / "or_vs_rd_scatter.pdf")
                scat.save(figures_dir / "or_vs_rd_scatter.html")
                logger.info("Scatter plot saved.")
        except Exception as e:
            logger.error(f"Scatter plot failed: {e}")

    # 6. Calibration plots for a few example targets (if policy provided)
    if policy is not None and flags_val is not None and df_val is not None:
        try:
            # Pick a few targets with enough data
            targets_to_plot = ["CIP", "MER", "GEN"]  # adjust as needed
            for target in targets_to_plot:
                if f"{target}_T" in flags_val.columns:
                    y_true = flags_val[f"{target}_T"].astype(int).to_numpy()
                    y_pred = policy.predict_proba(df_val, target)[:, 1]
                    cal = CalibrationPlot(y_true, y_pred, title=f"Calibration: {target}")
                    cal.save(figures_dir / f"calibration_{target}.png")
                    cal.save(figures_dir / f"calibration_{target}.pdf")
                    cal.save(figures_dir / f"calibration_{target}.html")
                    logger.info(f"Calibration plot for {target} saved.")
        except Exception as e:
            logger.error(f"Calibration plots failed: {e}")

    # 7. Bayesian shrinkage plot (if we run it later, we can add separately)
    # For now, we'll note that it requires post‑processing.

    # 8. Time trend for top pair (if year column exists)
    if "Year" in results.columns and not valid.empty:
        try:
            # For simplicity, assume we have a separate time‑stratified results.
            # Here we'll just show a placeholder.
            logger.info("Time trend plot requires stratified results – skipping.")
        except Exception as e:
            logger.error(f"Time trend failed: {e}")


def main():
    # ------------------------------------------------------------------
    # 1. Load and filter data
    # ------------------------------------------------------------------
    data_path = "./datasets/structured/dataset_parquet"
    filter_config_path = "./src/controllers/filters/config_all_klebsiella.json"

    loader = DataLoader(data_path, strict=False, normalize_on_load=True)

    from src.controllers.filters.FilteringStrategy import FilterConfig
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
    # 2. Phase 1 screening to select promising pairs
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
    # 3. Configure the causal pipeline
    # ------------------------------------------------------------------
    config = RunConfig(
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
            use_joint_selection=True,
            random_state=42,
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
    # 4. Run the causal pipeline with parallel processing
    # ------------------------------------------------------------------
    pipeline = CausalPipeline(config, n_jobs=4)
    results = pipeline.run(
        df=df,
        flags=flags,
        all_codes=all_codes,
        pairs=pairs,
    )

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    output_dir = Path("./output")
    save_results(
        output_dir,
        results,
        config.model_dump(),
        cohort_meta={
            "name": meta.name,
            "n_rows": meta.n_rows,
            "n_labs": meta.n_labs,
            "year_range": f"{meta.yearmonth_min} – {meta.yearmonth_max}",
            "pathogen": meta.pathogen,
        },
    )

    logger.info(f"Results saved to {output_dir}")

    # ------------------------------------------------------------------
    # 6. Generate failure summary
    # ------------------------------------------------------------------
    generate_failure_summary(results, output_dir)
    
    # results is the DataFrame from pipeline.run()
    exporter = EscalationExporter(escalation_df=results)
    exporter.filter(
        p_threshold=0.05,
        rd_min=0.05,
        effect_direction="positive",
        min_ess=50,
        keep_failed=False
    )
    export = exporter.export("./output/", prefix="kleb_escalation")

    print(f"Exported {export.metadata['stats']['n_pairs']} cascade edges.")
    print("Dependencies:", export.dependencies)


    # ------------------------------------------------------------------
    # 7. Generate publication plots
    # ------------------------------------------------------------------
    # We need to pass policy and validation data for calibration plots.
    # Here we use the test split from the pipeline (but we don't have it directly).
    # Instead, we can re‑run the policy on the full data? Better: pipeline stores policy.
    # We can extract the test set from pipeline? Not currently stored.
    # For simplicity, we'll pass None for policy and skip calibration plots.
    # In a real analysis, you might save the test set or policy to disk.
    generate_plots(results, output_dir, phase1_df=phase1_df)

    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()