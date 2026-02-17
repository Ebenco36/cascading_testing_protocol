#!/usr/bin/env python3
"""
Multi‑pathogen analysis script for antibiotic escalation.

For each pathogen genus (specified by a filter config JSON file):
  - Loads the full dataset, applies the genus‑specific filter.
  - Splits into discovery and estimation sets (by laboratory).
  - Performs Phase 1 screening on discovery set.
  - Runs the causal pipeline on estimation set.
  - Applies Bayesian shrinkage and generates plots.
  - Runs heterogeneity analysis (causal forest) for the top pair (if test data stored).
  - Computes yearly time‑trend estimates and plots.
  - Exports cascade dependencies.
  - Saves all results in a genus‑specific subdirectory.

Usage:
  python run_analysis_multipathogen.py
"""

import sys
import re
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

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
from src.controllers.escalation_causal.utils.io import save_results
from src.controllers.escalation_causal.pipeline import CausalPipeline
from src.controllers.escalation_causal.heterogeneity.causal_forest import CausalForestWrapper
from src.controllers.escalation_causal.sensitivity.cinelli_hazlett import CinelliHazlettSensitivity
from src.controllers.escalation_causal.screening.phase1_screener import Phase1Screener, Phase1Config
from src.controllers.escalation_causal.multiple_comparison.bayesian_shrinkage import BayesianShrinkage
from src.controllers.escalation_causal.export.co_resistance_exporter import EscalationExporter

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helper: run pipeline for a specific year (used for time‑trend)
# ----------------------------------------------------------------------
def run_pipeline_for_year(
    df: pd.DataFrame,
    flags: pd.DataFrame,
    all_codes: list,
    pairs: list,
    year: int,
    config_template: RunConfig,
) -> pd.DataFrame:
    """Run the causal pipeline on data from a single year."""
    year_mask = df["Year"] == year
    df_year = df[year_mask].copy()
    flags_year = flags[year_mask].copy()
    if df_year.empty:
        return pd.DataFrame()
    pipeline = CausalPipeline(config_template, n_jobs=1)
    results = pipeline.run(df_year, flags_year, all_codes, pairs)
    results["year"] = year
    return results


# ----------------------------------------------------------------------
# Helper to get feature names from covariate config (simplified)
# ----------------------------------------------------------------------
def get_covariate_names(config: CovariateConfig, df: pd.DataFrame) -> list:
    """Return list of one‑hot encoded feature names (dummy)."""
    # In practice you would use a proper encoder that stores feature names.
    # Here we just return a placeholder.
    return [f"cov_{i}" for i in range(10)]  # adjust as needed


# ----------------------------------------------------------------------
# generate_failure_summary (unchanged)
# ----------------------------------------------------------------------
def generate_failure_summary(results: pd.DataFrame, output_dir: Path):
    """Extract and save summary of failed pairs with reasons and group sizes."""
    failed = results[results["status"] == "failed"].copy()
    if failed.empty:
        logger.info("No failed pairs.")
        return

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


# ----------------------------------------------------------------------
# generate_plots (unchanged)
# ----------------------------------------------------------------------
def generate_plots(results: pd.DataFrame, output_dir: Path, phase1_df: pd.DataFrame = None,
                   policy=None, flags_val=None, df_val=None):
    """Generate and save all publication plots."""
    if results.empty:
        logger.warning("No results to plot.")
        return

    valid = results.dropna(subset=["rd", "ci_low", "ci_high"])
    if valid.empty:
        logger.warning("No valid estimates to plot.")
    else:
        logger.info(f"Generating plots for {len(valid)} valid pairs.")

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # 1. Forest plot
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

    # 2. Sensitivity contour for top pair
    if not valid.empty:
        try:
            top_idx = valid["rd"].abs().idxmax()
            row = valid.loc[top_idx]
            dof = row["n_used"] - 2
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

    # 3. Network plot
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

    # 4. Heatmap
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

    # 5. Scatter OR vs RD
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

    # 6. Calibration plots (if policy and validation data provided)
    if policy is not None and flags_val is not None and df_val is not None:
        try:
            targets_to_plot = ["CIP", "MER", "GEN"]
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


# ----------------------------------------------------------------------
# Main function for one pathogen
# ----------------------------------------------------------------------
def analyze_one_pathogen(
    loader: DataLoader,
    filter_config_path: Path,
    base_output_dir: Path,
    base_config: RunConfig,
):
    """Run the full analysis for a single pathogen genus."""
    logger.info(f"\n{'='*60}\nProcessing pathogen config: {filter_config_path.name}\n{'='*60}")

    pathogen_name = filter_config_path.stem.replace("config_", "").replace("_", "").lower()
    output_dir = base_output_dir / pathogen_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pathogen‑specific filter
    filter_config = FilterConfig.from_json(str(filter_config_path))

    # Apply filter to get cohort
    df, meta = loader.get_cohort(
        filter_config=filter_config,
        apply_exclusions=True,
        verbose=True,
    )
    if len(df) == 0:
        logger.error(f"No isolates remaining for {filter_config_path.name}. Skipping.")
        return

    logger.info(f"Cohort loaded: {meta.n_rows} rows from {meta.n_labs} labs")
    all_codes = sorted(loader.code_to_base.keys())
    flags = loader.get_abx_flags(df, codes=all_codes, recode_mode="R_vs_nonR", drop_I=True)

    # ------------------------------------------------------------------
    # Split into discovery and estimation sets
    # ------------------------------------------------------------------
    group_col = "Anonymized_Lab"
    groups = df[group_col].astype(str).fillna("NA").to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    discovery_idx, estimation_idx = next(gss.split(df.index, groups=groups))

    discovery_df = df.iloc[discovery_idx].copy()
    discovery_flags = flags.iloc[discovery_idx].copy()
    estimation_df = df.iloc[estimation_idx].copy()
    estimation_flags = flags.iloc[estimation_idx].copy()

    logger.info(f"Discovery set: {len(discovery_df)} isolates from {discovery_df[group_col].nunique()} labs")
    logger.info(f"Estimation set: {len(estimation_df)} isolates from {estimation_df[group_col].nunique()} labs")

    # ------------------------------------------------------------------
    # Phase 1 screening on discovery set
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
        df=discovery_df,
        flags=discovery_flags,
        all_codes=all_codes,
        top_n=100,
    )

    if phase1_df.empty:
        logger.error("No pairs passed Phase 1 screening. Exiting for this pathogen.")
        return

    pairs = list(zip(phase1_df["trigger"], phase1_df["target"]))
    logger.info(f"Selected {len(pairs)} pairs from Phase 1 screening")

    # ------------------------------------------------------------------
    # Run pipeline on estimation set
    # ------------------------------------------------------------------
    config = base_config  # already a RunConfig object
    pipeline = CausalPipeline(config, n_jobs=4)
    results = pipeline.run(
        df=estimation_df,
        flags=estimation_flags,
        all_codes=all_codes,
        pairs=pairs,
    )

    # Expose fitted policy (optional, for calibration plots)
    pipeline.policy_ = pipeline._policy

    # ------------------------------------------------------------------
    # Save results and failure summary
    # ------------------------------------------------------------------
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
    generate_failure_summary(results, output_dir)

    # ------------------------------------------------------------------
    # Export filtered cascade results
    # ------------------------------------------------------------------
    exporter = EscalationExporter(escalation_df=results)
    exporter.filter(
        p_threshold=0.05,
        rd_min=0.05,
        effect_direction="positive",
        min_ess=50,
        keep_failed=False,
    )
    export = exporter.export(str(output_dir), prefix=f"{pathogen_name}_escalation")
    logger.info(f"Exported {export.metadata['stats']['n_pairs']} cascade edges.")

    # ------------------------------------------------------------------
    # Bayesian shrinkage for multiple comparisons
    # ------------------------------------------------------------------
    shrinkage_df = results[results["status"] == "ok"].copy()
    shrinkage_df["pair"] = shrinkage_df["trigger"] + " → " + shrinkage_df["target"]
    shrinkage_df = shrinkage_df.dropna(subset=["rd", "se"])

    if len(shrinkage_df) > 1:
        bs = BayesianShrinkage(random_seed=42, draws=2000, tune=1000, target_accept=0.99)
        bs.fit(shrinkage_df, estimate_col="rd", se_col="se")
        shrunk_summary = bs.summary()
        shrunk_summary.to_csv(output_dir / "bayesian_shrinkage_summary.csv", index=False)

        # Posterior probability plot
        prob_plot = PosteriorProbabilityPlot(
            shrunk_summary,
            pair_col="pair",
            prob_col="prob_positive",
            title="Posterior Probability of Positive Escalation Effect",
        )
        prob_plot.save(output_dir / "posterior_prob.png", width=1000, height=800, scale=2)
        prob_plot.save(output_dir / "posterior_prob.pdf")

        # Shrinkage forest plot
        shrink_forest = ShrinkageForestPlot(
            shrunk_summary,
            original_est_col="rd",
            original_se_col="se",
            shrunken_est_col="theta_shrunken",
            shrunken_sd_col="theta_sd",
            pair_col="pair",
            title="Bayesian Shrinkage: Original vs. Shrunken Estimates",
        )
        shrink_forest.save(output_dir / "shrinkage_forest.png", width=1200, height=900, scale=2)
        shrink_forest.save(output_dir / "shrinkage_forest.pdf")
        shrink_forest.save(output_dir / "shrinkage_forest.html")
        logger.info("Bayesian shrinkage plots saved.")
    else:
        logger.warning("Not enough valid pairs for Bayesian shrinkage.")

    # ------------------------------------------------------------------
    # Heterogeneity analysis with causal forest (for the top pair)
    # ------------------------------------------------------------------
    if not results[results["status"] == "ok"].empty:
        valid = results[results["status"] == "ok"].copy()
        top_idx = valid["rd"].abs().idxmax()
        top_row = valid.loc[top_idx]
        trigger, target = top_row["trigger"], top_row["target"]
        logger.info(f"Running causal forest for top pair: {trigger} → {target}")

        # Check if pipeline stored test data (requires modification in pipeline.py)
        if hasattr(pipeline, "df_test") and hasattr(pipeline, "flags_test") and hasattr(pipeline, "esc_scores"):
            df_test = pipeline.df_test
            flags_test = pipeline.flags_test
            esc_scores = pipeline.esc_scores

            T_col = f"{trigger}_T"
            if T_col not in flags_test.columns:
                logger.error(f"Trigger column {T_col} missing in test flags.")
            else:
                tested_mask = flags_test[T_col].astype(int).to_numpy() == 1
                if tested_mask.sum() == 0:
                    logger.error("No tested isolates for trigger in test set.")
                else:
                    X = pipeline._encode_covariates(df_test, config.covariates.covariate_cols)
                    X_tested = X[tested_mask]

                    A = flags_test[f"{trigger}_R"].astype(int).to_numpy()[tested_mask]
                    Y = esc_scores[target][tested_mask]

                    # Testing model weights (recompute)
                    if config.nuisance.use_joint_selection:
                        from src.controllers.escalation_causal.nuisance.joint_selection import JointSelectionModel
                        joint_model = JointSelectionModel(min_prob=config.tmle.min_prob)
                        T_full = flags_test[T_col].astype(int).to_numpy()
                        R_full = flags_test[f"{trigger}_R"].astype(int).to_numpy()
                        joint_model.fit(X, T_full, R_full)
                        p_test = joint_model.predict_p_test(X)
                        w_full, _ = joint_model.compute_weights(p_test, tested_mask)
                    else:
                        from src.controllers.escalation_causal.nuisance.testing_model import TestingModel
                        test_model = TestingModel(
                            model_type=config.nuisance.testing_model,
                            calibrate=config.nuisance.calibrate_testing,
                            n_folds_cv=config.nuisance.testing_cv_folds,
                            min_prob=config.tmle.min_prob,
                            weight_cap_percentile=config.tmle.weight_cap_percentile,
                        )
                        test_model.fit(X, T_full)
                        p_test = test_model.get_oof_predictions() if test_model._is_cross_fitted else test_model.predict_proba(X)
                        w_full, _ = test_model.compute_weights(p_test, tested_mask)

                    w = w_full[tested_mask]

                    # Fit causal forest
                    feature_names = get_covariate_names(config.covariates, df_test)
                    cf = CausalForestWrapper(
                        n_estimators=400,
                        max_depth=20,
                        min_samples_leaf=10,
                        random_state=42,
                    )
                    cf.fit(X_tested, A, Y, sample_weight=w, feature_names=feature_names)

                    # Variable importance
                    imp = cf.feature_importances()
                    imp_df = pd.DataFrame(list(imp.items()), columns=["feature", "importance"])
                    imp_df.sort_values("importance", ascending=False).to_csv(output_dir / f"causal_forest_importance_{trigger}_{target}.csv", index=False)

                    # Plot variable importance
                    fig_imp = cf.plot_variable_importance(top_k=15)
                    import matplotlib.pyplot as plt
                    plt.tight_layout()
                    plt.savefig(output_dir / f"causal_forest_importance_{trigger}_{target}.png", dpi=300, bbox_inches="tight")
                    plt.savefig(output_dir / f"causal_forest_importance_{trigger}_{target}.pdf", bbox_inches="tight")
                    plt.close()

                    # CATE distribution
                    fig_hist = cf.plot_cate_distribution(X_tested)
                    plt.tight_layout()
                    plt.savefig(output_dir / f"cate_distribution_{trigger}_{target}.png", dpi=300, bbox_inches="tight")
                    plt.savefig(output_dir / f"cate_distribution_{trigger}_{target}.pdf", bbox_inches="tight")
                    plt.close()

                    # Subgroup summaries
                    ward_labels = df_test.loc[tested_mask, "ARS_WardType"].values
                    if ward_labels is not None:
                        ward_summary = cf.get_cate_summary(X_tested, group_labels=ward_labels)
                        ward_summary.to_csv(output_dir / f"cate_by_ward_{trigger}_{target}.csv", index=False)

                        if not ward_summary.empty:
                            sf = SubgroupForestPlot(
                                ward_summary,
                                subgroup_col="group",
                                estimate_col="mean_cate",
                                ci_low_col="ci_low",
                                ci_high_col="ci_high",
                                title=f"CATE by Ward Type: {trigger} → {target}",
                            )
                            sf.save(output_dir / f"subgroup_forest_ward_{trigger}_{target}.png")
                            sf.save(output_dir / f"subgroup_forest_ward_{trigger}_{target}.pdf")
                            sf.save(output_dir / f"subgroup_forest_ward_{trigger}_{target}.html")

                    age_labels = df_test.loc[tested_mask, "AgeGroup"].values
                    if age_labels is not None:
                        age_summary = cf.get_cate_summary(X_tested, group_labels=age_labels)
                        age_summary.to_csv(output_dir / f"cate_by_age_{trigger}_{target}.csv", index=False)

                        if not age_summary.empty:
                            sf_age = SubgroupForestPlot(
                                age_summary,
                                subgroup_col="group",
                                estimate_col="mean_cate",
                                ci_low_col="ci_low",
                                ci_high_col="ci_high",
                                title=f"CATE by Age Group: {trigger} → {target}",
                            )
                            sf_age.save(output_dir / f"subgroup_forest_age_{trigger}_{target}.png")
                            sf_age.save(output_dir / f"subgroup_forest_age_{trigger}_{target}.pdf")
                            sf_age.save(output_dir / f"subgroup_forest_age_{trigger}_{target}.html")

                    logger.info("Causal forest analysis completed and plots saved.")
        else:
            logger.warning("Pipeline did not store test data – skipping causal forest. "
                           "To enable, add the following to pipeline.py after run:\n"
                           "self.df_test = df_test\n"
                           "self.flags_test = flags_test\n"
                           "self.esc_scores = esc_scores")
    else:
        logger.warning("No valid pairs for heterogeneity analysis.")

    # ------------------------------------------------------------------
    # Generate publication plots (excluding calibration)
    # ------------------------------------------------------------------
    generate_plots(
        results,
        output_dir,
        phase1_df=phase1_df,
        policy=pipeline.policy_ if hasattr(pipeline, "policy_") else None,
        flags_val=None,   # would be pipeline._flags_test if available
        df_val=None,      # would be pipeline._df_test
    )

    # ------------------------------------------------------------------
    # Time trend analysis (optional)
    # ------------------------------------------------------------------
    years = sorted(estimation_df["Year"].dropna().unique())
    yearly_results = []
    for yr in years:
        logger.info(f"Running yearly analysis for {yr}...")
        yr_res = run_pipeline_for_year(estimation_df, estimation_flags, all_codes, pairs, yr, config)
        if not yr_res.empty:
            yearly_results.append(yr_res)

    if yearly_results:
        yearly_df = pd.concat(yearly_results, ignore_index=True)
        yearly_df.to_csv(output_dir / "yearly_results.csv", index=False)

        # For the top pair, create a time trend plot
        if not yearly_df.empty and 'trigger' in locals():
            top_pair_yearly = yearly_df[
                (yearly_df["trigger"] == trigger) & (yearly_df["target"] == target) & (yearly_df["status"] == "ok")
            ].copy()
            if not top_pair_yearly.empty:
                tt = TimeTrendPlot(
                    top_pair_yearly,
                    time_col="year",
                    estimate_col="rd",
                    se_col="se",
                    title=f"Time Trend: {trigger} → {target}",
                )
                tt.save(output_dir / "time_trend.png")
                tt.save(output_dir / "time_trend.pdf")
                tt.save(output_dir / "time_trend.html")
                logger.info("Time trend plot saved.")
    else:
        logger.warning("No yearly results generated.")

    logger.info(f"Analysis for {filter_config_path.name} complete.\n")


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    # Directory containing filter configs
    config_dir = Path("./src/controllers/filters/")
    filter_config_paths = list(config_dir.glob("config_*.json"))
    if not filter_config_paths:
        logger.error("No filter config files found.")
        return

    base_output_dir = Path("./output_multipathogen")
    base_output_dir.mkdir(exist_ok=True)

    # Base pipeline configuration (common to all pathogens)
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

    # Load full dataset once (shared across pathogens)
    data_path = "./datasets/structured/dataset_parquet"
    loader = DataLoader(data_path, strict=False, normalize_on_load=True)

    # Loop over each pathogen config
    for cfg_path in filter_config_paths:
        analyze_one_pathogen(loader, cfg_path, base_output_dir, base_config)

    logger.info("All pathogen analyses completed.")


if __name__ == "__main__":
    main()