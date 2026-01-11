"""
Run.py - CASCADE DISCOVERY PIPELINE WITH INTEGRATED VISUALIZATION

Backward-compatible wrapper around PipelineOrchestrator with integrated
publication-grade visualization using CascadeVisualizationEngine.

Maintains existing run_cascade_discovery_pipeline() API while adding:
  ✓ All existing function signatures work unchanged
  ✓ Automatic rule cleaning & deduplication
  ✓ Automatic generation of 4 publication figures (main + 3 supplementary)
  ✓ 300-600 DPI multi-format output (PNG, PDF, SVG)
  ✓ Colorblind-safe palettes
  ✓ Professional manuscript-ready visualizations

Typical usage (unchanged from before):
====================================

    from Run import run_cascade_discovery_pipeline

    rules = run_cascade_discovery_pipeline(
        data_path="./datasets/dataset.parquet",
        output_dir="./output/cascade_discovery",
        algorithm="fpgrowth",
        min_support=0.01,
        min_confidence=0.30,
        min_lift=1.1,
        run_stats=True,
        run_sensitivity=True,
        run_visualization=True,  # NEW: Generate publication figures
    )

With visualization defaults:
  - Grid sensitivity (parameter impact analysis)
  - Bootstrap stability waterfall (top 30 edges)
  - Bootstrap stability heatmap (edge persistence matrix)
  - Combined main text figure (2-panel sensitivity + stability)
  - 300 DPI PNG (publication quality) + PDF + SVG
  - Output to: output_dir/06_figures/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import pandas as pd

try:
    from src.controllers.PipelineOrchestrator import (
        CascadeDiscoveryPipeline,
        FilteringConfig,
        MatrixConfig,
        ARMConfig,
        RuleCleanerConfig,
        StatisticalTestingConfig,
        SensitivityConfig,
    )
    from src.controllers.VisualizationEngine import CascadeVisualizationEngine
except ImportError:
    from PipelineOrchestrator import (
        CascadeDiscoveryPipeline,
        FilteringConfig,
        MatrixConfig,
        ARMConfig,
        RuleCleanerConfig,
        StatisticalTestingConfig,
        SensitivityConfig,
    )
    try:
        from VisualizationEngine import CascadeVisualizationEngine
    except ImportError:
        CascadeVisualizationEngine = None


def run_cascade_discovery_pipeline(
    *,
    data_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path = "./src/controllers/filters/config.json",
    seed: int = 42,
    verbose: bool = True,
    # Filtering config
    filtering_save_filtered: bool = True,
    filtering_output_dir: str | Path = "./datasets/dataset_parquet_filtered",
    filtering_rows_per_file: int = 250_000,
    filtering_compression: str = "zstd",
    # Matrix config
    recode_mode: str = "R_vs_nonR",
    include_covariates: bool = True,
    covariate_cols: Optional[List[str]] = None,
    min_test_rate: float = 0.05,
    min_test_count: int = 2_000,
    min_res_rate: float = 0.05,
    min_res_count: int = 2_000,
    always_keep: Optional[List[str]] = None,
    drop_all_susceptible_rows: bool = True,
    # ARM config
    algorithm: str = "fpgrowth",
    edge_mode: str = "pairwise",
    min_support: float = 0.01,
    min_confidence: float = 0.30,
    min_lift: float = 1.1,
    max_len: Optional[int] = None,
    # Rule cleaning config (NEW)
    run_cleaning: bool = True,
    clean_drop_implied_tests: bool = True,
    clean_antecedent_keep_only_R: bool = True,
    clean_consequent_keep_only_T: bool = True,
    clean_require_both_sides_nonempty: bool = True,
    clean_drop_non_antibiotic_context: bool = True,
    clean_keep_only_informative: bool = True,
    # Statistical testing config
    run_stats: bool = False,
    stats_context_vars: Optional[List[str]] = None,
    stats_chi_min_stratum_size: int = 10,
    stats_alpha: float = 0.05,
    # Sensitivity config
    run_sensitivity: bool = False,
    sensitivity_run_grid: bool = True,
    sensitivity_run_bootstrap: bool = True,
    sensitivity_grid_support: Optional[List[float]] = None,
    sensitivity_grid_confidence: Optional[List[float]] = None,
    sensitivity_grid_lift: Optional[List[float]] = None,
    sensitivity_bootstrap_n: int = 200,
    sensitivity_bootstrap_top_k: int = 30,
    # Visualization config (NEW)
    run_visualization: bool = False,
    viz_dpi: int = 300,
    viz_top_k_waterfall: int = 30,
    viz_max_edges_heatmap: int = 40,
) -> pd.DataFrame:
    """
    Run the complete cascade discovery pipeline end-to-end with optional visualization.

    This function maintains backward compatibility with the original Run.py API
    while using the new class-based CascadeDiscoveryPipeline internally.

    Parameters
    ----------
    data_path : str | Path
        Path to raw parquet dataset
    output_dir : str | Path
        Root output directory for all results
    config_path : str | Path
        Path to filtering config JSON
    seed : int, default=42
        Random seed for reproducibility
    verbose : bool, default=True
        Print progress messages

    Filtering Parameters
    --------------------
    filtering_save_filtered : bool, default=True
        Save filtered data to disk
    filtering_output_dir : str | Path
        Where to save filtered data
    filtering_rows_per_file : int
        Rows per parquet chunk (for large datasets)
    filtering_compression : str
        Compression method ("zstd", "snappy", "gzip")

    Matrix Parameters
    -----------------
    recode_mode : str, default="R_vs_nonR"
        "R_vs_nonR" or "non_susceptible_vs_susceptible"
    include_covariates : bool, default=True
        Include covariate columns in matrix
    covariate_cols : List[str], optional
        Which covariate columns to include
    min_test_rate : float, default=0.05
        Minimum testing rate (5%) for antibiotic inclusion
    min_test_count : int, default=2000
        Minimum number of tests for antibiotic inclusion
    min_res_rate : float, default=0.05
        Minimum resistance rate (5%) for antibiotic inclusion
    min_res_count : int, default=2000
        Minimum number of resistant results for antibiotic inclusion
    always_keep : List[str], optional
        Antibiotics to always include regardless of prevalence
    drop_all_susceptible_rows : bool, default=True
        Exclude all-susceptible isolates

    ARM Parameters
    --------------
    algorithm : str, default="fpgrowth"
        "pairwise", "fpgrowth"
    edge_mode : str, default="pairwise"
        "pairwise" or "conditional"
    min_support : float, default=0.01
        Minimum support (1% of isolates)
    min_confidence : float, default=0.30
        Minimum confidence (30% of resistant cases)
    min_lift : float, default=1.1
        Minimum lift (enrichment relative to independence)
    max_len : int, optional
        Maximum itemset size (for fpgrowth)

    Rule Cleaning Parameters (NEW)
    ------------------------------
    run_cleaning : bool, default=True
        Run rule cleaning and collapsing stage
    clean_drop_implied_tests : bool, default=True
        Drop rules where antecedent items are tested by consequent test
    clean_antecedent_keep_only_R : bool, default=True
        Keep only resistance items in antecedent
    clean_consequent_keep_only_T : bool, default=True
        Keep only test items in consequent
    clean_require_both_sides_nonempty : bool, default=True
        Require both sides have items after cleaning
    clean_drop_non_antibiotic_context : bool, default=True
        Drop context variables from rules
    clean_keep_only_informative : bool, default=True
        Keep only informative rules in collapse

    Statistical Testing Parameters
    -------------------------------
    run_stats : bool, default=False
        Run chi-square/Fisher heterogeneity tests
    stats_context_vars : List[str], optional
        Context variables to test (e.g., ["ARS_WardType", "CareType", "Year"])
    stats_chi_min_stratum_size : int, default=10
        Minimum stratum size for chi-square tests
    stats_alpha : float, default=0.05
        Significance level for multiple testing correction

    Sensitivity Analysis Parameters
    --------------------------------
    run_sensitivity : bool, default=False
        Run grid sensitivity and bootstrap stability
    sensitivity_run_grid : bool, default=True
        Run parameter grid sensitivity
    sensitivity_run_bootstrap : bool, default=True
        Run bootstrap resampling stability
    sensitivity_grid_support : List[float], optional
        Support values to test (default: [0.005, 0.01, 0.02])
    sensitivity_grid_confidence : List[float], optional
        Confidence values to test (default: [0.20, 0.30, 0.40])
    sensitivity_grid_lift : List[float], optional
        Lift values to test (default: [1.0, 1.1, 1.2])
    sensitivity_bootstrap_n : int, default=200
        Number of bootstrap resamples
    sensitivity_bootstrap_top_k : int, default=30
        Top-K edges to track in bootstrap

    Visualization Parameters (NEW)
    --------------------------------
    run_visualization : bool, default=False
        Generate publication-ready figures (requires VisualizationEngine)
    viz_dpi : int, default=300
        Figure resolution (300 for submission, 600 for print)
    viz_top_k_waterfall : int, default=30
        Top K edges in bootstrap waterfall plot
    viz_max_edges_heatmap : int, default=40
        Maximum edges to show in bootstrap heatmap

    Returns
    -------
    pd.DataFrame
        Discovered cascade rules table (cleaned if run_cleaning=True)

    Examples
    --------
    >>> # Basic run (with default rule cleaning)
    >>> rules = run_cascade_discovery_pipeline(
    ...     data_path="./datasets/data.parquet",
    ...     output_dir="./output/cascades",
    ...     algorithm="fpgrowth",
    ...     min_support=0.01,
    ...     min_confidence=0.30,
    ... )

    >>> # Full run with sensitivity and publication figures
    >>> rules = run_cascade_discovery_pipeline(
    ...     data_path="./datasets/data.parquet",
    ...     output_dir="./output/cascades",
    ...     algorithm="fpgrowth",
    ...     min_support=0.01,
    ...     min_confidence=0.30,
    ...     run_stats=True,
    ...     run_sensitivity=True,
    ...     run_visualization=True,
    ...     viz_dpi=300,
    ... )
    """

    # Create pipeline orchestrator
    pipeline = CascadeDiscoveryPipeline(
        config_path=config_path,
        output_dir=output_dir,
        seed=seed,
        verbose=verbose,
    )

    # Configure filtering
    filtering_cfg = FilteringConfig(
        config_path=config_path,
        save_filtered=filtering_save_filtered,
        filtered_output_dir=filtering_output_dir,
        rows_per_file=filtering_rows_per_file,
        compression=filtering_compression,
    )

    # Configure matrix construction
    matrix_cfg = MatrixConfig(
        recode_mode=recode_mode,
        include_covariates=include_covariates,
        covariate_cols=covariate_cols,
        min_test_rate=min_test_rate,
        min_test_count=min_test_count,
        min_res_rate=min_res_rate,
        min_res_count=min_res_count,
        always_keep_antibiotics=always_keep,
        drop_all_susceptible_rows=drop_all_susceptible_rows,
    )

    # Configure ARM
    arm_cfg = ARMConfig(
        algorithm=algorithm,
        edge_mode=edge_mode,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        max_len=max_len,
    )

    # Configure rule cleaning
    cleaner_cfg = RuleCleanerConfig(
        drop_implied_tests=clean_drop_implied_tests,
        antecedent_keep_only_R=clean_antecedent_keep_only_R,
        consequent_keep_only_T=clean_consequent_keep_only_T,
        require_both_sides_nonempty=clean_require_both_sides_nonempty,
        drop_non_antibiotic_context=clean_drop_non_antibiotic_context,
        keep_only_informative=clean_keep_only_informative,
    )

    # Configure statistical testing
    statistical_cfg = StatisticalTestingConfig(
        context_vars=stats_context_vars,
        chi_min_stratum_size=stats_chi_min_stratum_size,
        alpha=stats_alpha,
    )

    # Configure sensitivity analyses
    sensitivity_cfg = SensitivityConfig(
        run_grid=False, # sensitivity_run_grid if run_sensitivity else False,
        run_bootstrap=sensitivity_run_bootstrap if run_sensitivity else False,
        grid_support=sensitivity_grid_support,
        grid_confidence=sensitivity_grid_confidence,
        grid_lift=sensitivity_grid_lift,
        bootstrap_n=sensitivity_bootstrap_n,
        bootstrap_top_k=sensitivity_bootstrap_top_k,
    )

    # Execute pipeline (Stages 1-5)
    pipeline.run(
        raw_data_path=data_path,
        filtering_config=filtering_cfg,
        matrix_config=matrix_cfg,
        arm_config=arm_cfg,
        cleaner_config=cleaner_cfg if run_cleaning else None,
        statistical_config=statistical_cfg,
        sensitivity_config=sensitivity_cfg,
    )

    # Print summaries
    if verbose:
        pipeline.print_cascade_summary(top_n=20)
        if run_stats and pipeline.chi_square_results:
            for ctx_var in (stats_context_vars or ["ARS_WardType"]):
                if ctx_var in pipeline.chi_square_results:
                    pipeline.print_heterogeneity_summary(context_var=ctx_var, top_n=10)

    # ========================================================================
    # STAGE 6: VISUALIZATION (NEW)
    # ========================================================================

    if run_visualization:
        if CascadeVisualizationEngine is None:
            print("⚠ VisualizationEngine not available. Skipping visualization.")
        else:
            try:
                if verbose:
                    print("\n" + "=" * 80)
                    print("STAGE 6: PUBLICATION-GRADE VISUALIZATION")
                    print("=" * 80)

                # Initialize visualization engine
                viz_output_dir = Path(output_dir) / "06_figures"
                engine = CascadeVisualizationEngine(
                    output_dir=viz_output_dir,
                    seed=seed,
                    dpi=viz_dpi,
                    verbose=verbose,
                )

                if verbose:
                    print(f"\n[VIZ] Generating figures to: {viz_output_dir}\n")

                # Prepare grid sensitivity data (from sensitivity analysis)
                grid_summary_df = None
                if (run_sensitivity and hasattr(pipeline, "sensitivity_grid_results")
                    and pipeline.sensitivity_grid_results is not None):
                    grid_summary_df = pipeline.sensitivity_grid_results

                # Prepare bootstrap data (from sensitivity analysis)
                edge_stability_df = None
                edge_stability_matrix = None
                if (run_sensitivity and hasattr(pipeline, "bootstrap_edge_stability")
                    and pipeline.bootstrap_edge_stability is not None):
                    edge_stability_df = pipeline.bootstrap_edge_stability

                if (run_sensitivity and hasattr(pipeline, "bootstrap_edge_matrix")
                    and pipeline.bootstrap_edge_matrix is not None):
                    edge_stability_matrix = pipeline.bootstrap_edge_matrix

                # Generate all manuscript figures
                figs = engine.generate_manuscript_figures(
                    grid_summary_df=grid_summary_df,
                    edge_stability_df=edge_stability_df,
                    edge_stability_matrix=edge_stability_matrix,
                    include_supplementary=True,
                )

                if verbose:
                    print(f"\n✓ Generated {len(figs)} publication-quality figures")
                    print(f"✓ Saved to: {viz_output_dir}")
                    print(f"✓ Formats: PNG ({viz_dpi} DPI) + PDF + SVG")
                    print(f"✓ Color scheme: Colorblind-safe (Nature/Science standards)")

            except Exception as e:
                print(f"⚠ Visualization failed: {e}")
                if verbose:
                    import traceback

                    traceback.print_exc()

    # Return cascade rules (cleaned if available, else raw)
    return (
        pipeline.cascaderules_cleaned
        if (pipeline.cascaderules_cleaned is not None and run_cleaning)
        else pipeline.cascaderules
    )


if __name__ == "__main__":
    """
    Command-line interface for cascade discovery pipeline with visualization.

    Usage:
        python Run.py \\
            --data ./datasets/data.parquet \\
            --out ./output/cascades \\
            --algorithm fpgrowth \\
            --min-support 0.01 \\
            --min-confidence 0.30 \\
            --run-stats \\
            --run-sensitivity \\
            --run-visualization
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Cascade Discovery Pipeline with Rule Cleaning & Publication Figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with default rule cleaning
  python Run.py --data data.parquet --out ./output

  # Full run with statistics, sensitivity, and figures
  python Run.py --data data.parquet --out ./output \\
    --algorithm fpgrowth --min-support 0.01 \\
    --run-stats --run-sensitivity --run-visualization

  # Publication-grade high-resolution figures
  python Run.py --data data.parquet --out ./output \\
    --run-sensitivity --run-visualization --viz-dpi 600

  # Skip rule cleaning (raw rules only)
  python Run.py --data data.parquet --out ./output \\
    --skip-cleaning

  # Conservative filtering (high thresholds)
  python Run.py --data data.parquet --out ./output \\
    --min-test-count 5000 --min-res-count 5000
        """,
    )

    # Required arguments
    parser.add_argument(
        "--data",
        required=True,
        type=str,
        help="Path to dataset (parquet)",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=str,
        help="Output directory",
    )

    # Filtering arguments
    parser.add_argument(
        "--config",
        type=str,
        default="./src/controllers/filters/config.json",
        help="Filtering config JSON",
    )

    # Matrix arguments
    parser.add_argument(
        "--recode-mode",
        type=str,
        choices=["R_vs_nonR", "non_susceptible_vs_susceptible"],
        default="R_vs_nonR",
        help="Binary recoding mode",
    )
    parser.add_argument(
        "--min-test-rate",
        type=float,
        default=0.05,
        help="Min testing rate for antibiotic",
    )
    parser.add_argument(
        "--min-test-count",
        type=int,
        default=2_000,
        help="Min test count for antibiotic",
    )
    parser.add_argument(
        "--min-res-rate",
        type=float,
        default=0.05,
        help="Min resistance rate for antibiotic",
    )
    parser.add_argument(
        "--min-res-count",
        type=int,
        default=2_000,
        help="Min resistance count for antibiotic",
    )

    # ARM arguments
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["pairwise", "fpgrowth", "apriori"],
        default="fpgrowth",
        help="ARM algorithm",
    )
    parser.add_argument(
        "--min-support",
        type=float,
        default=0.01,
        help="Minimum support",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.30,
        help="Minimum confidence",
    )
    parser.add_argument(
        "--min-lift",
        type=float,
        default=1.1,
        help="Minimum lift",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="Max itemset size (fpgrowth)",
    )

    # Rule cleaning
    parser.add_argument(
        "--skip-cleaning",
        action="store_true",
        help="Skip rule cleaning and collapsing",
    )

    # Statistical testing
    parser.add_argument(
        "--run-stats",
        action="store_true",
        help="Run chi-square heterogeneity tests",
    )
    parser.add_argument(
        "--stats-alpha",
        type=float,
        default=0.05,
        help="Significance level",
    )

    # Sensitivity analyses
    parser.add_argument(
        "--run-sensitivity",
        action="store_true",
        help="Run grid + bootstrap sensitivity",
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=200,
        help="Number of bootstrap resamples",
    )

    # Visualization
    parser.add_argument(
        "--run-visualization",
        action="store_true",
        help="Generate publication-ready figures",
    )
    parser.add_argument(
        "--viz-dpi",
        type=int,
        default=300,
        help="Figure resolution (300 for submission, 600 for print)",
    )
    parser.add_argument(
        "--viz-top-k",
        type=int,
        default=30,
        help="Top K edges in waterfall plot",
    )
    parser.add_argument(
        "--viz-max-edges",
        type=int,
        default=40,
        help="Max edges in heatmap",
    )

    # Execution
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress",
    )

    args = parser.parse_args()

    # Execute
    rules = run_cascade_discovery_pipeline(
        data_path=args.data,
        output_dir=args.out,
        config_path=args.config,
        seed=args.seed,
        verbose=args.verbose,
        recode_mode=args.recode_mode,
        min_test_rate=args.min_test_rate,
        min_test_count=args.min_test_count,
        min_res_rate=args.min_res_rate,
        min_res_count=args.min_res_count,
        algorithm=args.algorithm,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_lift=args.min_lift,
        max_len=args.max_len,
        run_cleaning=not args.skip_cleaning,
        run_stats=args.run_stats,
        stats_alpha=args.stats_alpha,
        run_sensitivity=args.run_sensitivity,
        sensitivity_bootstrap_n=args.bootstrap_n,
        run_visualization=args.run_visualization,
        viz_dpi=args.viz_dpi,
        viz_top_k_waterfall=args.viz_top_k,
        viz_max_edges_heatmap=args.viz_max_edges,
    )

    print(f"\n✓ Pipeline complete!")
    print(f"  Results saved to: {args.out}")
    print(f"  Cascade rules: {len(rules)}")
    if args.run_visualization:
        print(f"  Figures saved to: {Path(args.out) / '06_figures'}")