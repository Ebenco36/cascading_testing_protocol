"""
PipelineOrchestrator.py

Unified, class-based orchestration for the complete cascade discovery pipeline.

Integrates:
  1. Data filtering (FilteringStrategy)
  2. Transaction matrix construction (DataLoader)
  3. Cascade discovery (ARMEngine)
  4. Rule cleaning & collapsing (ARMRuleCleaner)
  5. Statistical testing (StatisticalTesting)
  6. Reproducibility (config files, audit trails, seed management)

Usage:
------

    from PipelineOrchestrator import CascadeDiscoveryPipeline

    pipeline = CascadeDiscoveryPipeline(
        config_path="./src/controllers/filters/config.json",
        output_dir="./output/cascade_discovery",
        seed=42
    )

    # Execute full pipeline
    pipeline.run(
        raw_data_path="./datasets/structured/dataset_parquet",
        arm_algorithm="fpgrowth",
        arm_params={"min_support": 0.01, "min_confidence": 0.30, "min_lift": 1.1},
        statistical_testing_contexts=["ARS_WardType", "CareType", "Year"],
    )

    # Access results
    print(pipeline.cascaderules)           # Raw discovered rules
    print(pipeline.cascaderules_cleaned)   # Cleaned rules (for analysis)
    print(pipeline.cascaderules_collapsed) # Collapsed/deduplicated (for visualization)
    print(pipeline.chi_square_results)     # Statistical test results

"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np

try:
    from src.controllers.filters.FilteringStrategy import FilterConfig
    from src.controllers.DataLoader import DataLoader
    from src.controllers.ARMEngine import ARMEngine
    from src.controllers.ARMRuleCleaner import ARMRuleCleaner, ARMRuleCleanerConfig
    from src.controllers.StatisticalTesting import CascadeStatisticalTesting
    from src.controllers.SensitivityAnalysis import (
        sensitivity_analysis_grid,
        sensitivity_analysis_bootstrap,
    )
    from src.runners.DataProcessing import save_parquet_flat
except ImportError:
    from controllers.filters.FilteringStrategy import FilterConfig
    from controllers.DataLoader import DataLoader
    from controllers.ARMEngine import ARMEngine
    from controllers.ARMRuleCleaner import ARMRuleCleaner, ARMRuleCleanerConfig
    from controllers.StatisticalTesting import CascadeStatisticalTesting
    from controllers.SensitivityAnalysis import (
        sensitivity_analysis_grid,
        sensitivity_analysis_bootstrap,
    )
    from runners.DataProcessing import save_parquet_flat


# ============================================================================
# Configuration Data Classes
# ============================================================================

@dataclass
class FilteringConfig:
    """Configuration for filtering stage."""
    config_path: str | Path
    save_filtered: bool = True
    filtered_output_dir: str | Path = "./datasets/dataset_parquet_filtered"
    rows_per_file: int = 250_000
    compression: str = "zstd"


@dataclass
class MatrixConfig:
    """Configuration for transaction matrix construction."""
    recode_mode: str = "R_vs_nonR"
    include_covariates: bool = True
    covariate_cols: List[str] | None = None
    min_test_rate: float = 0.05
    min_test_count: int = 2_000
    min_res_rate: float = 0.05
    min_res_count: int = 2_000
    always_keep_antibiotics: List[str] | None = None
    drop_all_susceptible_rows: bool = True

    def __post_init__(self):
        """Set defaults if not provided."""
        if self.covariate_cols is None:
            self.covariate_cols = ["ARS_WardType", "CareType", "AgeGroup", "Year"]
        if self.always_keep_antibiotics is None:
            self.always_keep_antibiotics = None


@dataclass
class ARMConfig:
    """Configuration for association rule mining."""
    algorithm: str = "fpgrowth"  # "apriori" or "fpgrowth"
    edge_mode: str = "pairwise"
    min_support: float = 0.01
    min_confidence: float = 0.30
    min_lift: float = 1.10
    max_len: Optional[int] = None


@dataclass
class RuleCleanerConfig:
    """Configuration for rule cleaning and collapsing."""
    drop_implied_tests: bool = True
    antecedent_keep_only_R: bool = True
    consequent_keep_only_T: bool = True
    require_both_sides_nonempty: bool = True
    drop_non_antibiotic_context: bool = True
    keep_only_informative: bool = True


@dataclass
class StatisticalTestingConfig:
    """Configuration for heterogeneity testing."""
    context_vars: List[str] | None = None
    chi_min_stratum_size: int = 10
    alpha: float = 0.05

    def __post_init__(self):
        if self.context_vars is None:
            self.context_vars = ["ARS_WardType", "CareType", "Year"]


@dataclass
class SensitivityConfig:
    """Configuration for sensitivity analyses."""
    run_grid: bool = True
    run_bootstrap: bool = True
    grid_support: List[float] | None = None
    grid_confidence: List[float] | None = None
    grid_lift: List[float] | None = None
    bootstrap_n: int = 200
    bootstrap_top_k: int = 30

    def __post_init__(self):
        if self.grid_support is None:
            self.grid_support = [0.005, 0.01, 0.02]
        if self.grid_confidence is None:
            self.grid_confidence = [0.20, 0.30, 0.40]
        if self.grid_lift is None:
            self.grid_lift = [1.0, 1.1, 1.2]


# ============================================================================
# Main Orchestrator Class
# ============================================================================

class CascadeDiscoveryPipeline:
    """
    Unified orchestration for cascade discovery pipeline.

    Stages:
      1. Filtering (6-stage, JSON-driven)
      2. Matrix construction (binary encoding, antibiotic selection)
      3. Cascade discovery (ARM with 2 backends: apriori, fpgrowth)
      4. Rule cleaning & collapsing (ARMRuleCleaner)
         ├─ cascaderules_cleaned: Full cleaned rules with intermediate columns
         └─ cascaderules_collapsed: Deduplicated rules (for visualization)
      5. Statistical testing (chi-square/Fisher on CLEANED rules) ← KEY FIX
      6. Sensitivity analyses (grid + bootstrap)
    """

    def __init__(
        self,
        config_path: str | Path,
        output_dir: str | Path = "./output/cascade_discovery",
        seed: int = 42,
        verbose: bool = True,
    ):
        """
        Initialize pipeline.

        Parameters
        ----------
        config_path : str | Path
            Path to filtering config JSON (defines 6-stage filtering)
        output_dir : str | Path
            Root output directory for all results
        seed : int
            Random seed for reproducibility
        verbose : bool
            Print progress messages
        """
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.verbose = verbose

        # Create output structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filtering_dir = self.output_dir / "01_filtering"
        self.matrix_dir = self.output_dir / "02_matrix"
        self.arm_dir = self.output_dir / "03_arm"
        self.cleaner_dir = self.output_dir / "04_cleaner"
        self.statistical_dir = self.output_dir / "05_statistical"
        self.sensitivity_dir = self.output_dir / "06_sensitivity"

        for d in [self.filtering_dir, self.matrix_dir, self.arm_dir, self.cleaner_dir,
                  self.statistical_dir, self.sensitivity_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = self._setup_logger()

        # State variables
        self.raw_df: pd.DataFrame | None = None
        self.filtered_df: pd.DataFrame | None = None
        self.matrix: pd.DataFrame | None = None
        self.cascaderules: pd.DataFrame | None = None
        self.cascaderules_cleaned: pd.DataFrame | None = None
        self.cascaderules_collapsed: pd.DataFrame | None = None
        self.chi_square_results: Dict[str, pd.DataFrame] = {}
        self.sensitivity_summary: pd.DataFrame | None = None
        self.edge_stability: pd.DataFrame | None = None

        self.logger.info(f"Pipeline initialized. Output dir: {self.output_dir}")

    def _setup_logger(self) -> logging.Logger:
        """Configure logging."""
        logger = logging.getLogger("CascadeDiscoveryPipeline")
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.output_dir / "pipeline.log")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        if self.verbose:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        return logger

    def _save_config(self, config_dict: Dict[str, Any], name: str) -> None:
        """Save configuration as JSON."""
        config_path = self.output_dir / f"{name}.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        self.logger.info(f"Saved config: {config_path}")

    # ========================================================================
    # Stage 1: Filtering
    # ========================================================================

    def filter_data(
        self,
        raw_data_path: str | Path,
        filtering_config: FilteringConfig | None = None,
    ) -> pd.DataFrame:
        """Apply 6-stage filtering using FilteringStrategy."""
        if filtering_config is None:
            filtering_config = FilteringConfig(config_path=self.config_path)

        self.logger.info("=" * 70)
        self.logger.info("STAGE 1: FILTERING")
        self.logger.info("=" * 70)

        self.logger.info(f"Loading raw data from {raw_data_path}...")
        self.raw_df = pd.read_parquet(raw_data_path)
        self.logger.info(f"Raw data shape: {self.raw_df.shape}")

        config = FilterConfig.from_json(filtering_config.config_path)
        self.logger.info(f"Loaded filtering config from {filtering_config.config_path}")

        self.filtered_df, reports = config.apply(self.raw_df)
        self.logger.info(f"Filtered data shape: {self.filtered_df.shape}")

        audit_df = pd.DataFrame(reports)
        audit_path = self.filtering_dir / "filter_audit.csv"
        audit_df.to_csv(audit_path, index=False)
        self.logger.info(f"Filter audit trail saved: {audit_path}")

        if filtering_config.save_filtered:
            self.logger.info(f"Saving filtered data to {filtering_config.filtered_output_dir}...")
            save_parquet_flat(
                self.filtered_df,
                filtering_config.filtered_output_dir,
                rows_per_file=filtering_config.rows_per_file,
                compression=filtering_config.compression,
            )
            self.logger.info("Filtered data saved")

        try:
            config_dict = config.to_dict() if hasattr(config, 'to_dict') else json.loads(config.to_json())
            self._save_config(config_dict, "filtering_config_used")
        except Exception as e:
            self.logger.warning(f"Could not save filtering config: {e}")

        return self.filtered_df

    # ========================================================================
    # Stage 2: Transaction Matrix Construction
    # ========================================================================

    def build_transaction_matrix(
        self,
        matrix_config: MatrixConfig | None = None,
    ) -> pd.DataFrame:
        """Construct binary transaction matrix from filtered data."""
        if self.filtered_df is None:
            raise ValueError("No filtered data. Run filter_data() first.")

        if matrix_config is None:
            matrix_config = MatrixConfig()

        self.logger.info("=" * 70)
        self.logger.info("STAGE 2: TRANSACTION MATRIX CONSTRUCTION")
        self.logger.info("=" * 70)

        filtered_path = Path("./datasets/dataset_parquet_filtered")
        print(matrix_config)
        loader = DataLoader(filtered_path)
        self.matrix = loader.get_transaction_matrix(
            filters=None,
            recode_mode=matrix_config.recode_mode,
            include_covariates=matrix_config.include_covariates,
            covariate_cols=matrix_config.covariate_cols,
            min_test_rate=matrix_config.min_test_rate,
            min_test_count=matrix_config.min_test_count,
            min_res_rate=matrix_config.min_res_rate,
            min_res_count=matrix_config.min_res_count,
            always_keep=matrix_config.always_keep_antibiotics,
            drop_all_susceptible_rows=matrix_config.drop_all_susceptible_rows,
        )

        self.logger.info(f"Transaction matrix shape: {self.matrix.shape}")
        self.logger.info(f"Sparsity: {100 * (1 - self.matrix.sum().sum() / self.matrix.size):.1f}%")

        matrix_metadata = {
            "shape": self.matrix.shape,
            "sparsity_pct": float(100 * (1 - self.matrix.sum().sum() / self.matrix.size)),
            "columns": list(self.matrix.columns),
        }
        self._save_config(matrix_metadata, "matrix_metadata")

        matrix_path = self.matrix_dir / "transaction_matrix.parquet"
        self.matrix.to_parquet(matrix_path)
        self.logger.info(f"Transaction matrix saved: {matrix_path}")

        config_dict = {
            "recode_mode": matrix_config.recode_mode,
            "covariate_cols": matrix_config.covariate_cols,
            "min_test_rate": matrix_config.min_test_rate,
            "min_test_count": matrix_config.min_test_count,
            "min_res_rate": matrix_config.min_res_rate,
            "min_res_count": matrix_config.min_res_count,
            "always_keep_antibiotics": matrix_config.always_keep_antibiotics,
        }
        self._save_config(config_dict, "matrix_config_used")

        return self.matrix

    # ========================================================================
    # Stage 3: Association Rule Mining
    # ========================================================================

    def discover_cascades(
        self,
        arm_config: ARMConfig | None = None,
    ) -> pd.DataFrame:
        """Discover cascade rules using association rule mining."""
        if self.matrix is None:
            raise ValueError("No transaction matrix. Run build_transaction_matrix() first.")

        if arm_config is None:
            arm_config = ARMConfig()

        self.logger.info("=" * 70)
        self.logger.info("STAGE 3: CASCADE DISCOVERY (ARM)")
        self.logger.info("=" * 70)

        engine = ARMEngine(self.matrix)
        self.cascaderules = engine.discover_rules(
            algorithm=arm_config.algorithm,
            edge_mode=arm_config.edge_mode,
            min_support=arm_config.min_support,
            min_confidence=arm_config.min_confidence,
            min_lift=arm_config.min_lift,
            max_len=arm_config.max_len,
            verbose=self.verbose,
        )

        self.cascaderules = self.cascaderules.sort_values(
            by=["Lift", "Confidence", "Support"], ascending=False
        ).reset_index(drop=True)

        self.logger.info(f"Discovered {len(self.cascaderules)} cascade rules")

        rules_path = self.arm_dir / "cascaderules.csv"
        self.cascaderules.to_csv(rules_path, index=False)
        self.logger.info(f"Cascade rules saved: {rules_path}")

        config_dict = {
            "algorithm": arm_config.algorithm,
            "edge_mode": arm_config.edge_mode,
            "min_support": arm_config.min_support,
            "min_confidence": arm_config.min_confidence,
            "min_lift": arm_config.min_lift,
            "max_len": arm_config.max_len,
        }
        self._save_config(config_dict, "arm_config_used")

        return self.cascaderules

    # ========================================================================
    # Stage 4: Rule Cleaning & Collapsing
    # ========================================================================

    def clean_and_collapse_rules(
        self,
        cleaner_config: RuleCleanerConfig | None = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Clean cascade rules and collapse duplicates.

        Returns:
            cascaderules_cleaned: Full cleaned rules with all intermediate columns
                                 (for statistical testing + analysis)
            cascaderules_collapsed: Deduplicated rules (for visualization + reporting)
        """
        if self.cascaderules is None:
            raise ValueError("No cascade rules. Run discover_cascades() first.")

        if cleaner_config is None:
            cleaner_config = RuleCleanerConfig()

        self.logger.info("=" * 70)
        self.logger.info("STAGE 4: RULE CLEANING & COLLAPSING")
        self.logger.info("=" * 70)

        # ====================================================================
        # VALIDATION: Check required columns exist
        # ====================================================================
        required_cols = ["Antecedents", "Consequents"]
        missing_cols = [c for c in required_cols if c not in self.cascaderules.columns]
        if missing_cols:
            raise ValueError(
                f"cascaderules missing required columns: {missing_cols}. "
                f"Expected: {required_cols}. Got: {list(self.cascaderules.columns)}"
            )
        self.logger.info(f"✓ Validated cascaderules structure: {required_cols}")

        # ====================================================================
        # STEP 1: CLEAN
        # ====================================================================
        self.logger.info("Step 1: Cleaning rules...")

        arm_cleaner_cfg = ARMRuleCleanerConfig(
            drop_implied_tests=cleaner_config.drop_implied_tests,
            antecedent_keep_only_R=cleaner_config.antecedent_keep_only_R,
            consequent_keep_only_T=cleaner_config.consequent_keep_only_T,
            require_both_sides_nonempty=cleaner_config.require_both_sides_nonempty,
            drop_non_antibiotic_context=cleaner_config.drop_non_antibiotic_context,
        )

        cleaner = ARMRuleCleaner(arm_cleaner_cfg)
        self.cascaderules_cleaned = cleaner.clean_dataframe(
            self.cascaderules,
            antecedent_col="Antecedents",
            consequent_col="Consequents",
        )

        if self.cascaderules_cleaned is None or self.cascaderules_cleaned.empty:
            raise ValueError("Rule cleaning produced empty DataFrame")

        n_inform = (self.cascaderules_cleaned["Is_informative"] == True).sum()
        n_cross = (self.cascaderules_cleaned["Is_cross_informative"] == True).sum()
        self.logger.info(f"  ✓ Cleaned rules: {len(self.cascaderules_cleaned)} total")
        self.logger.info(f"    - Informative: {n_inform}")
        self.logger.info(f"    - Cross-informative: {n_cross}")

        cleaned_path = self.cleaner_dir / "cascaderules_cleaned.csv"
        self.cascaderules_cleaned.to_csv(cleaned_path, index=False)
        self.logger.info(f"  ✓ Saved cleaned rules to: {cleaned_path}")

        # ====================================================================
        # STEP 2: COLLAPSE (for visualization/reporting ONLY)
        # ====================================================================
        self.logger.info("Step 2: Collapsing duplicate rules...")

        self.cascaderules_collapsed = cleaner.collapse_rules(
            self.cascaderules_cleaned,
            key_col="Rule_cross_normalized",  # ← EXPLICIT: use cross-normalized rules
            keep_only_informative=cleaner_config.keep_only_informative,
        )

        if self.cascaderules_collapsed is None or self.cascaderules_collapsed.empty:
            self.logger.warning("Rule collapsing produced empty DataFrame")
            self.cascaderules_collapsed = pd.DataFrame()
        else:
            avg_collapse = self.cascaderules_collapsed["n_rules"].mean()
            self.logger.info(f"  ✓ Collapsed to {len(self.cascaderules_collapsed)} unique rules")
            self.logger.info(f"  ✓ Average rules per group: {avg_collapse:.1f}")

        collapsed_path = self.cleaner_dir / "cascaderules_collapsed.csv"
        self.cascaderules_collapsed.to_csv(collapsed_path, index=False)
        self.logger.info(f"  ✓ Saved collapsed rules to: {collapsed_path}")

        # ====================================================================
        # SAVE CONFIG
        # ====================================================================
        config_dict = {
            "drop_implied_tests": cleaner_config.drop_implied_tests,
            "antecedent_keep_only_R": cleaner_config.antecedent_keep_only_R,
            "consequent_keep_only_T": cleaner_config.consequent_keep_only_T,
            "require_both_sides_nonempty": cleaner_config.require_both_sides_nonempty,
            "drop_non_antibiotic_context": cleaner_config.drop_non_antibiotic_context,
            "keep_only_informative": cleaner_config.keep_only_informative,
        }
        self._save_config(config_dict, "cleaner_config_used")

        return self.cascaderules_cleaned, self.cascaderules_collapsed

    # ========================================================================
    # Stage 5: Statistical Testing
    # ========================================================================

    def _expand_rules_to_edges(self, rules: pd.DataFrame) -> pd.DataFrame:
        """
        Expand (possibly multi-item) rules into edge-level rules with Source/Target.

        For each rule, generate all R×T edges:
          - Source: antibiotic code from Antecedents items ending with "_R"
          - Target: antibiotic code from Consequents items ending with "_T"

        Returns a DataFrame with at least:
            Source, Target, Antecedents, Consequents, Support, Confidence, Lift
        """
        rows: List[Dict[str, Any]] = []

        for _, row in rules.iterrows():
            ant = row["Antecedents"]
            cons = row["Consequents"]

            # Normalize Antecedents → list of strings
            if isinstance(ant, (set, frozenset)):
                ant_list = list(ant)
            elif isinstance(ant, str):
                ant_list = [
                    x.strip().strip("{}")
                    for x in ant.replace("{", "").replace("}", "").split(",")
                    if x.strip()
                ]
            else:
                ant_list = []

            # Normalize Consequents → list of strings
            if isinstance(cons, (set, frozenset)):
                cons_list = list(cons)
            elif isinstance(cons, str):
                cons_list = [
                    x.strip().strip("{}")
                    for x in cons.replace("{", "").replace("}", "").split(",")
                    if x.strip()
                ]
            else:
                cons_list = []

            # Extract all R and T antibiotics
            sources = [x.replace("_R", "") for x in ant_list if isinstance(x, str) and x.endswith("_R")]
            targets = [x.replace("_T", "") for x in cons_list if isinstance(x, str) and x.endswith("_T")]

            if not sources or not targets:
                # No valid edge for this rule
                continue

            # Generate all pairwise edges
            for s in sources:
                for t in targets:
                    rows.append(
                        {
                            "Source": s,
                            "Target": t,
                            "Antecedents": row["Antecedents"],
                            "Consequents": row["Consequents"],
                            "Support": row.get("Support", np.nan),
                            "Confidence": row.get("Confidence", np.nan),
                            "Lift": row.get("Lift", np.nan),
                            # "Is_informative": row.get("Is_informative", np.nan),
                            # "Is_cross_informative": row.get("Is_cross_informative", np.nan),
                        }
                    )

        edge_df = pd.DataFrame(rows)
        if edge_df.empty:
            self.logger.warning("Edge expansion produced an empty DataFrame.")
        else:
            self.logger.info(
                f"✓ Edge expansion: {len(rules)} rules → {len(edge_df)} edges "
                f"({edge_df[['Source','Target']].drop_duplicates().shape[0]} unique edges)"
            )
        return edge_df


    def test_heterogeneity(
        self,
        statistical_config: StatisticalTestingConfig | None = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Test cascade heterogeneity across clinical contexts.

        CRITICAL FIX: Uses cascaderules_CLEANED (not collapsed).
        Collapsed rules lose granularity needed for stratified testing.
        """
        if self.cascaderules_cleaned is None:
            self.logger.warning("cascaderules_cleaned not available; using raw cascaderules")
            rules_to_test = self.cascaderules
        else:
            rules_to_test = self.cascaderules_cleaned


        if rules_to_test is None:
            raise ValueError(
                "No cascade rules for testing. "
                "Run discover_cascades() and clean_and_collapse_rules() first."
            )

        if self.filtered_df is None:
            raise ValueError("No filtered data. Run filter_data() first.")

        if statistical_config is None:
            statistical_config = StatisticalTestingConfig()

        self.logger.info("=" * 70)
        self.logger.info("STAGE 5: STATISTICAL TESTING (Chi-Square / Fisher Heterogeneity)")
        self.logger.info("=" * 70)
        self.logger.info(f"Using {'CLEANED' if self.cascaderules_cleaned is not None else 'RAW'} rules for testing")

        # ====================================================================
        # VALIDATION: Check required columns for statistical testing
        # ====================================================================
        required_test_cols = ["Antecedents", "Consequents", "Support", "Confidence", "Lift"]
        missing_test_cols = [c for c in required_test_cols if c not in rules_to_test.columns]
        if missing_test_cols:
            raise ValueError(
                f"Rules for testing missing columns: {missing_test_cols}. "
                f"Got: {list(rules_to_test.columns)}"
            )
        self.logger.info(f"✓ Validated rules structure for testing: {required_test_cols}")

        # ====================================================================
        # RUN STATISTICAL TESTS
        # ====================================================================
        # ====================================================================
        # EXPAND CLEANED RULES TO EDGE-LEVEL RULES
        # ====================================================================
        # We test heterogeneity at the antibiotic-edge level (Source_R → Target_T),
        # not at the whole multi-antibiotic pattern level.
        edge_rules = self._expand_rules_to_edges(rules_to_test)

        if edge_rules is None or edge_rules.empty:
            raise ValueError(
                "Edge expansion produced no rules for testing. "
                "Check that Antecedents/Consequents contain *_R and *_T items."
            )

        # Validate again now that we have edge_rules
        required_test_cols = ["Antecedents", "Consequents", "Support", "Confidence", "Lift"]
        missing_test_cols = [c for c in required_test_cols if c not in edge_rules.columns]
        if missing_test_cols:
            raise ValueError(
                f"Edge-level rules for testing missing columns: {missing_test_cols}. "
                f"Got: {list(edge_rules.columns)}"
            )
        self.logger.info("✓ Using edge-level rules with Source/Target for heterogeneity testing")

        # ====================================================================
        # RUN STATISTICAL TESTS (EDGE-LEVEL)
        # ====================================================================
        tester = CascadeStatisticalTesting(
            raw_df=self.filtered_df,
            cascade_rules=edge_rules,  # ← now has Source & Target
            verbose=self.verbose,
        )

        self.chi_square_results = tester.test_all_contexts(
            context_vars=statistical_config.context_vars,
            min_stratum_size=statistical_config.chi_min_stratum_size,
            alpha=statistical_config.alpha,
        )


        # Save results
        n_results = 0
        for context_var, results_df in self.chi_square_results.items():
            if results_df is not None and not results_df.empty:
                n_results += len(results_df)
                output_path = self.statistical_dir / f"chi_square_{context_var.lower()}.csv"
                results_df.to_csv(output_path, index=False)
                self.logger.info(f"  ✓ Chi-square results for {context_var}: {len(results_df)} tests")

        self.logger.info(f"✓ Total statistical tests completed: {n_results}")

        # Save statistical config
        config_dict = {
            "context_vars": statistical_config.context_vars,
            "chi_min_stratum_size": statistical_config.chi_min_stratum_size,
            "alpha": statistical_config.alpha,
        }
        self._save_config(config_dict, "statistical_config_used")

        return self.chi_square_results

    # ========================================================================
    # Stage 6: Sensitivity Analyses
    # ========================================================================

    def run_sensitivity_analyses(
        self,
        sensitivity_config: SensitivityConfig | None = None,
    ) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """Run grid sensitivity and bootstrap stability analyses."""
        if self.matrix is None:
            raise ValueError("No transaction matrix. Run build_transaction_matrix() first.")

        if sensitivity_config is None:
            sensitivity_config = SensitivityConfig()

        self.logger.info("=" * 70)
        self.logger.info("STAGE 6: SENSITIVITY ANALYSES")
        self.logger.info("=" * 70)

        if sensitivity_config.run_grid:
            self.logger.info("Running grid sensitivity analysis...")
            self.sensitivity_summary = sensitivity_analysis_grid(
                matrix=self.matrix,
                output_dir=self.sensitivity_dir,
                support_grid=sensitivity_config.grid_support,
                confidence_grid=sensitivity_config.grid_confidence,
                lift_grid=sensitivity_config.grid_lift,
                top_k=sensitivity_config.bootstrap_top_k,
                seed=self.seed,
                verbose=self.verbose,
                algorithm="fpgrowth",
            )
            if self.sensitivity_summary is not None:
                self.logger.info(f"  ✓ Grid sensitivity: {len(self.sensitivity_summary)} settings tested")

        if sensitivity_config.run_bootstrap:
            self.logger.info("Running bootstrap stability analysis...")
            self.edge_stability = sensitivity_analysis_bootstrap(
                matrix=self.matrix,
                output_dir=self.sensitivity_dir,
                n_bootstrap=sensitivity_config.bootstrap_n,
                top_k=sensitivity_config.bootstrap_top_k,
                seed=self.seed,
                verbose=self.verbose,
                algorithm="fpgrowth",
                min_support=0.1,
                min_confidence=0.60,
                min_lift=1.10,
                max_len=4,
            )
            if self.edge_stability is not None:
                self.logger.info(f"  ✓ Bootstrap stability: {sensitivity_config.bootstrap_n} resamples")

        config_dict = {
            "run_grid": sensitivity_config.run_grid,
            "run_bootstrap": sensitivity_config.run_bootstrap,
            "grid_support": sensitivity_config.grid_support,
            "grid_confidence": sensitivity_config.grid_confidence,
            "grid_lift": sensitivity_config.grid_lift,
            "bootstrap_n": sensitivity_config.bootstrap_n,
            "bootstrap_top_k": sensitivity_config.bootstrap_top_k,
        }
        self._save_config(config_dict, "sensitivity_config_used")

        return self.sensitivity_summary, self.edge_stability

    # ========================================================================
    # Full Pipeline Orchestration
    # ========================================================================

    def run(
        self,
        raw_data_path: str | Path,
        filtering_config: FilteringConfig | None = None,
        matrix_config: MatrixConfig | None = None,
        arm_config: ARMConfig | None = None,
        cleaner_config: RuleCleanerConfig | None = None,
        statistical_config: StatisticalTestingConfig | None = None,
        sensitivity_config: SensitivityConfig | None = None,
    ) -> None:
        """
        Execute complete cascade discovery pipeline end-to-end.

        Pipeline Flow:
            1. filter_data() → cascaderules = None
            2. build_transaction_matrix() → matrix
            3. discover_cascades() → cascaderules (raw)
            4. clean_and_collapse_rules() → cascaderules_cleaned, cascaderules_collapsed
            5. test_heterogeneity(cascaderules_CLEANED) → chi_square_results
            6. run_sensitivity_analyses() → sensitivity_summary, edge_stability
        """
        try:
            self.logger.info("\n" + "=" * 70)
            self.logger.info("STARTING CASCADE DISCOVERY PIPELINE")
            self.logger.info("=" * 70 + "\n")

            # Stage 1: Filtering
            self.filter_data(raw_data_path, filtering_config)

            # Stage 2: Matrix construction
            self.build_transaction_matrix(matrix_config)

            # Stage 3: Cascade discovery
            self.discover_cascades(arm_config)

            # Stage 4: Rule cleaning & collapsing
            self.clean_and_collapse_rules(cleaner_config)

            # Stage 5: Statistical testing (on CLEANED rules) ← KEY FIX
            self.test_heterogeneity(statistical_config)

            # Stage 6: Sensitivity analyses
            self.run_sensitivity_analyses(sensitivity_config)

            self.logger.info("\n" + "=" * 70)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 70 + "\n")

            self._print_summary()

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

    def _print_summary(self) -> None:
        """Print execution summary."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("FINAL SUMMARY")
        self.logger.info("=" * 70)
        self.logger.info(f"  Raw data:              {self.raw_df.shape if self.raw_df is not None else 'N/A'}")
        self.logger.info(f"  Filtered data:         {self.filtered_df.shape if self.filtered_df is not None else 'N/A'}")
        self.logger.info(f"  Transaction matrix:    {self.matrix.shape if self.matrix is not None else 'N/A'}")
        self.logger.info(f"  Raw cascaderules:      {len(self.cascaderules) if self.cascaderules is not None else 'N/A'}")
        self.logger.info(f"  Cleaned cascaderules:  {len(self.cascaderules_cleaned) if self.cascaderules_cleaned is not None else 'N/A'}")
        self.logger.info(f"  Collapsed cascaderules: {len(self.cascaderules_collapsed) if self.cascaderules_collapsed is not None else 'N/A'}")
        self.logger.info(f"  Chi-square tests:      {sum(len(df) if df is not None else 0 for df in self.chi_square_results.values())}")
        self.logger.info(f"  Output directory:      {self.output_dir}")
        self.logger.info("=" * 70 + "\n")

    # ========================================================================
    # Export & Reporting
    # ========================================================================

    def export_results(self, format: str = "csv") -> None:
        """
        Export all results to specified format.

        Parameters
        ----------
        format : str
            Export format: "csv" or "parquet"
        """
        ext = ".csv" if format == "csv" else ".parquet"
        to_file = lambda df, path: df.to_csv(path, index=False) if format == "csv" else df.to_parquet(path)

        self.logger.info(f"\nExporting results in {format.upper()} format...")

        if self.cascaderules is not None:
            to_file(self.cascaderules, self.arm_dir / f"cascaderules{ext}")
            self.logger.info(f"  ✓ Raw rules: {self.arm_dir / f'cascaderules{ext}'}")

        if self.cascaderules_cleaned is not None:
            to_file(self.cascaderules_cleaned, self.cleaner_dir / f"cascaderules_cleaned{ext}")
            self.logger.info(f"  ✓ Cleaned rules: {self.cleaner_dir / f'cascaderules_cleaned{ext}'}")

        if self.cascaderules_collapsed is not None and not self.cascaderules_collapsed.empty:
            to_file(self.cascaderules_collapsed, self.cleaner_dir / f"cascaderules_collapsed{ext}")
            self.logger.info(f"  ✓ Collapsed rules: {self.cleaner_dir / f'cascaderules_collapsed{ext}'}")

        for ctx, df in self.chi_square_results.items():
            if df is not None and not df.empty:
                to_file(df, self.statistical_dir / f"chi_square_{ctx.lower()}{ext}")
                self.logger.info(f"  ✓ Chi-square ({ctx}): {self.statistical_dir / f'chi_square_{ctx.lower()}{ext}'}")

        self.logger.info(f"✓ Export complete\n")

    def print_cascade_summary(self, top_n: int = 20) -> None:
        """
        Print top cascades from collapsed rules (or raw if collapsed unavailable).
        """
        rules_to_print = self.cascaderules_collapsed if (
            self.cascaderules_collapsed is not None and not self.cascaderules_collapsed.empty
        ) else self.cascaderules

        if rules_to_print is None:
            self.logger.warning("No cascade rules to display")
            return

        self.logger.info(f"\nTop {top_n} Cascades (by max_lift):\n")
        print(rules_to_print.head(top_n).to_string())

    def print_heterogeneity_summary(self, context_var: str = "ARS_WardType", top_n: int = 20) -> None:
        """Print heterogeneity test results for a specific context."""
        if context_var not in self.chi_square_results:
            self.logger.warning(f"No results for context variable: {context_var}")
            return

        results = self.chi_square_results[context_var]
        if results is None or results.empty:
            self.logger.warning(f"Empty results for context variable: {context_var}")
            return

        self.logger.info(f"\nTop {top_n} Heterogeneous Cascades (by p-value, context={context_var}):\n")
        print(results.head(top_n).to_string())


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    pipeline = CascadeDiscoveryPipeline(
        config_path="./src/controllers/filters/config.json",
        output_dir="./output/cascade_discovery_full",
        seed=42,
        verbose=True,
    )

    filtering_cfg = FilteringConfig(
        config_path="./src/controllers/filters/config.json",
        save_filtered=True,
    )

    matrix_cfg = MatrixConfig(
        recode_mode="R_vs_nonR",
        min_test_rate=0.05,
        min_test_count=2_000,
        min_res_rate=0.05,
        min_res_count=2_000,
        drop_all_susceptible_rows=True,
    )

    arm_cfg = ARMConfig(
        algorithm="fpgrowth",
        min_support=0.01,
        min_confidence=0.30,
        min_lift=1.1,
        max_len=4,
    )

    cleaner_cfg = RuleCleanerConfig(
        drop_implied_tests=True,
        antecedent_keep_only_R=True,
        consequent_keep_only_T=True,
        require_both_sides_nonempty=True,
        drop_non_antibiotic_context=True,
        keep_only_informative=True,
    )

    statistical_cfg = StatisticalTestingConfig(
        context_vars=["ARS_WardType", "CareType", "Year"],
        chi_min_stratum_size=10,
        alpha=0.05,
    )

    sensitivity_cfg = SensitivityConfig(
        run_grid=True,
        run_bootstrap=True,
        bootstrap_n=200,
    )

    # Run full pipeline
    pipeline.run(
        raw_data_path="./datasets/structured/dataset_parquet",
        filtering_config=filtering_cfg,
        matrix_config=matrix_cfg,
        arm_config=arm_cfg,
        cleaner_config=cleaner_cfg,
        statistical_config=statistical_cfg,
        sensitivity_config=sensitivity_cfg,
    )

    # Print summaries
    pipeline.print_cascade_summary(top_n=30)
    pipeline.print_heterogeneity_summary(context_var="ARS_WardType", top_n=20)

    # Export results
    pipeline.export_results(format="csv")