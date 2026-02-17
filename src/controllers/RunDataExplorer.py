#!/usr/bin/env python3
"""
Multi‑pathogen Data Exploration for Antibiotic Escalation Study.

For each pathogen genus (specified by a filter config JSON file):
  - Loads the full dataset, applies the genus‑specific filter.
  - Runs the DataExplorer to generate summary statistics and exploratory plots.
  - Saves all outputs (tables, figures) in a genus‑specific subdirectory.

Also handles a special "all isolates" config (config_all.json) that includes all pathogens.
"""

import sys
import logging
from pathlib import Path
import json

# Add the directory containing this script to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.controllers.DataExplorer import DataExplorer
from src.controllers.filters.FilteringStrategy import FilterConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_pathogen_name(config_path: Path) -> str:
    """
    Extract a human‑readable pathogen name from the config file name.
    Example: "config_klebsiella.json" -> "Klebsiella pneumoniae"
    Special case: "config_all.json" -> "All Isolates"
    Falls back to the stem if no mapping is defined.
    """
    stem = config_path.stem.replace("config_", "").lower()
    
    # Special case for all isolates
    if stem == "all":
        return "All Isolates"
    
    # Simple mapping for common pathogens
    mapping = {
        "klebsiella": "Klebsiella pneumoniae",
        "ecoli": "Escherichia coli",
        "pseudomonas": "Pseudomonas aeruginosa",
        "staphylococcus": "Staphylococcus aureus",
        "proteus": "Proteus mirabilis",
        "streptococcus": "Streptococcus pneumoniae",
    }
    return mapping.get(stem, stem.replace("_", " ").title())


def explore_one_pathogen(
    config_path: Path,
    data_path: str,
    base_output_dir: Path,
):
    """Run DataExplorer for a single pathogen filter config."""
    pathogen_name = extract_pathogen_name(config_path)
    # Create a safe directory name (lowercase, underscores)
    dir_name = pathogen_name.replace(" ", "_").lower()
    output_dir = base_output_dir / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}\nProcessing: {pathogen_name} (config: {config_path.name})\n{'='*60}")

    # Load filter config
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    filter_config = FilterConfig.from_dict(config_dict)

    # Initialise explorer with the pathogen name
    explorer = DataExplorer(
        data_path=data_path,
        filter_config=filter_config,
        pathogen_name=pathogen_name,   # pass the dynamic name
        required_covariates=[
            "CareType",
            "ARS_WardType",
            "Sex",
            "AgeRange",
            "AgeGroup"
        ],
        drop_missing_covariates=True,
    )

    # Run analysis
    explorer.run_analysis()

    # Print summary to console
    explorer.print_summary_report()

    # Generate plots and tables
    explorer.generate_plots(
        output_dir=str(output_dir),
        formats=["html", "png", "svg", "pdf"],
        save_tables=True,
    )

    logger.info(f"Exploration for {pathogen_name} complete. Outputs saved to {output_dir}\n")


def main():
    # Path to directory containing all filter config JSON files
    config_dir = Path("./src/controllers/filters/")
    config_paths = list(config_dir.glob("config_*.json"))
    if not config_paths:
        logger.error("No filter config files found.")
        return

    # Base output directory for all explorer results
    base_output_dir = Path("./explorer_output_multipathogen")
    base_output_dir.mkdir(exist_ok=True)

    # Common data path
    data_path = "./datasets/structured/dataset_parquet"

    # Loop over each pathogen config
    for cfg_path in config_paths:
        explore_one_pathogen(cfg_path, data_path, base_output_dir)

    logger.info("All pathogen explorations completed.")


if __name__ == "__main__":
    main()