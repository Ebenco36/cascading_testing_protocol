from src.controllers.DataExplorer import DataExplorer
from src.controllers.filters.FilteringStrategy import FilterConfig
import json

# Load JSON file
with open('./src/controllers/filters/config_all.json', 'r') as f:
    config = json.load(f)
config = FilterConfig.from_dict(config)
if __name__ == "__main__":
    # Initialize explorer
    explorer = DataExplorer(
        data_path="./datasets/structured/dataset_parquet",
        filter_config=config,
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
    # Print summary
    explorer.print_summary_report()
    # Generate plots
    explorer.generate_plots(output_dir="./publication_global", formats=["html", "png", "svg", "pdf"])