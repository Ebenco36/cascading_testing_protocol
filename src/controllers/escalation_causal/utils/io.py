"""
Input/Output utilities for saving results with metadata.
"""

import json
import time
import platform
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd


def save_results(
    out_dir: str,
    results: pd.DataFrame,
    config_dict: Dict[str, Any],
    cohort_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Save results DataFrame and a manifest with metadata.

    Args:
        out_dir: Output directory (will be created if it doesn't exist).
        results: DataFrame with pair‑level results.
        config_dict: Configuration dictionary (e.g., from RunConfig.to_dict()).
        cohort_meta: Optional dictionary with cohort metadata.

    Returns:
        Dictionary with paths to saved files.
    """
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)

    # Save results as CSV
    results_path = p / "results.csv"
    results.to_csv(results_path, index=False)

    # Create manifest
    manifest = {
        "created_at_unix": time.time(),
        "created_at_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "config": config_dict,
        "cohort_meta": cohort_meta,
        "n_results": len(results),
        "columns": results.columns.tolist(),
    }

    manifest_path = p / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    return {
        "results_csv": str(results_path),
        "manifest_json": str(manifest_path),
    }


def load_results(results_dir: str) -> Dict[str, Any]:
    """
    Load previously saved results and manifest.

    Args:
        results_dir: Directory containing results.csv and manifest.json.

    Returns:
        Dictionary with keys 'results' (DataFrame) and 'manifest' (dict).
    """
    p = Path(results_dir)
    results_path = p / "results.csv"
    manifest_path = p / "manifest.json"

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    results = pd.read_csv(results_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    return {"results": results, "manifest": manifest}