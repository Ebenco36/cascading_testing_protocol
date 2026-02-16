"""
Utility functions for weight handling and diagnostics.
"""

import numpy as np
from typing import Dict, Tuple, Optional

import pandas as pd


def clip_prob(p: np.ndarray, lo: float = 0.01, hi: float = 0.99) -> np.ndarray:
    """Clip probabilities to avoid extremes."""
    return np.clip(p, lo, hi)


def effective_sample_size(weights: np.ndarray) -> float:
    """
    Compute Kish's effective sample size: (sum w)^2 / sum(w^2).
    Returns NaN if weights are empty or all zero.
    """
    w = np.asarray(weights, dtype=float)
    if w.size == 0 or np.sum(w) == 0:
        return float("nan")
    sum_w = np.sum(w)
    sum_w2 = np.sum(w**2)
    return float((sum_w * sum_w) / sum_w2) if sum_w2 > 0 else float("nan")


def stable_quantiles(
    x: np.ndarray,
    qs: Tuple[float, ...] = (0, 0.01, 0.5, 0.99, 1.0)
) -> Dict[str, float]:
    """
    Compute quantiles of array x, returning a dictionary with keys like 'q00', 'q01', etc.
    Returns NaN for each quantile if x is empty.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {f"q{int(q*100):02d}": float("nan") for q in qs}
    out = {}
    for q in qs:
        out[f"q{int(q*100):02d}"] = float(np.quantile(x, q))
    return out


def trim_weights(
    weights: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Trim extreme weights at specified percentiles.
    Returns trimmed weights and a dictionary with original min/max and trim bounds.
    """
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return w, {}
    lo = np.percentile(w, lower_percentile)
    hi = np.percentile(w, upper_percentile)
    trimmed = np.clip(w, lo, hi)
    diag = {
        "original_min": float(w.min()),
        "original_max": float(w.max()),
        "trim_lower": float(lo),
        "trim_upper": float(hi),
        "n_trimmed_lower": int((w < lo).sum()),
        "n_trimmed_upper": int((w > hi).sum()),
    }
    return trimmed, diag

def safe_series_to_numpy(s: pd.Series) -> np.ndarray:
    """
    Convert a pandas Series to a numpy array, coercing to numeric and filling NaN with 0.
    """
    return pd.to_numeric(s, errors="coerce").fillna(0).to_numpy()