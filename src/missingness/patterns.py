"""Missingness pattern helpers: univariate vs general column selection."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def select_target_columns(
    X: pd.DataFrame,
    metadata: dict[str, Any],
    feature_cols: list[str],
    pattern: str,
    rng: np.random.RandomState,
) -> list[str]:
    """Select which columns to inject missingness into based on pattern.

    Args:
        X: Complete data.
        metadata: Dataset metadata with feature type info.
        feature_cols: Eligible feature columns (target excluded).
        pattern: "univariate" or "general".
        rng: Random state for tie-breaking.

    Returns:
        List of column names to inject missingness into.
    """
    if pattern == "univariate":
        # Pick the numerical column with highest variance
        numerical_cols = metadata.get("numerical_features", [])
        num_candidates = [c for c in numerical_cols if c in feature_cols]
        if num_candidates:
            variances = {c: X[c].astype(float).var() for c in num_candidates}
            best_col = max(variances, key=variances.get)
            return [best_col]
        # Fallback: pick first feature column
        return [feature_cols[0]] if feature_cols else []
    else:
        # General: all feature columns
        return list(feature_cols)


def ensure_no_empty_rows(
    missing_mask: np.ndarray,
    col_indices: Sequence[int],
    rng: np.random.RandomState,
) -> None:
    """Ensure every row has at least one observed value among the given columns.

    Modifies missing_mask in-place.
    """
    if not col_indices:
        return
    sub_mask = missing_mask[:, col_indices]
    all_missing_rows = np.where(sub_mask.all(axis=1))[0]
    for row_idx in all_missing_rows:
        # Randomly restore one value in that row
        col_to_restore = rng.choice(col_indices)
        missing_mask[row_idx, col_to_restore] = False


def logistic_mask_with_calibration(
    driver_values: np.ndarray,
    n_rows: int,
    target_rate: float,
    rng: np.random.RandomState,
    slope: float = 2.0,
) -> np.ndarray:
    """Generate a missingness mask using logistic function with binary-search calibrated intercept.

    Args:
        driver_values: 1-D array of values that drive missingness probability.
        n_rows: Number of rows.
        target_rate: Desired overall missing rate.
        rng: Random state.
        slope: Steepness of the logistic curve (default 2.0 per spec).

    Returns:
        Boolean mask of shape (n_rows,).
    """
    # Standardize driver
    mean_val = np.nanmean(driver_values)
    std_val = np.nanstd(driver_values) + 1e-8
    x_std = (driver_values - mean_val) / std_val

    # Binary search for intercept so that mean(sigmoid(slope * x_std + intercept)) ≈ target_rate
    lo, hi = -10.0, 10.0
    for _ in range(64):
        mid = (lo + hi) / 2.0
        probs = _sigmoid(slope * x_std + mid)
        if np.nanmean(probs) < target_rate:
            lo = mid
        else:
            hi = mid

    intercept = (lo + hi) / 2.0
    probs = _sigmoid(slope * x_std + intercept)
    probs = np.clip(probs, 0.0, 0.99)

    return rng.rand(n_rows) < probs


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )
