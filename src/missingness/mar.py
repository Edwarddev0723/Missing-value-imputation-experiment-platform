"""MAR (Missing At Random) generator — logistic-function version per spec."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.missingness.base import BaseMissingnessGenerator, MissingnessResult
from src.missingness.patterns import (
    ensure_no_empty_rows,
    logistic_mask_with_calibration,
    select_target_columns,
)
from src.registry import MISSINGNESS_REGISTRY


@MISSINGNESS_REGISTRY.register("MAR")
class MARGenerator(BaseMissingnessGenerator):
    """Generate MAR missingness — missingness depends on observed variables.

    For each target feature X_k, a different observed feature X_j drives the
    missingness probability through a logistic function with slope=2.0.
    The intercept is calibrated via binary search so overall rate ≈ target rate.
    """

    name = "MAR"

    def generate(
        self,
        X: pd.DataFrame,
        metadata: dict[str, Any],
        seed: int = 42,
        rate: float = 0.1,
        **kwargs: Any,
    ) -> MissingnessResult:
        rng = np.random.RandomState(seed)

        exclude_target = kwargs.get("exclude_target", True)
        target_col_name = metadata.get("target_col")
        pattern = kwargs.get("pattern", "general")

        numerical_features = metadata.get("numerical_features", [])
        feature_cols = [c for c in X.columns if not (exclude_target and c == target_col_name)]
        target_columns = select_target_columns(X, metadata, feature_cols, pattern, rng)

        n_rows, n_cols = X.shape
        missing_mask = np.zeros((n_rows, n_cols), dtype=bool)

        # Available numerical driver columns
        num_cols_available = [c for c in numerical_features if c in feature_cols]

        for feat_col in target_columns:
            col_idx = X.columns.get_loc(feat_col)

            # Pick a driver column (different from feat_col)
            drivers = [c for c in num_cols_available if c != feat_col]
            if not drivers:
                # Fallback: MCAR for this column
                missing_mask[:, col_idx] = rng.rand(n_rows) < rate
                continue

            driver_col = drivers[rng.randint(len(drivers))]
            driver_values = X[driver_col].values.astype(float)

            missing_mask[:, col_idx] = logistic_mask_with_calibration(
                driver_values, n_rows, rate, rng, slope=2.0,
            )

        # Ensure no row is entirely missing (general pattern constraint)
        col_indices = [X.columns.get_loc(c) for c in target_columns]
        if pattern == "general":
            ensure_no_empty_rows(missing_mask, col_indices, rng)

        # Apply mask — use DataFrame operations to preserve mixed dtypes
        X_missing = X.copy()
        X_missing[missing_mask] = np.nan

        observed_mask = ~missing_mask

        return MissingnessResult(
            X_missing=X_missing,
            missing_mask=missing_mask,
            observed_mask=observed_mask,
            missing_config={
                "mechanism": "MAR",
                "pattern": pattern,
                "rate": rate,
                "seed": seed,
                "exclude_target": exclude_target,
                "strategy": "logistic_calibrated",
                "slope": 2.0,
                "actual_missing_rate": float(missing_mask.sum()) / missing_mask.size,
            },
            scenario_tag=kwargs.get("scenario_tag", "benchmark_mar"),
        )
