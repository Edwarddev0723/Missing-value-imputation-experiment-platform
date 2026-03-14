"""MNAR (Missing Not At Random) generator — self-value logistic version per spec."""

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


@MISSINGNESS_REGISTRY.register("MNAR")
class MNARGenerator(BaseMissingnessGenerator):
    """Generate MNAR missingness — missingness depends on the variable's own value.

    For each target feature X_k, higher values of X_k itself lead to
    higher probability of being missing, using a logistic function
    with slope=2.0 and binary-search calibrated intercept.
    """

    name = "MNAR"

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

        feature_cols = [c for c in X.columns if not (exclude_target and c == target_col_name)]
        target_columns = select_target_columns(X, metadata, feature_cols, pattern, rng)

        n_rows, n_cols = X.shape
        missing_mask = np.zeros((n_rows, n_cols), dtype=bool)

        for feat_col in target_columns:
            col_idx = X.columns.get_loc(feat_col)
            col_values = X[feat_col].values.astype(float)

            # Check if column has numeric variance
            if np.nanstd(col_values) < 1e-10:
                # Constant column — fall back to MCAR
                missing_mask[:, col_idx] = rng.rand(n_rows) < rate
                continue

            # Use the column's own values as driver (self-masking)
            missing_mask[:, col_idx] = logistic_mask_with_calibration(
                col_values, n_rows, rate, rng, slope=2.0,
            )

        # Ensure no row is entirely missing (general pattern constraint)
        col_indices = [X.columns.get_loc(c) for c in target_columns]
        if pattern == "general":
            ensure_no_empty_rows(missing_mask, col_indices, rng)

        # Apply mask
        X_missing = X.copy()
        X_missing_values = X_missing.values.copy().astype(float)
        X_missing_values[missing_mask] = np.nan
        X_missing = pd.DataFrame(X_missing_values, columns=X.columns, index=X.index)

        observed_mask = ~missing_mask

        return MissingnessResult(
            X_missing=X_missing,
            missing_mask=missing_mask,
            observed_mask=observed_mask,
            missing_config={
                "mechanism": "MNAR",
                "pattern": pattern,
                "rate": rate,
                "seed": seed,
                "exclude_target": exclude_target,
                "strategy": "self_logistic_calibrated",
                "slope": 2.0,
                "actual_missing_rate": float(missing_mask.sum()) / missing_mask.size,
            },
            scenario_tag=kwargs.get("scenario_tag", "benchmark_mnar"),
        )
