"""MCAR (Missing Completely At Random) generator."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.missingness.base import BaseMissingnessGenerator, MissingnessResult
from src.registry import MISSINGNESS_REGISTRY


@MISSINGNESS_REGISTRY.register("MCAR")
class MCARGenerator(BaseMissingnessGenerator):
    """Generate MCAR missingness — each value has equal probability of being missing."""

    name = "MCAR"

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
        target_col = metadata.get("target_col")

        # Determine which columns to inject missingness into
        feature_cols = [c for c in X.columns if not (exclude_target and c == target_col)]
        col_indices = [X.columns.get_loc(c) for c in feature_cols]

        n_rows, n_cols = X.shape
        missing_mask = np.zeros((n_rows, n_cols), dtype=bool)

        # Generate random mask for feature columns only
        for idx in col_indices:
            mask_col = rng.rand(n_rows) < rate
            missing_mask[:, idx] = mask_col

        # Apply mask
        X_missing = X.copy()
        X_missing_values = X_missing.values.copy().astype(float)
        X_missing_values[missing_mask] = np.nan
        X_missing = pd.DataFrame(X_missing_values, columns=X.columns, index=X.index)

        # Restore original dtypes for non-missing columns where possible
        observed_mask = ~missing_mask

        return MissingnessResult(
            X_missing=X_missing,
            missing_mask=missing_mask,
            observed_mask=observed_mask,
            missing_config={
                "mechanism": "MCAR",
                "rate": rate,
                "seed": seed,
                "exclude_target": exclude_target,
                "actual_missing_rate": float(missing_mask.sum()) / missing_mask.size,
            },
            scenario_tag=kwargs.get("scenario_tag", "benchmark_mcar"),
        )
