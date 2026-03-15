"""MCAR (Missing Completely At Random) generator."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.missingness.base import BaseMissingnessGenerator, MissingnessResult
from src.missingness.patterns import select_target_columns, ensure_no_empty_rows
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
        pattern = kwargs.get("pattern", "general")

        # Determine which columns to inject missingness into
        feature_cols = [c for c in X.columns if not (exclude_target and c == target_col)]
        target_columns = select_target_columns(
            X, metadata, feature_cols, pattern, rng,
        )
        col_indices = np.array([X.columns.get_loc(c) for c in target_columns])

        n_rows, n_cols = X.shape
        missing_mask = np.zeros((n_rows, n_cols), dtype=bool)

        # Vectorized comparison, column-first draw order to preserve RNG sequence
        if len(col_indices) > 0:
            rand_block = np.column_stack([rng.rand(n_rows) for _ in col_indices])
            missing_mask[:, col_indices] = rand_block < rate

        # Ensure no row is entirely missing (general pattern constraint)
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
                "mechanism": "MCAR",
                "pattern": pattern,
                "rate": rate,
                "seed": seed,
                "exclude_target": exclude_target,
                "actual_missing_rate": float(missing_mask.sum()) / missing_mask.size,
            },
            scenario_tag=kwargs.get("scenario_tag", "benchmark_mcar"),
        )
