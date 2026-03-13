"""MAR (Missing At Random) generator — simplified rule-based version."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.missingness.base import BaseMissingnessGenerator, MissingnessResult
from src.registry import MISSINGNESS_REGISTRY


@MISSINGNESS_REGISTRY.register("MAR")
class MARGenerator(BaseMissingnessGenerator):
    """Generate MAR missingness — missingness depends on observed variables.

    Strategy: For each target feature, the probability of being missing
    depends on the value of another (observed) feature. Specifically,
    we pick a "driver" feature and make high/low values of the driver
    increase the probability of the target feature being missing.
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

        numerical_features = metadata.get("numerical_features", [])
        feature_cols = [c for c in X.columns if not (exclude_target and c == target_col_name)]

        n_rows, n_cols = X.shape
        missing_mask = np.zeros((n_rows, n_cols), dtype=bool)

        # For each feature column, pick a different numerical column as "driver"
        # If no numerical driver is available, fall back to MCAR for that column
        num_cols_available = [c for c in numerical_features if c in feature_cols]

        for feat_col in feature_cols:
            col_idx = X.columns.get_loc(feat_col)

            # Pick a driver column (different from feat_col)
            drivers = [c for c in num_cols_available if c != feat_col]
            if not drivers:
                # Fallback: MCAR for this column
                missing_mask[:, col_idx] = rng.rand(n_rows) < rate
                continue

            driver_col = drivers[rng.randint(len(drivers))]
            driver_values = X[driver_col].values.astype(float)

            # Compute missingness probability based on driver quantile
            # Higher driver value → higher probability of missing
            finite_mask = np.isfinite(driver_values)
            if not finite_mask.any():
                missing_mask[:, col_idx] = rng.rand(n_rows) < rate
                continue

            # Normalize driver to [0, 1] range
            d_min = np.nanmin(driver_values[finite_mask])
            d_max = np.nanmax(driver_values[finite_mask])
            if d_max - d_min > 0:
                normalized = (driver_values - d_min) / (d_max - d_min)
            else:
                normalized = np.full(n_rows, 0.5)

            # Scale probabilities so that the overall rate approximately matches
            # p(missing) = rate * 2 * normalized  (capped at some max)
            probs = np.clip(rate * 2 * normalized, 0, min(rate * 3, 0.95))

            # Adjust to match target rate approximately
            # Scale factor so mean(probs) ≈ rate
            current_mean = probs[finite_mask].mean()
            if current_mean > 0:
                probs = probs * (rate / current_mean)
                probs = np.clip(probs, 0, 0.95)

            missing_mask[:, col_idx] = rng.rand(n_rows) < probs

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
                "mechanism": "MAR",
                "rate": rate,
                "seed": seed,
                "exclude_target": exclude_target,
                "strategy": "driver_value_dependent",
                "actual_missing_rate": float(missing_mask.sum()) / missing_mask.size,
            },
            scenario_tag=kwargs.get("scenario_tag", "benchmark_mar"),
        )
