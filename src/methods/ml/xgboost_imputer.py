"""XGBoost iterative imputation (M4).

Uses XGBRegressor when available, falls back to HistGradientBoostingRegressor.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.constants import FAMILY_ML
from src.methods.base import BaseImputer
from src.registry import METHOD_REGISTRY
from src.utils.logging import get_logger

logger = get_logger(__name__)

# XGBoost availability is checked lazily in _make_gb_model() to avoid loading
# OpenMP at import time, which conflicts with PyTorch MPS on Apple Silicon.


def _make_gb_model(
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    seed: int,
    n_jobs: int = 1,
) -> Any:
    """Create the best available gradient boosting regressor."""
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
            n_jobs=n_jobs,
            tree_method="hist",
            verbosity=0,
        )
    except ImportError:
        pass
    # Fallback: HistGradientBoostingRegressor (~10x faster than GBR)
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_iter=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
        verbose=0,
    )


@METHOD_REGISTRY.register("xgboost")
class XGBoostImputer(BaseImputer):
    """XGBoost-based iterative imputation — M4.

    For each column with missing values, trains a gradient boosting model using
    the other columns as features. Iterates until convergence.

    Uses xgboost.XGBRegressor if installed, otherwise falls back to
    sklearn.ensemble.HistGradientBoostingRegressor.
    """

    name = "xgboost"
    family = FAMILY_ML

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        max_iterations: int = 5,
        convergence_threshold: float = 1e-3,
        n_jobs: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            n_jobs=n_jobs,
            **kwargs,
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.n_jobs = n_jobs
        self._columns: list[str] = []
        self._metadata: dict[str, Any] = {}
        self._models: dict[str, Any] = {}
        self._init_values: dict[str, float] = {}

    def fit(self, X_train: pd.DataFrame, metadata: dict[str, Any] | None = None) -> None:
        self._metadata = metadata or {}
        self._columns = list(X_train.columns)

        # Compute initial fill values
        for col in self._columns:
            val = X_train[col].mean()
            self._init_values[col] = val if not np.isnan(val) else 0.0

        # Iterative fitting
        X_arr = X_train.values.astype(float).copy()
        mask = np.isnan(X_arr)

        if not mask.any():
            return

        # Initialize missing values with column means
        for j in range(X_arr.shape[1]):
            col_mask = mask[:, j]
            if col_mask.any():
                X_arr[col_mask, j] = self._init_values[self._columns[j]]

        # Pre-compute column order once (sort by missing count)
        miss_counts = mask.sum(axis=0)
        col_order = np.argsort(miss_counts)
        # Pre-compute per-column indices
        other_cols_map = {
            j: [k for k in range(X_arr.shape[1]) if k != j] for j in col_order if mask[:, j].any()
        }

        try:
            import xgboost  # noqa: F401
            backend = "xgboost"
        except ImportError:
            backend = "HistGradientBoosting"
        logger.debug(f"XGBoostImputer using backend: {backend}")

        for iteration in range(self.max_iterations):
            prev = X_arr.copy()

            for j in col_order:
                if not mask[:, j].any():
                    continue

                obs_rows = ~mask[:, j]
                miss_rows = mask[:, j]
                other_cols = other_cols_map[j]

                if not obs_rows.any():
                    continue

                model = _make_gb_model(
                    self.n_estimators, self.max_depth,
                    self.learning_rate, self._seed, self.n_jobs,
                )
                model.fit(X_arr[obs_rows][:, other_cols], X_arr[obs_rows, j])
                X_arr[miss_rows, j] = model.predict(X_arr[miss_rows][:, other_cols])
                self._models[self._columns[j]] = model

            diff = np.linalg.norm(X_arr - prev) / (np.linalg.norm(X_arr) + 1e-10)
            if diff < self.convergence_threshold:
                logger.debug(f"XGBoost imputer converged at iteration {iteration + 1}")
                break

    def impute(self, X_missing: pd.DataFrame) -> pd.DataFrame:
        X_arr = X_missing.values.astype(float).copy()
        mask = np.isnan(X_arr)

        # Initial fill
        for j in range(X_arr.shape[1]):
            col_mask = mask[:, j]
            if col_mask.any():
                X_arr[col_mask, j] = self._init_values.get(self._columns[j], 0.0)

        # Apply fitted models
        for j in range(X_arr.shape[1]):
            col_name = self._columns[j]
            if col_name in self._models and mask[:, j].any():
                miss_rows = mask[:, j]
                other_cols = [k for k in range(X_arr.shape[1]) if k != j]
                X_arr[miss_rows, j] = self._models[col_name].predict(
                    X_arr[miss_rows][:, other_cols]
                )

        return pd.DataFrame(X_arr, columns=self._columns, index=X_missing.index)
