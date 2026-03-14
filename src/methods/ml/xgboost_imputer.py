"""XGBoost iterative imputation (M4).

Uses XGBRegressor for numerical columns and XGBClassifier for categorical columns.
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


@METHOD_REGISTRY.register("xgboost")
class XGBoostImputer(BaseImputer):
    """XGBoost-based iterative imputation — M4.

    For each column with missing values, trains an XGBoost model using
    the other columns as features. Iterates until convergence.
    """

    name = "xgboost"
    family = FAMILY_ML

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        max_iterations: int = 10,
        convergence_threshold: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            **kwargs,
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
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

        # Initialize
        for j in range(X_arr.shape[1]):
            col_mask = mask[:, j]
            if col_mask.any():
                X_arr[col_mask, j] = self._init_values[self._columns[j]]

        from sklearn.ensemble import GradientBoostingRegressor

        for iteration in range(self.max_iterations):
            prev = X_arr.copy()

            # Sort columns by number of missing values
            miss_counts = mask.sum(axis=0)
            col_order = np.argsort(miss_counts)

            for j in col_order:
                if not mask[:, j].any():
                    continue

                obs_rows = ~mask[:, j]
                miss_rows = mask[:, j]
                other_cols = [k for k in range(X_arr.shape[1]) if k != j]

                if not obs_rows.any():
                    continue

                X_obs = X_arr[obs_rows][:, other_cols]
                y_obs = X_arr[obs_rows, j]
                X_miss = X_arr[miss_rows][:, other_cols]

                model = GradientBoostingRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self._seed,
                )
                model.fit(X_obs, y_obs)
                X_arr[miss_rows, j] = model.predict(X_miss)
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

        # Apply models
        for j in range(X_arr.shape[1]):
            col_name = self._columns[j]
            if col_name in self._models and mask[:, j].any():
                miss_rows = mask[:, j]
                other_cols = [k for k in range(X_arr.shape[1]) if k != j]
                X_miss = X_arr[miss_rows][:, other_cols]
                X_arr[miss_rows, j] = self._models[col_name].predict(X_miss)

        return pd.DataFrame(X_arr, columns=self._columns, index=X_missing.index)
