"""Expectation-Maximization (EM) imputation (S3).

Numerical only — uses Gaussian assumption. For mixed datasets,
categorical columns use mode fallback.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.constants import FAMILY_STATISTICAL
from src.methods.base import BaseImputer
from src.registry import METHOD_REGISTRY
from src.utils.logging import get_logger

logger = get_logger(__name__)


@METHOD_REGISTRY.register("em")
class EMImputer(BaseImputer):
    """EM (Expectation-Maximization) imputation — S3.

    Iteratively estimates mean and covariance, then imputes missing values
    by conditional expectations. Categorical columns get mode fallback.
    """

    name = "em"
    family = FAMILY_STATISTICAL

    def __init__(
        self,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            **kwargs,
        )
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self._mean: np.ndarray | None = None
        self._cov: np.ndarray | None = None
        self._columns: list[str] = []
        self._metadata: dict[str, Any] = {}
        self._cat_fill: dict[str, Any] = {}

    def fit(self, X_train: pd.DataFrame, metadata: dict[str, Any] | None = None) -> None:
        self._metadata = metadata or {}
        self._columns = list(X_train.columns)
        numerical_cols = self._metadata.get("numerical_features", [])
        categorical_cols = self._metadata.get("categorical_features", [])

        # Mode fallback for categorical
        for col in categorical_cols:
            if col in X_train.columns:
                mode = X_train[col].mode()
                self._cat_fill[col] = mode.iloc[0] if len(mode) > 0 else 0

        # EM on numerical columns
        num_col_indices = [X_train.columns.get_loc(c) for c in numerical_cols if c in X_train.columns]
        if not num_col_indices:
            # No numerical columns — just store means
            self._mean = np.nanmean(X_train.values.astype(float), axis=0)
            return

        X_num = X_train.iloc[:, num_col_indices].values.astype(float)
        n, p = X_num.shape

        # Initialize with column means
        col_means = np.nanmean(X_num, axis=0)
        X_filled = X_num.copy()
        for j in range(p):
            mask_j = np.isnan(X_filled[:, j])
            X_filled[mask_j, j] = col_means[j]

        # EM iterations
        for it in range(self.max_iterations):
            # M step: estimate mean and covariance
            mu = np.mean(X_filled, axis=0)
            centered = X_filled - mu
            cov = (centered.T @ centered) / n
            # Regularize covariance
            cov += np.eye(p) * 1e-6

            # E step: impute missing with conditional expectations
            X_new = X_filled.copy()
            mask = np.isnan(X_num)

            for i in range(n):
                miss_idx = np.where(mask[i])[0]
                obs_idx = np.where(~mask[i])[0]
                if len(miss_idx) == 0:
                    continue
                if len(obs_idx) == 0:
                    X_new[i, miss_idx] = mu[miss_idx]
                    continue

                cov_mo = cov[np.ix_(miss_idx, obs_idx)]
                cov_oo = cov[np.ix_(obs_idx, obs_idx)]
                try:
                    cov_oo_inv = np.linalg.solve(
                        cov_oo, np.eye(len(obs_idx))
                    )
                    X_new[i, miss_idx] = mu[miss_idx] + cov_mo @ cov_oo_inv @ (
                        X_num[i, obs_idx] - mu[obs_idx]
                    )
                except np.linalg.LinAlgError:
                    X_new[i, miss_idx] = mu[miss_idx]

            # Check convergence
            diff = np.linalg.norm(X_new - X_filled) / (np.linalg.norm(X_filled) + 1e-10)
            X_filled = X_new
            if diff < self.convergence_threshold:
                logger.debug(f"EM converged at iteration {it + 1}")
                break

        self._mean = mu
        self._cov = cov
        self._num_col_indices = num_col_indices

    def impute(self, X_missing: pd.DataFrame) -> pd.DataFrame:
        df = X_missing.copy()

        # Categorical: mode fill
        for col, val in self._cat_fill.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)

        # Numerical: EM conditional expectation
        if self._mean is not None and self._cov is not None:
            num_cols = self._metadata.get("numerical_features", [])
            num_col_indices = [df.columns.get_loc(c) for c in num_cols if c in df.columns]
            if num_col_indices:
                X = df.iloc[:, num_col_indices].values.astype(float)
                mask = np.isnan(X)
                p = X.shape[1]

                for i in range(X.shape[0]):
                    miss_idx = np.where(mask[i])[0]
                    obs_idx = np.where(~mask[i])[0]
                    if len(miss_idx) == 0:
                        continue
                    if len(obs_idx) == 0:
                        X[i, miss_idx] = self._mean[miss_idx]
                        continue
                    cov_mo = self._cov[np.ix_(miss_idx, obs_idx)]
                    cov_oo = self._cov[np.ix_(obs_idx, obs_idx)]
                    try:
                        cov_oo_inv = np.linalg.solve(
                            cov_oo, np.eye(len(obs_idx))
                        )
                        X[i, miss_idx] = self._mean[miss_idx] + cov_mo @ cov_oo_inv @ (
                            X[i, obs_idx] - self._mean[obs_idx]
                        )
                    except np.linalg.LinAlgError:
                        X[i, miss_idx] = self._mean[miss_idx]

                for j_idx, j_col in enumerate(num_col_indices):
                    df.iloc[:, j_col] = X[:, j_idx]

        # Fill any remaining NaN with column means
        for col in df.columns:
            if df[col].isna().any():
                if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    df[col] = df[col].fillna(df[col].mean() if not df[col].isna().all() else 0)
                else:
                    mode = df[col].mode()
                    df[col] = df[col].fillna(mode.iloc[0] if len(mode) > 0 else 0)

        return df
