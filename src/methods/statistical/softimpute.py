"""softImpute — nuclear-norm regularized matrix completion.

This is a simplified implementation using iterative SVD thresholding.
For production use, consider fancyimpute or a dedicated library.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

from src.constants import FAMILY_ML
from src.methods.base import BaseImputer
from src.registry import METHOD_REGISTRY
from src.utils.logging import get_logger

logger = get_logger(__name__)


@METHOD_REGISTRY.register("softimpute")
class SoftImputeImputer(BaseImputer):
    """Simplified softImpute via iterative SVD soft-thresholding.

    Algorithm:
        1. Initialize missing values (e.g., column means).
        2. Repeat until convergence:
            a. Compute SVD of current matrix.
            b. Soft-threshold singular values by lambda.
            c. Reconstruct matrix.
            d. Replace only missing positions with new values.
    """

    name = "softimpute"
    family = FAMILY_ML

    def __init__(
        self,
        max_rank: int = 50,
        lam: float = 0.0,
        max_iter: int = 100,
        tol: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_rank=max_rank, lam=lam, max_iter=max_iter, tol=tol, **kwargs)
        self.max_rank = max_rank
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self._fill_values: np.ndarray | None = None
        self._columns: list[str] = []

    def fit(self, X_train: pd.DataFrame, metadata: dict[str, Any] | None = None) -> None:
        """Learn column means from training data for initialization."""
        self._columns = list(X_train.columns)
        self._fill_values = np.nanmean(X_train.values, axis=0)

    def impute(self, X_missing: pd.DataFrame) -> pd.DataFrame:
        if self._fill_values is None:
            raise RuntimeError("SoftImputeImputer must be fit before impute.")

        X = X_missing.values.copy().astype(float)
        mask = np.isnan(X)

        # Initialize missing values with column means (vectorized)
        fill_matrix = np.broadcast_to(self._fill_values, X.shape)
        X[mask] = fill_matrix[mask]

        n_components = min(self.max_rank, min(X.shape) - 1)
        if n_components < 1:
            n_components = 1

        for iteration in range(self.max_iter):
            # SVD
            U, s, Vt = randomized_svd(X, n_components=n_components, random_state=self._seed)

            # Soft threshold
            s_thresh = np.maximum(s - self.lam, 0)

            # Reconstruct
            X_reconstructed = U @ np.diag(s_thresh) @ Vt

            # Convergence: norm of change at missing positions only
            delta = X_reconstructed[mask] - X[mask]
            diff = np.linalg.norm(delta) / (np.linalg.norm(X[mask]) + 1e-10)

            # Replace only missing positions
            X[mask] = X_reconstructed[mask]

            if diff < self.tol:
                logger.debug(f"softImpute converged at iteration {iteration + 1}")
                break

        return pd.DataFrame(X, columns=self._columns, index=X_missing.index)
