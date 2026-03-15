"""MICE (Multiple Imputation by Chained Equations) — S4.

Uses sklearn's IterativeImputer with BayesianRidge as default estimator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

from src.constants import FAMILY_STATISTICAL
from src.methods.base import BaseImputer
from src.registry import METHOD_REGISTRY
from src.utils.logging import get_logger

logger = get_logger(__name__)


@METHOD_REGISTRY.register("mice")
class MICEImputer(BaseImputer):
    """MICE imputation — S4. Multiple imputation by chained equations."""

    name = "mice"
    family = FAMILY_STATISTICAL

    def __init__(
        self,
        max_iterations: int = 10,
        n_imputations: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            max_iterations=max_iterations,
            n_imputations=n_imputations,
            **kwargs,
        )
        self.max_iterations = max_iterations
        self.n_imputations = n_imputations
        self._imputers: list[IterativeImputer] = []
        self._columns: list[str] = []

    def fit(self, X_train: pd.DataFrame, metadata: dict[str, Any] | None = None) -> None:
        self._columns = list(X_train.columns)
        X_arr = X_train.values.astype(float)

        self._imputers = []
        for i in range(self.n_imputations):
            imp = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=self.max_iterations,
                random_state=self._seed + i,
                sample_posterior=True,
            )
            imp.fit(X_arr)
            self._imputers.append(imp)

    def impute(self, X_missing: pd.DataFrame) -> pd.DataFrame:
        if not self._imputers:
            raise RuntimeError("MICEImputer must be fit before impute.")

        X_arr = X_missing.values.astype(float)

        # Multiple imputations, then average
        imputed_arrays = []
        for imp in self._imputers:
            imputed_arrays.append(imp.transform(X_arr))

        X_result = np.mean(imputed_arrays, axis=0)
        return pd.DataFrame(X_result, columns=self._columns, index=X_missing.index)
