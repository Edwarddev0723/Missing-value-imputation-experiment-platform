"""KNN imputation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer as SklearnKNNImputer

from src.constants import FAMILY_ML
from src.methods.base import BaseImputer
from src.registry import METHOD_REGISTRY


@METHOD_REGISTRY.register("knn")
class KNNImputer(BaseImputer):
    """K-Nearest Neighbors imputation using sklearn."""

    name = "knn"
    family = FAMILY_ML

    def __init__(self, n_neighbors: int = 5, weights: str = "distance", **kwargs: Any) -> None:
        super().__init__(n_neighbors=n_neighbors, weights=weights, **kwargs)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._imputer: SklearnKNNImputer | None = None
        self._columns: list[str] = []

    def fit(self, X_train: pd.DataFrame, metadata: dict[str, Any] | None = None) -> None:
        self._columns = list(X_train.columns)
        self._imputer = SklearnKNNImputer(
            n_neighbors=self.n_neighbors, weights=self.weights
        )
        self._imputer.fit(X_train.values)

    def impute(self, X_missing: pd.DataFrame) -> pd.DataFrame:
        if self._imputer is None:
            raise RuntimeError("KNNImputer must be fit before impute.")
        imputed = self._imputer.transform(X_missing.values)
        return pd.DataFrame(imputed, columns=self._columns, index=X_missing.index)
