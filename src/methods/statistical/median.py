"""Median imputation (S2)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.constants import FAMILY_STATISTICAL
from src.methods.base import BaseImputer
from src.registry import METHOD_REGISTRY


@METHOD_REGISTRY.register("median")
class MedianImputer(BaseImputer):
    """Median (numerical) / mode (categorical) imputation — S2."""

    name = "median"
    family = FAMILY_STATISTICAL

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._fill_values: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}

    def fit(self, X_train: pd.DataFrame, metadata: dict[str, Any] | None = None) -> None:
        self._metadata = metadata or {}
        numerical_cols = self._metadata.get("numerical_features", [])
        categorical_cols = self._metadata.get("categorical_features", [])

        for col in X_train.columns:
            if col in numerical_cols:
                self._fill_values[col] = X_train[col].median()
            elif col in categorical_cols:
                mode = X_train[col].mode()
                self._fill_values[col] = mode.iloc[0] if len(mode) > 0 else 0
            else:
                try:
                    self._fill_values[col] = X_train[col].median()
                except TypeError:
                    mode = X_train[col].mode()
                    self._fill_values[col] = mode.iloc[0] if len(mode) > 0 else 0

    def impute(self, X_missing: pd.DataFrame) -> pd.DataFrame:
        return X_missing.fillna(self._fill_values)
