"""Canonical preprocessing: basic data cleaning and encoding."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from src.utils.logging import get_logger

logger = get_logger(__name__)


class CanonicalPreprocessor:
    """Standard preprocessing applied before missingness injection.

    Handles:
        - Drop duplicates (optional)
        - Encode categoricals to integers (ordinal encoding for tree methods)
        - Record encoders for inverse transform
        - Optionally scale numerical features
    """

    def __init__(
        self,
        drop_duplicates: bool = True,
        encode_categoricals: bool = True,
        scale_numericals: bool = False,
    ) -> None:
        self.drop_duplicates = drop_duplicates
        self.encode_categoricals = encode_categoricals
        self.scale_numericals = scale_numericals
        self._cat_encoders: dict[str, OrdinalEncoder] = {}
        self._num_scaler: StandardScaler | None = None
        self._fitted = False

    def fit(self, X: pd.DataFrame, metadata: dict[str, Any]) -> "CanonicalPreprocessor":
        """Fit encoders/scalers on complete training data."""
        numerical_cols = metadata.get("numerical_features", [])
        categorical_cols = metadata.get("categorical_features", [])

        if self.encode_categoricals:
            for col in categorical_cols:
                if col in X.columns:
                    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                    enc.fit(X[[col]].astype(str))
                    self._cat_encoders[col] = enc

        if self.scale_numericals:
            num_cols = [c for c in numerical_cols if c in X.columns]
            if num_cols:
                self._num_scaler = StandardScaler()
                self._num_scaler.fit(X[num_cols])

        self._metadata = metadata
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted encoders/scalers."""
        if not self._fitted:
            raise RuntimeError("CanonicalPreprocessor must be fit before transform.")

        df = X.copy()

        if self.drop_duplicates:
            df = df.drop_duplicates().reset_index(drop=True)

        if self.encode_categoricals:
            for col, enc in self._cat_encoders.items():
                if col in df.columns:
                    df[col] = enc.transform(df[[col]].astype(str)).ravel()

        if self.scale_numericals and self._num_scaler is not None:
            numerical_cols = self._metadata.get("numerical_features", [])
            num_cols = [c for c in numerical_cols if c in df.columns]
            if num_cols:
                df[num_cols] = self._num_scaler.transform(df[num_cols])

        return df

    def fit_transform(self, X: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, metadata)
        return self.transform(X)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform (mainly for numeric scaling)."""
        df = X.copy()
        if self.scale_numericals and self._num_scaler is not None:
            numerical_cols = self._metadata.get("numerical_features", [])
            num_cols = [c for c in numerical_cols if c in df.columns]
            if num_cols:
                df[num_cols] = self._num_scaler.inverse_transform(df[num_cols])
        return df
