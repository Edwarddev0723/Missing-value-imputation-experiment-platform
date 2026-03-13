"""Base dataset adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from src.registry import DATASET_REGISTRY


class BaseDataset(ABC):
    """Abstract base class for all dataset adapters.

    Every dataset adapter must implement:
        - load(): Load raw data into a canonical DataFrame.
        - get_metadata(): Return dataset metadata dict.
        - get_splits(seed): Return train/valid/test DataFrames.
    """

    name: str = ""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load the raw dataset and return as a pandas DataFrame."""

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Return dataset metadata including feature types, target info, etc.

        Expected keys:
            - dataset_name: str
            - n_samples: int
            - n_features: int
            - task_type: str  ('classification' or 'regression')
            - target_col: str
            - numerical_features: list[str]
            - categorical_features: list[str]
            - feature_types: dict[str, str]  (col_name -> type)
        """

    @abstractmethod
    def get_splits(
        self, seed: int = 42, train_ratio: float = 0.7, valid_ratio: float = 0.1
    ) -> dict[str, pd.DataFrame]:
        """Return dict with keys 'train', 'valid', 'test', each a DataFrame."""

    def get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return feature columns (excluding target)."""
        meta = self.get_metadata()
        target = meta.get("target_col")
        if target and target in df.columns:
            return df.drop(columns=[target])
        return df

    def get_target(self, df: pd.DataFrame) -> pd.Series | None:
        """Return target column series."""
        meta = self.get_metadata()
        target = meta.get("target_col")
        if target and target in df.columns:
            return df[target]
        return None
