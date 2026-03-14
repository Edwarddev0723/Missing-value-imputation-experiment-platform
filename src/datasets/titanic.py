"""Titanic dataset adapter — Kaggle (MX3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import fetch_openml

from src.constants import CATEGORICAL, CLASSIFICATION, NUMERICAL
from src.datasets.base import BaseDataset
from src.datasets.utils import stratified_split
from src.registry import DATASET_REGISTRY
from src.utils.logging import get_logger

logger = get_logger(__name__)

TITANIC_NUMERICAL = ["age", "sibsp", "parch", "fare", "body"]
TITANIC_CATEGORICAL = ["pclass", "sex", "embarked", "boat", "home.dest"]


@DATASET_REGISTRY.register("titanic")
class TitanicDataset(BaseDataset):
    """Titanic dataset — MX3. Remove rows with missing Age/Cabin/Embarked."""

    name = "titanic"

    def __init__(self, data_dir: str = "data/raw", **kwargs: Any) -> None:
        self.data_dir = Path(data_dir)
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        cache_path = self.data_dir / "titanic.csv"
        if cache_path.exists():
            logger.info(f"Loading Titanic dataset from cache: {cache_path}")
            df = pd.read_csv(cache_path)
        else:
            logger.info("Downloading Titanic dataset from OpenML...")
            data = fetch_openml("titanic", version=1, as_frame=True, parser="auto")
            df = data.frame  # type: ignore
            self.data_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False)
            logger.info(f"Cached Titanic dataset to {cache_path}")

        # Standardize column names to lowercase
        df.columns = [c.lower().strip() for c in df.columns]

        # Identify target
        target_col = "survived"
        if target_col not in df.columns:
            for c in df.columns:
                if "surviv" in c.lower():
                    df = df.rename(columns={c: target_col})
                    break

        # Binarize target
        if target_col in df.columns:
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
            df = df.dropna(subset=[target_col])
            df[target_col] = df[target_col].astype(int)

        # Remove rows with missing values to get complete subset
        # Spec: Age, Cabin, Embarked have natural missing
        df = df.dropna().reset_index(drop=True)

        # Drop non-informative columns (name, ticket, cabin can be noisy)
        drop_cols = [c for c in ["name", "ticket", "cabin"] if c in df.columns]
        df = df.drop(columns=drop_cols, errors="ignore")

        # Determine actual numerical and categorical features
        feature_cols = [c for c in df.columns if c != target_col]
        num_cols = []
        cat_cols = []
        for col in feature_cols:
            try:
                df[col] = pd.to_numeric(df[col])
                num_cols.append(col)
            except (ValueError, TypeError):
                cat_cols.append(col)
                df[col] = df[col].astype(str)

        df = df.dropna().reset_index(drop=True)
        self._df = df
        self._num_cols = num_cols
        self._cat_cols = cat_cols
        return df

    def get_metadata(self) -> dict[str, Any]:
        if self._df is None:
            self.load()
        df = self._df
        assert df is not None

        feature_types = {c: NUMERICAL for c in self._num_cols}
        feature_types.update({c: CATEGORICAL for c in self._cat_cols})

        return {
            "dataset_name": self.name,
            "dataset_id": "MX3",
            "n_samples": len(df),
            "n_features": len(self._num_cols) + len(self._cat_cols),
            "task_type": CLASSIFICATION,
            "target_col": "survived",
            "numerical_features": self._num_cols,
            "categorical_features": self._cat_cols,
            "feature_types": feature_types,
            "source": "kaggle/openml",
        }

    def get_splits(
        self, seed: int = 42, train_ratio: float = 0.8, valid_ratio: float = 0.0
    ) -> dict[str, pd.DataFrame]:
        if self._df is None:
            self.load()
        return stratified_split(
            self._df, target_col="survived", train_ratio=train_ratio,
            valid_ratio=valid_ratio, seed=seed,
        )
