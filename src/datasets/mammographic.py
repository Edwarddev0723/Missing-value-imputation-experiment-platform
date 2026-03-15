"""Mammographic Masses dataset adapter — UCI (MX4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import fetch_openml

from src.constants import CATEGORICAL, CLASSIFICATION, NUMERICAL
from src.datasets.base import BaseDataset
from src.datasets.utils import atomic_save_csv, stratified_split
from src.registry import DATASET_REGISTRY
from src.utils.logging import get_logger

logger = get_logger(__name__)

MAMMO_NUMERICAL = ["age", "density"]
MAMMO_CATEGORICAL = ["bi_rads", "shape", "margin"]


@DATASET_REGISTRY.register("mammographic")
class MammographicDataset(BaseDataset):
    """UCI Mammographic Masses dataset — MX4."""

    name = "mammographic"

    def __init__(self, data_dir: str = "data/raw", **kwargs: Any) -> None:
        self.data_dir = Path(data_dir)
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        cache_path = self.data_dir / "mammographic.csv"
        if cache_path.exists():
            logger.info(f"Loading Mammographic dataset from cache: {cache_path}")
            df = pd.read_csv(cache_path)
        else:
            logger.info("Downloading Mammographic Masses dataset from OpenML...")
            data = fetch_openml(data_id=310, as_frame=True, parser="auto")
            df = data.frame  # type: ignore
            self.data_dir.mkdir(parents=True, exist_ok=True)
            atomic_save_csv(df, cache_path)
            logger.info(f"Cached Mammographic dataset to {cache_path}")

        # Standardize column names
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

        # Identify target - severity
        target_col = "severity"
        if target_col not in df.columns:
            # May be called "class" in OpenML
            for c in df.columns:
                if c in ("class", "Class"):
                    df = df.rename(columns={c: target_col})
                    break

        if target_col in df.columns:
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
            df = df.dropna(subset=[target_col])
            df[target_col] = df[target_col].astype(int)

        # Drop rows with missing values
        df = df.dropna().reset_index(drop=True)

        # Determine feature types from actual data
        feature_cols = [c for c in df.columns if c != target_col]
        num_cols = []
        cat_cols = []
        for col in feature_cols:
            unique_count = df[col].nunique()
            try:
                df[col] = pd.to_numeric(df[col])
                if unique_count <= 10:
                    cat_cols.append(col)
                    df[col] = df[col].astype(str)
                else:
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
            "dataset_id": "MX4",
            "n_samples": len(df),
            "n_features": len(self._num_cols) + len(self._cat_cols),
            "task_type": CLASSIFICATION,
            "target_col": "severity",
            "numerical_features": self._num_cols,
            "categorical_features": self._cat_cols,
            "feature_types": feature_types,
            "source": "uci/openml",
        }

    def get_splits(
        self, seed: int = 42, train_ratio: float = 0.8, valid_ratio: float = 0.0
    ) -> dict[str, pd.DataFrame]:
        if self._df is None:
            self.load()
        return stratified_split(
            self._df, target_col="severity", train_ratio=train_ratio,
            valid_ratio=valid_ratio, seed=seed,
        )
