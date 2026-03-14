"""Spambase dataset adapter — UCI (PN3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import fetch_openml

from src.constants import NUMERICAL, CLASSIFICATION
from src.datasets.base import BaseDataset
from src.datasets.utils import stratified_split
from src.registry import DATASET_REGISTRY
from src.utils.logging import get_logger

logger = get_logger(__name__)


@DATASET_REGISTRY.register("spambase")
class SpambaseDataset(BaseDataset):
    """UCI Spambase dataset — PN3."""

    name = "spambase"

    def __init__(self, data_dir: str = "data/raw", **kwargs: Any) -> None:
        self.data_dir = Path(data_dir)
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        cache_path = self.data_dir / "spambase.csv"
        if cache_path.exists():
            logger.info(f"Loading Spambase dataset from cache: {cache_path}")
            df = pd.read_csv(cache_path)
        else:
            logger.info("Downloading Spambase dataset from OpenML...")
            data = fetch_openml("spambase", version=1, as_frame=True, parser="auto")
            df = data.frame  # type: ignore
            # OpenML target column is usually 'class'
            if "class" in df.columns:
                df = df.rename(columns={"class": "is_spam"})
            elif "Class" in df.columns:
                df = df.rename(columns={"Class": "is_spam"})
            self.data_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False)
            logger.info(f"Cached Spambase dataset to {cache_path}")

        # Ensure target is binary int
        target_col = "is_spam"
        if target_col in df.columns:
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce").astype(int)

        df = df.dropna().reset_index(drop=True)
        feature_cols = [c for c in df.columns if c != target_col]
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna().reset_index(drop=True)

        self._df = df
        return df

    def get_metadata(self) -> dict[str, Any]:
        if self._df is None:
            self.load()
        df = self._df
        assert df is not None

        feature_cols = [c for c in df.columns if c != "is_spam"]
        feature_types = {c: NUMERICAL for c in feature_cols}

        return {
            "dataset_name": self.name,
            "dataset_id": "PN3",
            "n_samples": len(df),
            "n_features": len(feature_cols),
            "task_type": CLASSIFICATION,
            "target_col": "is_spam",
            "numerical_features": feature_cols,
            "categorical_features": [],
            "feature_types": feature_types,
            "source": "uci/openml",
        }

    def get_splits(
        self, seed: int = 42, train_ratio: float = 0.8, valid_ratio: float = 0.0
    ) -> dict[str, pd.DataFrame]:
        if self._df is None:
            self.load()
        return stratified_split(
            self._df, target_col="is_spam", train_ratio=train_ratio,
            valid_ratio=valid_ratio, seed=seed,
        )
