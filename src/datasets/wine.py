"""Wine Quality (White) dataset adapter — UCI (PN1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import fetch_openml

from src.constants import NUMERICAL, REGRESSION
from src.datasets.base import BaseDataset
from src.datasets.utils import stratified_split
from src.registry import DATASET_REGISTRY
from src.utils.logging import get_logger

logger = get_logger(__name__)


@DATASET_REGISTRY.register("wine")
class WineDataset(BaseDataset):
    """UCI Wine Quality dataset (White wine) — PN1."""

    name = "wine"

    def __init__(self, data_dir: str = "data/raw", **kwargs: Any) -> None:
        self.data_dir = Path(data_dir)
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        cache_path = self.data_dir / "wine_white.csv"
        if cache_path.exists():
            logger.info(f"Loading Wine White dataset from cache: {cache_path}")
            df = pd.read_csv(cache_path)
        else:
            logger.info("Downloading Wine Quality (White) dataset from OpenML...")
            data = fetch_openml("wine-quality-white", version=1, as_frame=True, parser="auto")
            df = data.frame  # type: ignore
            self.data_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False)
            logger.info(f"Cached Wine White dataset to {cache_path}")

        target_col = "quality"
        df = df.dropna().reset_index(drop=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna().reset_index(drop=True)

        self._df = df
        return df

    def get_metadata(self) -> dict[str, Any]:
        if self._df is None:
            self.load()
        df = self._df
        assert df is not None

        feature_cols = [c for c in df.columns if c != "quality"]
        feature_types = {c: NUMERICAL for c in feature_cols}

        return {
            "dataset_name": self.name,
            "dataset_id": "PN1",
            "n_samples": len(df),
            "n_features": len(feature_cols),
            "task_type": REGRESSION,
            "target_col": "quality",
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
            self._df, target_col=None, train_ratio=train_ratio,  # type: ignore
            valid_ratio=valid_ratio, seed=seed,
        )
