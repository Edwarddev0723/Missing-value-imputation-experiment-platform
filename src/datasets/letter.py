"""Letter Recognition dataset adapter — UCI (PN6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import fetch_openml

from src.constants import NUMERICAL, CLASSIFICATION
from src.datasets.base import BaseDataset
from src.datasets.utils import atomic_save_csv, stratified_split
from src.registry import DATASET_REGISTRY
from src.utils.logging import get_logger

logger = get_logger(__name__)


@DATASET_REGISTRY.register("letter")
class LetterDataset(BaseDataset):
    """UCI Letter Recognition dataset — PN6. Multi-class classification."""

    name = "letter"

    def __init__(self, data_dir: str = "data/raw", **kwargs: Any) -> None:
        self.data_dir = Path(data_dir)
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        cache_path = self.data_dir / "letter.csv"
        if cache_path.exists():
            logger.info(f"Loading Letter Recognition dataset from cache: {cache_path}")
            df = pd.read_csv(cache_path)
        else:
            logger.info("Downloading Letter Recognition dataset from OpenML...")
            data = fetch_openml("letter", version=1, as_frame=True, parser="auto")
            df = data.frame  # type: ignore
            if "class" in df.columns:
                df = df.rename(columns={"class": "letter"})
            elif "lettr" in df.columns:
                df = df.rename(columns={"lettr": "letter"})
            self.data_dir.mkdir(parents=True, exist_ok=True)
            atomic_save_csv(df, cache_path)
            logger.info(f"Cached Letter Recognition dataset to {cache_path}")

        target_col = "letter"
        # Encode target to integers
        if target_col in df.columns:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df[target_col] = le.fit_transform(df[target_col].astype(str))

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

        feature_cols = [c for c in df.columns if c != "letter"]
        feature_types = {c: NUMERICAL for c in feature_cols}

        return {
            "dataset_name": self.name,
            "dataset_id": "PN6",
            "n_samples": len(df),
            "n_features": len(feature_cols),
            "task_type": CLASSIFICATION,
            "target_col": "letter",
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
            self._df, target_col="letter", train_ratio=train_ratio,
            valid_ratio=valid_ratio, seed=seed,
        )
