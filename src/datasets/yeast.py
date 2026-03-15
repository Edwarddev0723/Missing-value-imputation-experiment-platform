"""Yeast dataset adapter — UCI (PN4)."""

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


@DATASET_REGISTRY.register("yeast")
class YeastDataset(BaseDataset):
    """UCI Yeast dataset — PN4. Multi-class classification."""

    name = "yeast"

    def __init__(self, data_dir: str = "data/raw", **kwargs: Any) -> None:
        self.data_dir = Path(data_dir)
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        cache_path = self.data_dir / "yeast.csv"
        if cache_path.exists():
            df = pd.read_csv(cache_path)
            # Detect wrong cache (multi-label yeast v4 with 103+ columns)
            if len(df.columns) > 20:
                logger.warning("Cached yeast.csv is the wrong dataset version — re-downloading.")
                cache_path.unlink()
                df = None
            else:
                logger.info(f"Loading Yeast dataset from cache: {cache_path}")

        if not cache_path.exists():
            logger.info("Downloading Yeast dataset from OpenML (data_id=181)...")
            data = fetch_openml(data_id=181, as_frame=True, parser="auto")
            df = data.frame  # type: ignore
            # Rename target
            target_candidates = ["class", "Class", "class_protein_localization"]
            for tc in target_candidates:
                if tc in df.columns:
                    df = df.rename(columns={tc: "localization_site"})
                    break
            self.data_dir.mkdir(parents=True, exist_ok=True)
            atomic_save_csv(df, cache_path)
            logger.info(f"Cached Yeast dataset to {cache_path}")

        target_col = "localization_site"
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

        feature_cols = [c for c in df.columns if c != "localization_site"]
        feature_types = {c: NUMERICAL for c in feature_cols}

        return {
            "dataset_name": self.name,
            "dataset_id": "PN4",
            "n_samples": len(df),
            "n_features": len(feature_cols),
            "task_type": CLASSIFICATION,
            "target_col": "localization_site",
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
            self._df, target_col="localization_site", train_ratio=train_ratio,
            valid_ratio=valid_ratio, seed=seed,
        )
