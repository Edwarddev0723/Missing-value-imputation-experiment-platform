"""Breast Cancer Wisconsin (Original) dataset adapter — UCI (PN5)."""

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


@DATASET_REGISTRY.register("breast_cancer")
class BreastCancerDataset(BaseDataset):
    """UCI Breast Cancer Wisconsin (Original) dataset — PN5."""

    name = "breast_cancer"

    def __init__(self, data_dir: str = "data/raw", **kwargs: Any) -> None:
        self.data_dir = Path(data_dir)
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        cache_path = self.data_dir / "breast_cancer.csv"
        if cache_path.exists():
            logger.info(f"Loading Breast Cancer dataset from cache: {cache_path}")
            df = pd.read_csv(cache_path)
        else:
            logger.info("Downloading Breast Cancer Wisconsin dataset from OpenML...")
            data = fetch_openml(
                data_id=15, as_frame=True, parser="auto"
            )
            df = data.frame  # type: ignore
            self.data_dir.mkdir(parents=True, exist_ok=True)
            atomic_save_csv(df, cache_path)
            logger.info(f"Cached Breast Cancer dataset to {cache_path}")

        # Rename target
        if "Class" in df.columns:
            df = df.rename(columns={"Class": "class"})

        target_col = "class"
        # Binarize: benign=0, malignant=1  (or legacy 2=0, 4=1)
        if target_col in df.columns:
            col = df[target_col]
            if pd.api.types.is_string_dtype(col) or col.dtype.name == "category":
                # String-based labels
                df[target_col] = col.astype(str).str.strip().str.lower().map(
                    lambda x: 1 if x == "malignant" else 0
                ).astype(int)
            else:
                df[target_col] = pd.to_numeric(col, errors="coerce")
                df = df.dropna(subset=[target_col]).reset_index(drop=True)
                df[target_col] = (df[target_col] == 4).astype(int)

        # Drop rows with original missing values (spec says ~16 rows in Bare_Nuclei)
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

        feature_cols = [c for c in df.columns if c != "class"]
        feature_types = {c: NUMERICAL for c in feature_cols}

        return {
            "dataset_name": self.name,
            "dataset_id": "PN5",
            "n_samples": len(df),
            "n_features": len(feature_cols),
            "task_type": CLASSIFICATION,
            "target_col": "class",
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
            self._df, target_col="class", train_ratio=train_ratio,
            valid_ratio=valid_ratio, seed=seed,
        )
