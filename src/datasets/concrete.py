"""Concrete Compressive Strength dataset adapter — UCI (PN2)."""

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


@DATASET_REGISTRY.register("concrete")
class ConcreteDataset(BaseDataset):
    """UCI Concrete Compressive Strength dataset — PN2."""

    name = "concrete"

    def __init__(self, data_dir: str = "data/raw", **kwargs: Any) -> None:
        self.data_dir = Path(data_dir)
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        cache_path = self.data_dir / "concrete.csv"
        if cache_path.exists():
            logger.info(f"Loading Concrete dataset from cache: {cache_path}")
            df = pd.read_csv(cache_path)
        else:
            logger.info("Downloading Concrete dataset from OpenML...")
            data = fetch_openml(data_id=4353, as_frame=True, parser="auto")
            df = data.frame  # type: ignore
            self.data_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False)
            logger.info(f"Cached Concrete dataset to {cache_path}")

        # Standardize column names
        col_map = {}
        for col in df.columns:
            new_name = col.strip().lower().replace(" ", "_")
            col_map[col] = new_name
        df = df.rename(columns=col_map)

        # Ensure target column exists
        target_candidates = [c for c in df.columns if "strength" in c.lower() or "compressive" in c.lower()]
        if target_candidates:
            target_col = target_candidates[0]
            if target_col != "compressive_strength":
                df = df.rename(columns={target_col: "compressive_strength"})

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

        feature_cols = [c for c in df.columns if c != "compressive_strength"]
        feature_types = {c: NUMERICAL for c in feature_cols}

        return {
            "dataset_name": self.name,
            "dataset_id": "PN2",
            "n_samples": len(df),
            "n_features": len(feature_cols),
            "task_type": REGRESSION,
            "target_col": "compressive_strength",
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
            self._df, target_col=None, train_ratio=train_ratio,
            valid_ratio=valid_ratio, seed=seed,
        )
