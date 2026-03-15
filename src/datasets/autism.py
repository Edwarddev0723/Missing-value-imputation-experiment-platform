"""Autism Screening Adult dataset adapter — UCI (MX5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.constants import CATEGORICAL, CLASSIFICATION, NUMERICAL
from src.datasets.base import BaseDataset
from src.datasets.utils import atomic_save_csv, stratified_split
from src.registry import DATASET_REGISTRY
from src.utils.logging import get_logger

logger = get_logger(__name__)


@DATASET_REGISTRY.register("autism")
class AutismDataset(BaseDataset):
    """UCI Autism Screening Adult dataset — MX5. Categorical-heavy."""

    name = "autism"

    def __init__(self, data_dir: str = "data/raw", **kwargs: Any) -> None:
        self.data_dir = Path(data_dir)
        self._df: pd.DataFrame | None = None
        self._num_cols: list[str] = []
        self._cat_cols: list[str] = []

    def load(self) -> pd.DataFrame:
        cache_path = self.data_dir / "autism.csv"
        if cache_path.exists():
            df = pd.read_csv(cache_path)
            # Detect wrong cache (car evaluation dataset cached by mistake)
            if "buying" in df.columns or "lug_boot" in df.columns:
                logger.warning("Cached autism.csv contains wrong dataset — re-downloading.")
                cache_path.unlink()
                df = None
            else:
                logger.info(f"Loading Autism dataset from cache: {cache_path}")

        if not cache_path.exists():
            logger.info("Downloading Autism Screening Adult dataset from UCI (id=426)...")
            from ucimlrepo import fetch_ucirepo
            data = fetch_ucirepo(id=426)
            df = data.data.original
            self.data_dir.mkdir(parents=True, exist_ok=True)
            atomic_save_csv(df, cache_path)
            logger.info(f"Cached Autism dataset to {cache_path}")

        # Standardize column names
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

        # Identify target
        target_col = "asd_class"
        if target_col not in df.columns:
            for c in df.columns:
                if "class" in c.lower() or "asd" in c.lower():
                    df = df.rename(columns={c: target_col})
                    break

        # Binarize target
        if target_col in df.columns:
            val = df[target_col].astype(str).str.strip().str.lower()
            df[target_col] = val.isin(["yes", "1", "true"]).astype(int)

        # Drop rows with missing values
        df = df.dropna().reset_index(drop=True)

        # Determine feature types
        feature_cols = [c for c in df.columns if c != target_col]
        num_cols = []
        cat_cols = []
        for col in feature_cols:
            try:
                numeric_vals = pd.to_numeric(df[col])
                if df[col].nunique() <= 10:
                    cat_cols.append(col)
                    df[col] = df[col].astype(str)
                else:
                    num_cols.append(col)
                    df[col] = numeric_vals
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
            "dataset_id": "MX5",
            "n_samples": len(df),
            "n_features": len(self._num_cols) + len(self._cat_cols),
            "task_type": CLASSIFICATION,
            "target_col": "asd_class",
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
            self._df, target_col="asd_class", train_ratio=train_ratio,
            valid_ratio=valid_ratio, seed=seed,
        )
