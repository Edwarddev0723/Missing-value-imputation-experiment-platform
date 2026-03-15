"""Heart Disease dataset adapter — UCI."""

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

HEART_NUMERICAL = ["age", "trestbps", "chol", "thalach", "oldpeak"]
HEART_CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


@DATASET_REGISTRY.register("heart")
class HeartDataset(BaseDataset):
    """UCI Heart Disease dataset (Cleveland)."""

    name = "heart"

    def __init__(self, data_dir: str = "data/raw", **kwargs: Any) -> None:
        self.data_dir = Path(data_dir)
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        cache_path = self.data_dir / "heart.csv"
        if cache_path.exists():
            logger.info(f"Loading Heart dataset from cache: {cache_path}")
            df = pd.read_csv(cache_path)
        else:
            logger.info("Downloading Heart Disease dataset from OpenML...")
            # Use the 'heart-statlog' dataset (simplified, no missing values)
            data = fetch_openml("heart-statlog", version=1, as_frame=True, parser="auto")
            df = data.frame  # type: ignore
            # The target is typically the last column
            if "class" in df.columns:
                df = df.rename(columns={"class": "target"})
            self.data_dir.mkdir(parents=True, exist_ok=True)
            atomic_save_csv(df, cache_path)

        target_col = "target" if "target" in df.columns else df.columns[-1]
        df = df.dropna().reset_index(drop=True)

        # Binarize target if needed
        if target_col in df.columns:
            unique_vals = df[target_col].unique()
            if len(unique_vals) == 2:
                df[target_col] = pd.factorize(df[target_col])[0]

        # Coerce numericals
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

        self._df = df
        self._target_col = target_col
        return df

    def get_metadata(self) -> dict[str, Any]:
        if self._df is None:
            self.load()
        df = self._df
        assert df is not None

        target_col = getattr(self, "_target_col", "target")
        feature_cols = [c for c in df.columns if c != target_col]

        # Identify numerical vs categorical by dtype
        num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]

        feature_types = {c: NUMERICAL for c in num_cols}
        feature_types.update({c: CATEGORICAL for c in cat_cols})

        return {
            "dataset_name": self.name,
            "n_samples": len(df),
            "n_features": len(feature_cols),
            "task_type": CLASSIFICATION,
            "target_col": target_col,
            "numerical_features": num_cols,
            "categorical_features": cat_cols,
            "feature_types": feature_types,
            "source": "uci/openml",
        }

    def get_splits(
        self, seed: int = 42, train_ratio: float = 0.7, valid_ratio: float = 0.1
    ) -> dict[str, pd.DataFrame]:
        if self._df is None:
            self.load()
        target_col = getattr(self, "_target_col", "target")
        return stratified_split(
            self._df, target_col=target_col, train_ratio=train_ratio,  # type: ignore
            valid_ratio=valid_ratio, seed=seed,
        )
