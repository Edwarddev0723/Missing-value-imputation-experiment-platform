"""Adult (Census Income) dataset adapter — UCI."""

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

ADULT_NUMERICAL = [
    "age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week",
]
ADULT_CATEGORICAL = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]


@DATASET_REGISTRY.register("adult")
class AdultDataset(BaseDataset):
    """UCI Adult (Census Income) dataset."""

    name = "adult"

    def __init__(self, data_dir: str = "data/raw", **kwargs: Any) -> None:
        self.data_dir = Path(data_dir)
        self._df: pd.DataFrame | None = None
        self._config = kwargs

    def load(self) -> pd.DataFrame:
        """Load Adult dataset via sklearn's fetch_openml, cache locally."""
        cache_path = self.data_dir / "adult.csv"
        if cache_path.exists():
            logger.info(f"Loading Adult dataset from cache: {cache_path}")
            df = pd.read_csv(cache_path)
        else:
            logger.info("Downloading Adult dataset from OpenML...")
            data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
            df = data.frame  # type: ignore[assignment]
            # Save cache
            self.data_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False)
            logger.info(f"Cached Adult dataset to {cache_path}")

        # Basic cleaning: rename target column if needed
        # OpenML uses 'class' as the target column name
        if "class" in df.columns and "income" not in df.columns:
            df = df.rename(columns={"class": "income"})

        target_col = "income"
        if target_col in df.columns:
            df = df.dropna(subset=[target_col]).reset_index(drop=True)
            # Binarize target
            df[target_col] = (df[target_col].astype(str).str.strip().str.replace(".", "", regex=False))
            df[target_col] = (df[target_col] == ">50K").astype(int)

        # Drop rows with original missing in features (for controlled benchmark)
        # Keep track of original missing rate before dropping
        feature_cols = ADULT_NUMERICAL + ADULT_CATEGORICAL
        existing_cols = [c for c in feature_cols if c in df.columns]
        df = df.dropna(subset=existing_cols).reset_index(drop=True)

        # Ensure numerical columns are float
        for col in ADULT_NUMERICAL:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Ensure categorical columns are string
        for col in ADULT_CATEGORICAL:
            if col in df.columns:
                df[col] = df[col].astype(str)

        self._df = df
        return df

    def get_metadata(self) -> dict[str, Any]:
        if self._df is None:
            self.load()
        df = self._df
        assert df is not None

        num_cols = [c for c in ADULT_NUMERICAL if c in df.columns]
        cat_cols = [c for c in ADULT_CATEGORICAL if c in df.columns]

        feature_types = {c: NUMERICAL for c in num_cols}
        feature_types.update({c: CATEGORICAL for c in cat_cols})

        return {
            "dataset_name": self.name,
            "n_samples": len(df),
            "n_features": len(num_cols) + len(cat_cols),
            "task_type": CLASSIFICATION,
            "target_col": "income",
            "numerical_features": num_cols,
            "categorical_features": cat_cols,
            "feature_types": feature_types,
            "source": "uci/openml",
        }

    def get_splits(
        self, seed: int = 42, train_ratio: float = 0.8, valid_ratio: float = 0.0
    ) -> dict[str, pd.DataFrame]:
        if self._df is None:
            self.load()
        return stratified_split(
            self._df, target_col="income", train_ratio=train_ratio,  # type: ignore
            valid_ratio=valid_ratio, seed=seed,
        )
