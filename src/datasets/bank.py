"""Bank Marketing dataset adapter — UCI (MX2)."""

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

BANK_NUMERICAL = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
BANK_CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]

# OpenML sometimes returns generic V1..V16 column names; map them back.
_BANK_COL_RENAME = {
    "V1": "age", "V2": "job", "V3": "marital", "V4": "education",
    "V5": "default", "V6": "balance", "V7": "housing", "V8": "loan",
    "V9": "contact", "V10": "day", "V11": "month", "V12": "duration",
    "V13": "campaign", "V14": "pdays", "V15": "previous", "V16": "poutcome",
    "Class": "deposit",
}


@DATASET_REGISTRY.register("bank")
class BankDataset(BaseDataset):
    """UCI Bank Marketing dataset — MX2."""

    name = "bank"

    def __init__(self, data_dir: str = "data/raw", **kwargs: Any) -> None:
        self.data_dir = Path(data_dir)
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        cache_path = self.data_dir / "bank.csv"
        if cache_path.exists():
            logger.info(f"Loading Bank Marketing dataset from cache: {cache_path}")
            df = pd.read_csv(cache_path)
        else:
            logger.info("Downloading Bank Marketing dataset from OpenML...")
            data = fetch_openml("bank-marketing", version=1, as_frame=True, parser="auto")
            df = data.frame  # type: ignore
            self.data_dir.mkdir(parents=True, exist_ok=True)
            atomic_save_csv(df, cache_path)
            logger.info(f"Cached Bank Marketing dataset to {cache_path}")

        # Rename generic OpenML column names if present
        if "V1" in df.columns:
            df = df.rename(columns=_BANK_COL_RENAME)

        # Identify and rename target
        target_candidates = ["y", "Class", "class", "deposit"]
        target_col = "deposit"
        for tc in target_candidates:
            if tc in df.columns:
                if tc != target_col:
                    df = df.rename(columns={tc: target_col})
                break

        # Binarize target
        if target_col in df.columns:
            df[target_col] = (df[target_col].astype(str).str.strip().str.lower().isin(["yes", "1", "2"])).astype(int)

        # Drop rows with missing values
        df = df.dropna().reset_index(drop=True)

        # Ensure numerical columns are float
        for col in BANK_NUMERICAL:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Ensure categorical columns are string
        for col in BANK_CATEGORICAL:
            if col in df.columns:
                df[col] = df[col].astype(str)

        df = df.dropna().reset_index(drop=True)
        self._df = df
        return df

    def get_metadata(self) -> dict[str, Any]:
        if self._df is None:
            self.load()
        df = self._df
        assert df is not None

        num_cols = [c for c in BANK_NUMERICAL if c in df.columns]
        cat_cols = [c for c in BANK_CATEGORICAL if c in df.columns]

        feature_types = {c: NUMERICAL for c in num_cols}
        feature_types.update({c: CATEGORICAL for c in cat_cols})

        return {
            "dataset_name": self.name,
            "dataset_id": "MX2",
            "n_samples": len(df),
            "n_features": len(num_cols) + len(cat_cols),
            "task_type": CLASSIFICATION,
            "target_col": "deposit",
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
            self._df, target_col="deposit", train_ratio=train_ratio,
            valid_ratio=valid_ratio, seed=seed,
        )
