# Adding a New Dataset

This guide walks you through adding a new dataset to ImputeBench. Datasets use a **registry pattern** — implement a class, decorate it, and the pipeline picks it up automatically.

## Step 1: Implement the Dataset Adapter

Create a new file in `src/datasets/`:

```python
# src/datasets/my_dataset.py

from __future__ import annotations
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import fetch_openml

from src.constants import NUMERICAL, CATEGORICAL, CLASSIFICATION
from src.datasets.base import BaseDataset
from src.datasets.utils import atomic_save_csv, stratified_split
from src.registry import DATASET_REGISTRY
from src.utils.logging import get_logger

logger = get_logger(__name__)


@DATASET_REGISTRY.register("my_dataset")
class MyDataset(BaseDataset):
    """Brief description — source and ID."""

    name = "my_dataset"

    def __init__(self, data_dir: str = "data/raw", **kwargs: Any) -> None:
        self.data_dir = Path(data_dir)
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        cache_path = self.data_dir / "my_dataset.csv"

        if cache_path.exists():
            logger.info(f"Loading from cache: {cache_path}")
            df = pd.read_csv(cache_path)
        else:
            logger.info("Downloading from OpenML...")
            data = fetch_openml("my-dataset-name", version=1, as_frame=True, parser="auto")
            df = data.frame
            self.data_dir.mkdir(parents=True, exist_ok=True)
            atomic_save_csv(df, cache_path)

        # Remove rows with native missing values (create clean ground truth)
        df = df.dropna().reset_index(drop=True)

        self._df = df
        return df

    def get_metadata(self) -> dict[str, Any]:
        if self._df is None:
            self.load()
        df = self._df

        numerical = ["col_a", "col_b", "col_c"]
        categorical = ["col_x", "col_y"]
        feature_cols = numerical + categorical
        feature_types = {
            **{c: NUMERICAL for c in numerical},
            **{c: CATEGORICAL for c in categorical},
        }

        return {
            "dataset_name": self.name,
            "n_samples": len(df),
            "n_features": len(feature_cols),
            "task_type": CLASSIFICATION,     # or REGRESSION
            "target_col": "target",
            "numerical_features": numerical,
            "categorical_features": categorical,
            "feature_types": feature_types,
            "source": "uci/openml",
        }

    def get_splits(
        self, seed: int = 42, train_ratio: float = 0.8, valid_ratio: float = 0.0
    ) -> dict[str, pd.DataFrame]:
        if self._df is None:
            self.load()
        return stratified_split(
            self._df, target_col="target",  # Set to None for regression
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            seed=seed,
        )
```

### The `BaseDataset` API

| Method | Required | Description |
|--------|----------|-------------|
| `load()` | Yes | Download/cache the data, return a clean `DataFrame` (no NaN). |
| `get_metadata()` | Yes | Return dict with column types, target, task type. |
| `get_splits(seed, train_ratio, valid_ratio)` | Yes | Return `{"train": df, "test": df}` split dict. |

### Important Conventions

1. **Remove native missing values** — `df.dropna()` to create a clean ground truth. The platform injects synthetic missingness.
2. **Don't encode or scale** — the pipeline handles preprocessing (label encoding, min-max scaling) automatically.
3. **Cache locally** — download from OpenML/UCI on first use, save to `data/raw/` for subsequent runs.
4. **Use `atomic_save_csv()`** — prevents corrupted caches from partial writes.

## Step 2: Create a YAML Config

```yaml
# configs/datasets/my_dataset.yaml

dataset:
  name: my_dataset
  source: uci
  target_col: target
  task_type: classification  # or regression
  split:
    train_ratio: 0.8
    valid_ratio: 0.0
    test_ratio: 0.2
    stratify: true            # true for classification datasets
```

## Step 3: Register the Import

Add your module to `src/factory.py`:

```python
# In src/factory.py, add with the other dataset imports:
import src.datasets.my_dataset  # noqa: F401
```

## Step 4: Add to Experiment Configs

Reference your dataset in a batch experiment config:

```yaml
# In configs/experiments/my_benchmark.yaml
datasets:
  - configs/datasets/my_dataset.yaml
  - configs/datasets/wine.yaml  # existing datasets for comparison
```

Or test standalone:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/single_adult_mean.yaml \
  --no-wandb
```

(Modify the config to point to your dataset.)

## Step 5: Add to Analysis Configs

To include your dataset in the analysis pipeline, update these files:

### `analysis/configs/dataset_track_mapping.yaml`

```yaml
# Add your dataset to the appropriate track
pure_numerical:
  - wine
  - concrete
  # ...

mixed_type:
  - adult
  - my_dataset    # Add here if mixed-type
```

### `analysis/configs/method_eligibility.yaml`

If your dataset is numerical-only, check that numerical-only methods (EM, SoftImpute) are marked as eligible.

## Step 6: Run and Verify

```bash
# Quick test
pytest tests/test_datasets.py -v

# Run a single experiment with your dataset
python scripts/run_experiment.py --config configs/experiments/test_config.yaml --no-wandb
```

## Example: Wine Quality Dataset (Full Reference)

Here's the complete `WineDataset` implementation for reference:

```python
@DATASET_REGISTRY.register("wine")
class WineDataset(BaseDataset):
    name = "wine"

    def __init__(self, data_dir: str = "data/raw", **kwargs: Any) -> None:
        self.data_dir = Path(data_dir)
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        cache_path = self.data_dir / "wine_white.csv"
        if cache_path.exists():
            df = pd.read_csv(cache_path)
        else:
            data = fetch_openml("wine-quality-white", version=1, as_frame=True, parser="auto")
            df = data.frame
            self.data_dir.mkdir(parents=True, exist_ok=True)
            atomic_save_csv(df, cache_path)

        df = df.dropna().reset_index(drop=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna().reset_index(drop=True)
        self._df = df
        return df

    def get_metadata(self) -> dict[str, Any]:
        if self._df is None:
            self.load()
        feature_cols = [c for c in self._df.columns if c != "quality"]
        return {
            "dataset_name": self.name,
            "n_samples": len(self._df),
            "n_features": len(feature_cols),
            "task_type": REGRESSION,
            "target_col": "quality",
            "numerical_features": feature_cols,
            "categorical_features": [],
            "feature_types": {c: NUMERICAL for c in feature_cols},
            "source": "uci/openml",
        }

    def get_splits(self, seed=42, train_ratio=0.8, valid_ratio=0.0):
        if self._df is None:
            self.load()
        return stratified_split(
            self._df, target_col=None, train_ratio=train_ratio,
            valid_ratio=valid_ratio, seed=seed,
        )
```

## Data Source Tips

| Source | Loading Method |
|--------|---------------|
| OpenML | `fetch_openml("dataset-name", version=1, as_frame=True)` |
| UCI CSV | Download with `urllib` or `requests`, save to `data/raw/` |
| Kaggle | Manual download, place CSV in `data/raw/` |
| Local CSV | Read directly from `data/raw/your_file.csv` |
