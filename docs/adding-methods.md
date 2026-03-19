# Adding a New Imputation Method

This guide walks you through adding a new imputation method to ImputeBench. The platform uses a **registry pattern** — you implement a class, decorate it, and the pipeline picks it up automatically.

## Step 1: Implement the Imputer Class

Create a new file in the appropriate subdirectory:

| Method type | Directory |
|-------------|-----------|
| Statistical | `src/methods/statistical/` |
| Machine Learning | `src/methods/ml/` |
| Deep Learning | `src/methods/dl/` |

Your class must extend `BaseImputer` and implement two methods:

```python
# src/methods/ml/my_method.py

from typing import Any
import pandas as pd
from src.methods.base import BaseImputer
from src.registry import METHOD_REGISTRY


@METHOD_REGISTRY.register("my_method")
class MyMethodImputer(BaseImputer):
    """One-line description of your method."""

    name = "my_method"
    family = "ml"  # Options: "statistical", "ml", "dl"

    def __init__(self, n_neighbors: int = 5, **kwargs: Any) -> None:
        super().__init__(n_neighbors=n_neighbors, **kwargs)
        self.n_neighbors = n_neighbors

    def fit(self, X_train: pd.DataFrame, metadata: dict[str, Any] | None = None) -> None:
        """Learn imputation parameters from training data (which may contain NaN)."""
        # metadata contains: numerical_features, categorical_features, feature_types, etc.
        self._metadata = metadata or {}
        # ... fit your model here ...

    def impute(self, X_missing: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN values in X_missing and return a complete DataFrame."""
        # ... apply your model ...
        return X_imputed
```

### The `BaseImputer` API

| Method | Required | Description |
|--------|----------|-------------|
| `__init__(**kwargs)` | Yes | Store hyperparameters. Call `super().__init__(**kwargs)`. |
| `fit(X_train, metadata)` | Yes | Learn from training data. `metadata` provides column types. |
| `impute(X_missing)` | Yes | Return a complete DataFrame (no NaN). |
| `fit_impute(X_train, X_eval, metadata)` | Optional | Override for methods needing joint fit+impute (common in DL). |
| `set_seed(seed)` | Inherited | Called by the runner to set `self._seed` before `fit()`. |
| `get_params()` | Inherited | Returns `{name, family, seed, **kwargs}` for logging. |

### The `metadata` Dictionary

The `metadata` dict passed to `fit()` contains:

```python
{
    "dataset_name": "adult",
    "numerical_features": ["age", "hours_per_week", ...],
    "categorical_features": ["workclass", "education", ...],
    "feature_types": {"age": "numerical", "workclass": "categorical", ...},
    "task_type": "classification",
    "target_col": "income",
}
```

Use `metadata["numerical_features"]` and `metadata["categorical_features"]` to handle different column types appropriately.

## Step 2: Create a YAML Config

```yaml
# configs/methods/my_method.yaml

method:
  name: my_method
  family: ml
  device: cpu          # or "auto" for DL methods
  params:
    n_neighbors: 5     # matches __init__ kwargs
```

## Step 3: Register the Import

Add your module to `src/factory.py` so it's imported at startup:

```python
# In src/factory.py, add with the other ML method imports:
import src.methods.ml.my_method  # noqa: F401
```

This import triggers the `@METHOD_REGISTRY.register()` decorator.

## Step 4: Add to an Experiment Config

Reference your method in a batch experiment config:

```yaml
# In configs/experiments/my_benchmark.yaml
methods:
  - configs/methods/my_method.yaml
  - configs/methods/mean.yaml       # include baselines for comparison
  - configs/methods/missforest.yaml
```

Or test it standalone:

```yaml
# configs/experiments/test_my_method.yaml
project:
  name: missing-value-imputation-benchmark
  seed: 42
  output_dir: results/

dataset:
  name: wine
  split:
    train_ratio: 0.8

missingness:
  mechanism: MCAR
  rate: 0.2
  seed: 123
  exclude_target: true

method:
  name: my_method
  params:
    n_neighbors: 5

tracking:
  use_wandb: false
  output_dir: results/logs

downstream:
  enabled: true
  params:
    n_estimators: 100
```

## Step 5: Run and Verify

```bash
# Quick test
python scripts/run_experiment.py --config configs/experiments/test_my_method.yaml --no-wandb

# Run tests
pytest tests/test_methods.py -v
```

## Tips

- **Numerical-only methods**: If your method only handles numerical data, process `metadata["numerical_features"]` columns and use mode imputation as a fallback for categorical columns. See `src/methods/statistical/mean.py` for an example.
- **DL methods**: Set `family = "dl"` and handle device selection. Use `self.params.get("device", "cpu")`. See `src/methods/dl/gain.py` for MPS/CPU handling patterns.
- **Reproducibility**: Use `self._seed` (set by the runner) for all random operations.
- **Mixed-type data**: Categorical features are label-encoded before reaching your imputer. You receive numerical values but should respect the column type metadata for any special handling.

## Complete Example: A Minimal Statistical Imputer

Here's the full `MeanImputer` for reference — one of the simplest implementations:

```python
@METHOD_REGISTRY.register("mean")
class MeanImputer(BaseImputer):
    name = "mean"
    family = "statistical"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._fill_values: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}

    def fit(self, X_train: pd.DataFrame, metadata: dict[str, Any] | None = None) -> None:
        self._metadata = metadata or {}
        numerical_cols = self._metadata.get("numerical_features", [])
        categorical_cols = self._metadata.get("categorical_features", [])

        for col in X_train.columns:
            if col in numerical_cols:
                self._fill_values[col] = X_train[col].mean()
            elif col in categorical_cols:
                mode = X_train[col].mode()
                self._fill_values[col] = mode.iloc[0] if len(mode) > 0 else 0
            else:
                try:
                    self._fill_values[col] = X_train[col].mean()
                except TypeError:
                    mode = X_train[col].mode()
                    self._fill_values[col] = mode.iloc[0] if len(mode) > 0 else 0

    def impute(self, X_missing: pd.DataFrame) -> pd.DataFrame:
        return X_missing.fillna(self._fill_values)
```
