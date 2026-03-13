# Missing Value Imputation Benchmark Platform

A reproducible, extensible, config-driven platform for systematically comparing missing value imputation methods across datasets, missingness mechanisms, and evaluation metrics.

## Features

- **Unified API**: All imputation methods share `fit()` / `impute()` / `fit_impute()`.
- **Config-driven**: Experiments fully defined via YAML configs — datasets, methods, missingness, evaluation.
- **Registry pattern**: Register new datasets/methods/generators without modifying pipeline code.
- **W&B integration**: Automatic experiment tracking, config logging, metric logging, artifact management.
- **Reproducibility**: Shared splits, shared masks, fixed seeds across all methods.
- **Batch execution**: Cross-product experiment matrix (dataset × missingness × method × seed).

## Quick Start

### 1. Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Copy and fill environment variables
cp .env.example .env
```

### 2. Run a single experiment

```bash
python scripts/run_experiment.py \
  --config configs/experiments/single_adult_mean.yaml \
  --no-wandb
```

### 3. Run a batch benchmark

```bash
python scripts/run_batch.py \
  --config configs/experiments/benchmark_core.yaml \
  --no-wandb
```

### 4. Run with W&B tracking

```bash
# Set WANDB_API_KEY in .env, then:
python scripts/run_experiment.py \
  --config configs/experiments/single_adult_mean.yaml
```

### 5. Run tests

```bash
pytest tests/ -v
```

## Project Structure

```
├── configs/                 # YAML experiment configurations
│   ├── base.yaml            # Default settings (tracking, downstream, runtime)
│   ├── datasets/            # Dataset configs (adult, wine, heart)
│   ├── methods/             # Method configs (mean, knn, missforest, softimpute)
│   ├── missingness/         # Missingness configs (mcar_10, mar_30)
│   ├── downstream/          # Downstream task configs
│   └── experiments/         # Full experiment / batch configs
├── src/                     # Core source code
│   ├── config.py            # Config loading, merging, validation
│   ├── registry.py          # Component registry (dataset, method, missingness, tracker)
│   ├── factory.py           # ComponentFactory for creating components from config
│   ├── constants.py         # Platform-wide constants
│   ├── schemas/             # Result schema (ExperimentResult dataclass)
│   ├── datasets/            # Dataset adapters (BaseDataset, Adult, Wine, Heart)
│   ├── preprocessing/       # Canonical preprocessing (encoding, scaling)
│   ├── missingness/         # Missingness generators (MCAR, MAR)
│   ├── methods/             # Imputation methods
│   │   ├── base.py          # BaseImputer abstract class
│   │   ├── statistical/     # Mean, KNN, softImpute
│   │   ├── ml/              # MissForest
│   │   └── dl/              # (placeholder for GAIN, MIWAE, CSDI)
│   ├── evaluation/          # Imputation metrics + downstream metrics
│   ├── tracking/            # W&B tracker + local tracker
│   ├── orchestration/       # ExperimentRunner, BatchExperimentRunner
│   ├── reporting/           # Summary tables (CSV, future LaTeX)
│   └── utils/               # Seed, device, timing, I/O, logging
├── scripts/                 # Entry points
│   ├── run_experiment.py    # Single experiment runner
│   └── run_batch.py         # Batch experiment runner
├── tests/                   # Test suite (40 tests)
├── results/                 # Output directory (gitignored)
├── data/                    # Data directory (gitignored, auto-downloaded)
└── pyproject.toml           # Project dependencies
```

## Config System

Configs are **hierarchical and composable**:

- `configs/base.yaml` — defaults (tracking, downstream, runtime)
- `configs/datasets/*.yaml` — dataset definitions
- `configs/methods/*.yaml` — method hyperparameters
- `configs/missingness/*.yaml` — missingness mechanism settings
- `configs/experiments/*.yaml` — full experiments or batch definitions

A batch config like `benchmark_core.yaml` expands into `datasets × missingness_profiles × methods × seeds`.

## Currently Implemented

### Datasets
| Name  | Type           | Source   | Task           |
|-------|---------------|----------|----------------|
| adult | Mixed (num+cat)| UCI/OpenML | Classification |
| wine  | Numerical     | UCI/OpenML | Regression     |
| heart | Numerical     | UCI/OpenML | Classification |

### Imputation Methods
| Name       | Family      | Description                        |
|------------|-------------|------------------------------------|
| mean       | statistical | Mean (numerical) / Mode (categorical) |
| knn        | statistical | K-Nearest Neighbors (sklearn)      |
| softimpute | statistical | Iterative SVD soft-thresholding    |
| missforest | ml          | Iterative Random Forest (sklearn)  |

### Missingness Generators
| Name | Mechanism | Strategy                    |
|------|----------|-----------------------------|
| MCAR | MCAR     | Uniform random masking       |
| MAR  | MAR      | Driver-value dependent       |

### Evaluation Metrics
- **Numerical**: RMSE, MAE, NRMSE
- **Categorical**: Accuracy
- **Downstream**: Accuracy, F1, ROC-AUC (classification); RMSE, MAE, R² (regression)

## Extending the Platform

### Add a new imputation method

```python
from src.methods.base import BaseImputer
from src.registry import METHOD_REGISTRY

@METHOD_REGISTRY.register("my_method")
class MyImputer(BaseImputer):
    name = "my_method"
    family = "ml"

    def fit(self, X_train, metadata=None):
        # Learn from training data
        pass

    def impute(self, X_missing):
        # Return completed DataFrame
        pass
```

Then add `configs/methods/my_method.yaml` and use it in experiments.

### Add a new dataset

```python
from src.datasets.base import BaseDataset
from src.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register("my_dataset")
class MyDataset(BaseDataset):
    name = "my_dataset"

    def load(self): ...
    def get_metadata(self): ...
    def get_splits(self, seed=42, train_ratio=0.7, valid_ratio=0.1): ...
```

Register the import in `src/factory.py`.

## TODO / Roadmap

### Phase 2 — Extended Methods
- [ ] MICE (Multiple Imputation by Chained Equations)
- [ ] XGBoost-based iterative imputation
- [ ] GAIN (Generative Adversarial Imputation Nets)
- [ ] MIWAE (Missing data IWAE)
- [ ] CSDI (Conditional Score-based Diffusion)

### Phase 2 — Extended Missingness
- [ ] MNAR generator
- [ ] Hybrid/Scenario-based missingness (survey-like, medical-like)
- [ ] Robustness protocol (train/test distribution shift)

### Phase 2 — Reporting & Analysis
- [ ] LaTeX table export
- [ ] Plots (missing rate vs. performance, method comparison, heatmaps)
- [ ] Recommendation matrix auto-generation
- [ ] Case study output (per-feature error analysis)
- [ ] Critical difference diagrams

### Phase 2 — Engineering
- [ ] W&B sweep support
- [ ] DVC data versioning
- [ ] Parallel batch execution
- [ ] Checkpoint/resume for long experiments

## Hardware

Designed for local-first development on **MacBook Pro M4 Pro / 48GB RAM**:
- CPU is the default device for all methods
- MPS support is preserved for PyTorch-based DL methods
- No CUDA-only dependencies

## License

MIT

