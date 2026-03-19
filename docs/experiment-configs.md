# Experiment Configurations

ImputeBench uses a hierarchical YAML configuration system. Experiments are fully defined by configs — no code changes needed.

## Config Hierarchy

```
configs/
├── base.yaml              # Global defaults (tracking, downstream, runtime)
├── datasets/              # Dataset definitions
├── methods/               # Method hyperparameters
├── missingness/           # Missingness mechanism + pattern + rate
├── downstream/            # Downstream task settings
└── experiments/           # Experiment definitions (compose the above)
```

## Single Experiment Config

A single experiment specifies one dataset + one missingness setting + one method:

```yaml
# configs/experiments/single_adult_mean.yaml

project:
  name: missing-value-imputation-benchmark
  seed: 42
  output_dir: results/

dataset:
  name: adult
  split:
    train_ratio: 0.7
    valid_ratio: 0.1

missingness:
  mechanism: MCAR         # MCAR | MAR | MNAR
  rate: 0.1               # 0.1 | 0.2 | 0.3 | 0.5
  seed: 123
  scenario_tag: benchmark_mcar
  exclude_target: true     # Don't inject missingness in target column

method:
  name: mean
  params: {}

tracking:
  use_wandb: false
  output_dir: results/logs

downstream:
  enabled: true
  params:
    n_estimators: 100
    max_depth: 5
```

Run it:
```bash
python scripts/run_experiment.py --config configs/experiments/single_adult_mean.yaml --no-wandb
```

## Batch Experiment Config

A batch config defines lists of datasets, methods, missingness profiles, and seeds. The runner expands them into a **Cartesian product**:

```yaml
# configs/experiments/full_benchmark_nondl.yaml

project:
  name: missing-value-imputation-benchmark
  output_dir: results/

datasets:
  - configs/datasets/wine.yaml
  - configs/datasets/concrete.yaml
  - configs/datasets/spambase.yaml
  - configs/datasets/adult.yaml
  # ... more datasets

methods:
  - configs/methods/mean.yaml
  - configs/methods/knn.yaml
  - configs/methods/missforest.yaml
  - configs/methods/xgboost.yaml
  # ... more methods

missingness_profiles:
  - configs/missingness/mcar_10.yaml
  - configs/missingness/mcar_20.yaml
  - configs/missingness/mar_30.yaml
  - configs/missingness/mnar_50.yaml
  # ... more profiles

seeds: [42, 123, 456, 789, 1024]

tracking:
  use_wandb: false
  output_dir: results/logs

downstream:
  enabled: true
  params:
    n_estimators: 100
```

This expands to: `len(datasets) × len(missingness_profiles) × len(methods) × len(seeds)` individual runs.

Run it:
```bash
python scripts/run_batch.py --config configs/experiments/full_benchmark_nondl.yaml --no-wandb --workers 8
```

## Dataset Config

```yaml
# configs/datasets/wine.yaml
dataset:
  name: wine              # Must match @DATASET_REGISTRY.register("wine")
  dataset_id: PN1         # Optional ID for tracking
  source: uci
  target_col: quality
  task_type: regression   # classification | regression
  split:
    train_ratio: 0.8
    valid_ratio: 0.0
    test_ratio: 0.2
    stratify: false       # true for classification
```

## Method Config

```yaml
# configs/methods/missforest.yaml
method:
  name: missforest        # Must match @METHOD_REGISTRY.register("missforest")
  family: ml
  device: cpu
  params:
    n_estimators: 100
    max_iter: 10
    max_depth: 12
```

For DL methods:
```yaml
# configs/methods/gain.yaml
method:
  name: gain
  family: dl
  device: auto            # auto-selects MPS on Apple Silicon, else CPU
  params:
    batch_size: 128
    epochs: 300
    alpha: 100
```

## Missingness Config

```yaml
# configs/missingness/mcar_30.yaml
missingness:
  mechanism: MCAR         # MCAR | MAR | MNAR
  pattern: general        # general | univariate
  rate: 0.3               # 0.1 | 0.2 | 0.3 | 0.5
  exclude_target: true
```

Naming convention: `{mechanism}_{rate*100}.yaml` — e.g., `mcar_30.yaml`, `mar_10.yaml`, `mnar_50.yaml`.

## Available Configs

### Pre-built Experiment Configs

| Config | Description |
|--------|-------------|
| `single_adult_mean.yaml` | Quick test: one dataset + one method |
| `phase1_smoke_test.yaml` | Fast smoke test across a few settings |
| `phase2_breast_cancer_full.yaml` | Full benchmark on one dataset |
| `phase3_pure_numerical.yaml` | All numerical datasets only |
| `phase4_mixed.yaml` | All mixed-type datasets only |
| `full_benchmark_nondl.yaml` | Full benchmark: non-DL methods |
| `full_benchmark_dl.yaml` | Full benchmark: DL methods |
| `full_benchmark.yaml` | Complete benchmark: all methods + all datasets |
| `benchmark_core.yaml` | Core benchmark subset |

### Config Merging

Experiment configs are merged with `configs/base.yaml` defaults. You can override any base setting in your experiment config.
