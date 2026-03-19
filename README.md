# ImputeBench

> A comprehensive, reproducible benchmark for missing value imputation — 13 methods · 12 datasets · 3 mechanisms · 17,160 experimental runs — with a reusable, config-driven experiment platform.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-40%20passed-brightgreen.svg)]()

---

## Leaderboard

| Rank | Method | Type | Avg Rank ↓ | Stability (std) | RMSE Mean | Runtime |
|:----:|--------|:----:|:----------:|:---------------:|:---------:|--------:|
| 🥇 | **MissForest** | ML | **2.06** | 1.79 | 0.161 | 79.2 s |
| 🥈 | **XGBoost** | ML | **2.71** | 1.75 | 0.170 | 71.6 s |
| 🥉 | **KNN** | ML | **4.49** | 2.64 | 0.196 | 10.4 s |
| 4 | EM | Stat | 5.05 | 2.94 | 0.219 | 2.1 s |
| 5 | MICE | Stat | 5.52 | 2.82 | 0.209 | 3.5 s |
| 6 | Mean | Stat | 7.23 | 2.19 | 0.273 | 0.002 s |
| 7 | MIDA | DL | 7.66 | 2.74 | 0.298 | 8.1 s |
| 8 | MIWAE | DL | 7.69 | 2.91 | 0.298 | 17.4 s |
| 9 | SoftImpute | Stat | 8.41 | 2.67 | 0.312 | 1.3 s |
| 10 | GAIN | DL | 8.64 | 2.54 | 0.333 | 1.9 s |
| 11 | Median | Stat | 9.22 | 2.56 | 0.304 | 0.003 s |
| 12 | MIRACLE | DL | 10.66 | 2.60 | 0.381 | 21.0 s |
| 13 | TabCSDI | DL | 11.66 | 1.93 | 0.849 | 9.5 s |

> Ranked by average within-condition rank across all 17,160 runs (lower = better). Full results in [`benchmark/`](benchmark/).

## Key Findings

1. **MissForest wins overall** — best imputation rank on 9 out of 11 datasets, with strong stability.
2. **Non-DL > DL across the board** — Non-DL avg rank 5.59 vs DL 9.26, while being comparably fast.
3. **No universal winner** — best method varies by dataset, underscoring the need for dataset-specific selection.
4. **MNAR is the hardest mechanism** — all methods show degraded performance under Missing Not At Random.
5. **Simple baselines remain competitive** — mean/median imputation is viable at low missing rates, and orders of magnitude faster.
6. **XGBoost is most stable** — lowest rank standard deviation (1.75) across all experimental conditions.

> All comparisons backed by Friedman + pairwise Wilcoxon signed-rank tests with Holm correction (76/78 pairs significant at α = 0.05).

## What This Repository Provides

| For researchers wanting results | For researchers wanting to experiment |
|:---:|:---:|
| Pre-computed benchmark results across 17,160 runs | Config-driven platform: define experiments in YAML, no code changes |
| Publication-ready LaTeX tables and figures | Unified `fit()` / `impute()` API for all 13 methods |
| Statistical significance tests (Friedman, Wilcoxon) | Easy to add new methods, datasets, or missingness mechanisms |
| → See [`benchmark/`](benchmark/) | → See [Getting Started](docs/getting-started.md) |

## Quick Start

```bash
# 1. Install
git clone https://github.com/yourusername/ImputeBench.git
cd ImputeBench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# 2. Run a single experiment
python scripts/run_experiment.py \
  --config configs/experiments/single_adult_mean.yaml --no-wandb

# 3. Run the full benchmark
python scripts/run_batch.py \
  --config configs/experiments/full_benchmark_nondl.yaml --no-wandb --workers 8

# 4. Generate analysis (tables, figures, stats)
python analysis/scripts/run_all.py

# 5. Run tests
pytest tests/ -v
```

> See [docs/getting-started.md](docs/getting-started.md) for detailed setup instructions.

## Benchmark Design

### 13 Imputation Methods

| Category | Methods |
|----------|---------|
| **Statistical** (6) | Mean, Median, EM, MICE, KNN, SoftImpute |
| **Machine Learning** (2) | MissForest, XGBoost |
| **Deep Learning** (5) | MIDA, MIWAE, GAIN, TabCSDI, MIRACLE |

### 12 Datasets (UCI / OpenML)

| Track | Datasets |
|-------|----------|
| **Pure Numerical** (6) | Wine Quality, Concrete, Spambase, Yeast, Breast Cancer, Letter Recognition |
| **Mixed Type** (6) | Adult, Bank Marketing, Heart, Titanic, Mammographic, Autism |

### Missingness Configurations

- **3 mechanisms**: MCAR, MAR, MNAR
- **2 patterns**: Univariate, General
- **4 missing rates**: 10%, 20%, 30%, 50%
- **5 random seeds** per setting → mean ± std reported

### Evaluation

- **Imputation quality**: RMSE, MAE (numerical); Accuracy, Macro-F1 (categorical)
- **Downstream task**: RandomForest classifier/regressor on imputed data
- **Statistical tests**: Friedman test + pairwise Wilcoxon with Holm correction

## Project Structure

```
├── benchmark/               # 🏆 Pre-computed results, figures, and tables
├── docs/                    # 📖 User guides and documentation
│   ├── getting-started.md
│   ├── adding-methods.md
│   ├── adding-datasets.md
│   ├── experiment-configs.md
│   ├── analysis-pipeline.md
│   └── design/              # Internal architecture docs
├── configs/                 # YAML experiment configurations
│   ├── datasets/            # 12 dataset configs
│   ├── methods/             # 13 method configs
│   ├── missingness/         # Missingness configs (mechanism × pattern × rate)
│   └── experiments/         # Batch & single experiment definitions
├── src/                     # Core platform code
│   ├── methods/             # Imputation methods (statistical / ML / DL)
│   ├── datasets/            # Dataset adapters (auto-download + caching)
│   ├── missingness/         # MCAR, MAR, MNAR generators
│   ├── evaluation/          # Metrics (imputation + downstream)
│   ├── orchestration/       # Experiment & batch runners
│   └── ...                  # Config, registry, factory, tracking, utils
├── analysis/                # 7-step automated analysis pipeline
│   ├── scripts/             # Pipeline scripts (consolidate → conclusions)
│   └── outputs/             # Generated tables, figures, reports, stats
├── scripts/                 # CLI entry points
├── tests/                   # Test suite (40 tests)
└── paper/                   # Survey paper materials
```

## Extending the Platform

ImputeBench is designed to be extended. Add your own methods, datasets, or missingness mechanisms without modifying core pipeline code:

- **[Adding Methods](docs/adding-methods.md)** — Implement `fit()` + `impute()`, register with `@METHOD_REGISTRY.register()`.
- **[Adding Datasets](docs/adding-datasets.md)** — Implement `load()` + `get_metadata()` + `get_splits()`, register with `@DATASET_REGISTRY.register()`.
- **[Experiment Configs](docs/experiment-configs.md)** — Define experiments in YAML; batch configs expand as datasets × missingness × methods × seeds.
- **[Analysis Pipeline](docs/analysis-pipeline.md)** — One command to generate all tables, figures, and statistical tests.

## Citation

If you use ImputeBench in your research, please cite:

```bibtex
@software{imputebench2025,
  title     = {ImputeBench: A Comprehensive Benchmark for Missing Value Imputation},
  author    = {Huang, Edward},
  year      = {2025},
  url       = {https://github.com/yourusername/ImputeBench},
  license   = {MIT}
}
```

## License

MIT

