# Getting Started

## Prerequisites

- Python 3.10 or higher
- macOS, Linux, or Windows (macOS with Apple Silicon fully supported)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ImputeBench.git
cd ImputeBench

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install with all dependencies (including DL and ML extras)
pip install -e ".[all]"

# Or install only core + statistical methods
pip install -e "."

# Or install specific extras
pip install -e ".[dl]"    # Adds PyTorch for DL methods
pip install -e ".[ml]"    # Adds XGBoost
pip install -e ".[dev]"   # Adds pytest, ruff, pre-commit
```

## Run Your First Experiment

```bash
python scripts/run_experiment.py \
  --config configs/experiments/single_adult_mean.yaml \
  --no-wandb
```

This runs mean imputation on the Adult dataset with MCAR 10% missingness. Results are saved to `results/logs/`.

## Run a Batch Benchmark

A batch experiment expands a set of datasets × missingness configs × methods × seeds:

```bash
# Non-DL methods (use more workers for parallelism)
python scripts/run_batch.py \
  --config configs/experiments/full_benchmark_nondl.yaml \
  --no-wandb --workers 8

# DL methods (fewer workers to avoid OOM)
python scripts/run_batch.py \
  --config configs/experiments/full_benchmark_dl.yaml \
  --no-wandb --workers 2
```

## Generate Analysis

After running experiments, produce tables, figures, and statistical tests:

```bash
python analysis/scripts/run_all.py
```

Outputs go to `analysis/outputs/`:
- `tables/` — 8 LaTeX + CSV tables
- `figures/` — 10 publication-ready figures (PDF + PNG)
- `reports/` — executive summary, key findings, paper results draft
- `stats/` — statistical test results (Friedman, Wilcoxon)

## Optional: Weights & Biases Tracking

```bash
# Set your API key
echo "WANDB_API_KEY=your_key_here" > .env

# Run with W&B tracking (omit --no-wandb)
python scripts/run_experiment.py \
  --config configs/experiments/single_adult_mean.yaml
```

## Run Tests

```bash
pytest tests/ -v
```

## What's Next?

- [Experiment Configs](experiment-configs.md) — learn the YAML configuration system
- [Adding Methods](adding-methods.md) — add your own imputation method
- [Adding Datasets](adding-datasets.md) — add a new dataset
- [Analysis Pipeline](analysis-pipeline.md) — understand the automated analysis
