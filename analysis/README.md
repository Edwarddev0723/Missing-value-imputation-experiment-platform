# Analysis Pipeline — Missing Value Imputation Benchmark

This directory contains a complete, reproducible analysis pipeline for the missing value imputation benchmark (17,160 experiments).

## Quick Start

```bash
# From the project root:
.venv/bin/python analysis/scripts/run_all.py
```

This runs all 8 steps and generates tables, figures, reports, and statistical tests.

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 1 | `step1_consolidate.py` | Load 17,160 JSON results, flatten, validate completeness |
| 2 | `step2_audit.py` | Metric inventory, per-method/dataset/setting summary |
| 3-4 | `step3_analysis.py` | Stratified analysis, track analysis, degradation |
| 5 | `step5_stats.py` | Friedman test, pairwise Wilcoxon, baseline gap analysis |
| 6 | `step6_visualize.py` | 10 publication-ready figures (PNG + PDF) |
| 7 | `step7_tables.py` | 8 paper-ready tables (CSV + LaTeX + Markdown) |
| 8 | `step8_conclusions.py` | Auto-generated executive summary, paper draft, key findings |

## Directory Structure

```
analysis/
├── scripts/
│   ├── run_all.py              # Master runner
│   ├── paths.py                # Centralized path/constant management
│   ├── step1_consolidate.py    # Data consolidation & validation
│   ├── step2_audit.py          # Exploratory audit
│   ├── step3_analysis.py       # Stratified & track analysis
│   ├── step5_stats.py          # Statistical testing
│   ├── step6_visualize.py      # Visualization
│   ├── step7_tables.py         # Paper-ready tables
│   └── step8_conclusions.py    # Text conclusions
├── configs/
│   ├── dataset_track_mapping.yaml   # Pure numerical vs mixed-type classification
│   └── method_eligibility.yaml      # Method eligibility per track
├── outputs/
│   ├── aggregated_results.csv       # Full long-format dataset
│   ├── aggregated_results.parquet   # Same in parquet format
│   ├── tables/                      # CSV, LaTeX, Markdown tables
│   ├── figures/                     # PNG + PDF figures
│   ├── reports/                     # Analysis reports
│   └── stats/                       # Statistical test results
└── README.md
```

## Dependencies

- Python 3.10+
- pandas, numpy, scipy, matplotlib, seaborn, pyyaml, tabulate, pyarrow

## Experiment Design

- **11 datasets**: adult, autism, bank, breast_cancer, concrete, letter, mammographic, spambase, titanic, wine, yeast
- **13 methods**: mean, median, em, mice, knn, missforest, softimpute, xgboost (Non-DL); mida, miwae, gain, tabcsdi, miracle (DL)
- **24 missing settings**: 3 mechanisms (MCAR/MAR/MNAR) × 2 patterns (general/univariate) × 4 ratios (10/20/30/50%)
- **5 random seeds**: 42, 123, 456, 789, 1024
- **Total**: 11 × 13 × 24 × 5 = 17,160 runs

## Key Design Choices

1. **Within-dataset ranking**: Methods are ranked 1–13 within each (dataset, mechanism, pattern, ratio, seed) condition to avoid unfair cross-dataset aggregation.
2. **Normalized scores**: Min-max normalization within each condition (0–1, higher=better).
3. **Track analysis**: Datasets divided into pure_numerical (7) and mixed_type (4) tracks.
4. **Statistical rigor**: Friedman test + pairwise Wilcoxon signed-rank with Holm correction.

## Customization

- Edit `configs/dataset_track_mapping.yaml` to modify dataset classification
- Edit `configs/method_eligibility.yaml` to control method eligibility per track
- Edit `scripts/paths.py` to change output directories or metric choices
