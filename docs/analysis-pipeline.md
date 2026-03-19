# Analysis Pipeline

ImputeBench includes a 7-step automated analysis pipeline that transforms raw experiment results into publication-ready outputs.

## Quick Start

```bash
python analysis/scripts/run_all.py
```

This runs all steps sequentially. Outputs go to `analysis/outputs/`.

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 1 | `step1_consolidate.py` | Load raw JSON results from `results/runs/`, flatten into a single DataFrame, validate completeness |
| 2 | `step2_audit.py` | Exploratory audit — metric availability, per-method stats, data quality checks |
| 3–4 | `step3_analysis.py` | Stratified analysis with within-dataset normalized ranks; track-level analysis (pure numerical vs mixed) |
| 5 | `step5_stats.py` | Statistical significance testing — Friedman test, pairwise Wilcoxon signed-rank with Holm correction |
| 6 | `step6_visualize.py` | Generate 10 figures (heatmaps, boxplots, scatter, violin, degradation plots) |
| 7 | `step7_tables.py` | Create 8 paper-ready tables in LaTeX, CSV, and Markdown formats |
| 8 | `step8_conclusions.py` | Auto-generate key findings, executive summary, and paper results section draft |

## Output Structure

```
analysis/outputs/
├── aggregated_results.csv      # Consolidated results (base data for all analyses)
├── aggregated_results.parquet  # Same data in Parquet format
├── tables/                     # 8 tables × 3 formats (CSV, LaTeX, Markdown)
│   ├── table1_overall_summary.*
│   ├── table2_dataset_best_method.*
│   ├── table3_mechanism_comparison.*
│   ├── table4_ratio_comparison.*
│   ├── table5_dl_vs_nondl.*
│   ├── table6_top3_methods.*
│   ├── table7_stability_ranking.*
│   ├── table8_runtime_efficiency.*
│   └── ... (additional analysis CSVs)
├── figures/                    # 10 figures (PDF + PNG)
│   ├── fig1_heatmap_method_dataset_rank.*
│   ├── fig2_line_ratio_by_mechanism.*
│   ├── fig3_boxplot_rank_distribution.*
│   ├── fig4_bar_dl_vs_nondl.*
│   ├── fig6_scatter_perf_vs_runtime.*
│   ├── fig7_stability_rmse.*
│   ├── fig8_degradation_plot.*
│   ├── fig9_mechanism_comparison.*
│   ├── fig10_violin_rank.*
│   └── fig11_scatter_imp_vs_downstream.*
├── reports/                    # Auto-generated prose
│   ├── executive_summary.md
│   ├── key_findings_bullets.md
│   ├── paper_results_section_draft.md
│   ├── statistical_testing_report.md
│   └── ... (audit reports)
└── stats/                      # Statistical test outputs
    ├── pairwise_wilcoxon_imputation.csv
    ├── pairwise_wilcoxon_downstream.csv
    └── ...
```

## Tables Reference

| Table | Content | Use in Paper |
|-------|---------|--------------|
| **Table 1** | Overall method ranking (avg rank, RMSE, runtime) | Main results table |
| **Table 2** | Best method per dataset | Dataset-specific analysis |
| **Table 3** | Rankings by missingness mechanism (MCAR/MAR/MNAR) | Mechanism sensitivity |
| **Table 4** | Rankings by missing rate (10%–50%) | Rate sensitivity |
| **Table 5** | DL vs Non-DL family comparison | Key finding support |
| **Table 6** | Top-3 methods per experimental condition | Detailed recommendations |
| **Table 7** | Stability ranking (rank std across conditions) | Robustness analysis |
| **Table 8** | Runtime efficiency (mean, median, min, max) | Practical considerations |

## Figures Reference

| Figure | Type | Shows |
|--------|------|-------|
| **Fig 1** | Heatmap | Method × Dataset average rank |
| **Fig 2** | Line plot | Performance trend across missing rates, by mechanism |
| **Fig 3** | Boxplot | Rank distribution per method |
| **Fig 4** | Bar chart | DL vs Non-DL aggregate comparison |
| **Fig 6** | Scatter | Performance vs Runtime trade-off |
| **Fig 7** | Plot | Stability (RMSE std across conditions) |
| **Fig 8** | Plot | Degradation from 10% → 50% missing rate |
| **Fig 9** | Grouped bars | Mechanism comparison per method |
| **Fig 10** | Violin | Full rank distribution per method |
| **Fig 11** | Scatter | Imputation quality vs Downstream performance correlation |

## Statistical Tests

The pipeline performs:

1. **Friedman test** — tests whether there are statistically significant differences among the 13 methods.
2. **Pairwise Wilcoxon signed-rank tests** — all ${13 \choose 2} = 78$ pairs, with Holm correction for multiple comparisons.

Results are reported for both imputation quality metrics and downstream task metrics.

## Configuration

### `analysis/configs/dataset_track_mapping.yaml`

Maps datasets to analysis tracks (pure numerical vs mixed type) for track-specific analyses.

### `analysis/configs/method_eligibility.yaml`

Defines which methods are eligible for which dataset types (e.g., EM and SoftImpute are numerical-only).

## Running Individual Steps

```bash
cd analysis/scripts

# Run just Step 1
python step1_consolidate.py

# Run just the visualization step
python step6_visualize.py

# Run just the table generation
python step7_tables.py
```

Each step reads from the output of previous steps. Run `step1_consolidate.py` first if starting fresh.
