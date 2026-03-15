"""
Central path and constant definitions for the analysis pipeline.
All scripts import from here to avoid hardcoded paths.
"""
import os
from pathlib import Path

# Project root (two levels up from analysis/scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ANALYSIS_ROOT = PROJECT_ROOT / "analysis"

# Input paths
RAW_RESULTS_DIR = PROJECT_ROOT / "results" / "raw"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Config paths
ANALYSIS_CONFIGS = ANALYSIS_ROOT / "configs"
DATASET_TRACK_CONFIG = ANALYSIS_CONFIGS / "dataset_track_mapping.yaml"
METHOD_ELIGIBILITY_CONFIG = ANALYSIS_CONFIGS / "method_eligibility.yaml"

# Output paths
OUTPUTS_DIR = ANALYSIS_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
STATS_DIR = OUTPUTS_DIR / "stats"

# Aggregated data
AGGREGATED_PARQUET = OUTPUTS_DIR / "aggregated_results.parquet"
AGGREGATED_CSV = OUTPUTS_DIR / "aggregated_results.csv"

# Ensure output directories exist
for d in [TABLES_DIR, FIGURES_DIR, REPORTS_DIR, STATS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Experiment design constants
DATASETS = [
    "adult", "autism", "bank", "breast_cancer", "concrete",
    "letter", "mammographic", "spambase", "titanic", "wine", "yeast"
]
METHODS = [
    "mean", "median", "em", "mice", "knn", "missforest", "softimpute", "xgboost",
    "mida", "miwae", "gain", "tabcsdi", "miracle"
]
NON_DL_METHODS = ["mean", "median", "em", "mice", "knn", "missforest", "softimpute", "xgboost"]
DL_METHODS = ["mida", "miwae", "gain", "tabcsdi", "miracle"]
MECHANISMS = ["MCAR", "MAR", "MNAR"]
PATTERNS = ["general", "univariate"]
RATES = [0.1, 0.2, 0.3, 0.5]
SEEDS = [42, 123, 456, 789, 1024]

EXPECTED_TOTAL = len(DATASETS) * len(METHODS) * len(MECHANISMS) * len(PATTERNS) * len(RATES) * len(SEEDS)

# Primary metrics for analysis
# - Imputation quality: numerical RMSE (universal), categorical accuracy (mixed datasets only)
# - Downstream: varies by task type
PRIMARY_IMPUTATION_METRIC = "imp_num_rmse"
PRIMARY_DOWNSTREAM_CLF_METRIC = "ds_f1_weighted"
PRIMARY_DOWNSTREAM_REG_METRIC = "ds_rmse"

# Figure settings
FIGURE_DPI = 300
FIGURE_FORMAT = ["png", "pdf"]
