"""
Step 1: Data consolidation and validation.
Reads all raw JSON result files, flattens nested structure, validates completeness.
Outputs: aggregated_results.parquet, aggregated_results.csv, schema_report.md, data_validation_report.md
"""
import json
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
from collections import Counter

from paths import (
    RAW_RESULTS_DIR, OUTPUTS_DIR, REPORTS_DIR,
    AGGREGATED_PARQUET, AGGREGATED_CSV,
    DATASETS, METHODS, MECHANISMS, PATTERNS, RATES, SEEDS,
    EXPECTED_TOTAL, NON_DL_METHODS, DL_METHODS
)


def load_all_results():
    """Load all JSON result files and flatten into rows."""
    files = sorted(glob.glob(str(RAW_RESULTS_DIR / "*.json")))
    print(f"Found {len(files)} result files.")

    rows = []
    parse_errors = []
    for fpath in files:
        try:
            with open(fpath) as f:
                d = json.load(f)
        except Exception as e:
            parse_errors.append((fpath, str(e)))
            continue

        row = {
            "dataset": d.get("dataset_name", ""),
            "dataset_id": d.get("dataset_id", ""),
            "method": d.get("method_name", ""),
            "mechanism": d.get("mechanism", ""),
            "pattern": d.get("pattern", ""),
            "ratio": d.get("missing_rate", np.nan),
            "seed": d.get("seed", np.nan),
            "n_samples": d.get("n_samples", np.nan),
            "n_features": d.get("n_features", np.nan),
            "runtime_seconds": d.get("runtime_seconds", np.nan),
        }

        # Flatten imputation quality
        iq = d.get("imputation_quality", {})
        num_iq = iq.get("numerical", {})
        cat_iq = iq.get("categorical", {})
        row["imp_num_rmse"] = num_iq.get("rmse")
        row["imp_num_mae"] = num_iq.get("mae")
        row["imp_cat_accuracy"] = cat_iq.get("accuracy")
        row["imp_cat_macro_f1"] = cat_iq.get("macro_f1")

        # Flatten downstream performance
        dp = d.get("downstream_performance", {})
        row["task_type"] = dp.get("task_type", "")
        row["downstream_model"] = dp.get("model", "")
        metrics = dp.get("metrics", {})
        row["ds_accuracy"] = metrics.get("accuracy")
        row["ds_f1_weighted"] = metrics.get("f1_weighted")
        row["ds_f1_macro"] = metrics.get("f1_macro")
        row["ds_roc_auc"] = metrics.get("roc_auc")
        row["ds_rmse"] = metrics.get("rmse")
        row["ds_mae"] = metrics.get("mae")
        row["ds_r2"] = metrics.get("r2")

        # Flatten distribution preservation
        dist = d.get("distribution_preservation", {})
        row["dist_wasserstein_mean"] = dist.get("wasserstein_distance_mean")
        row["dist_jsd_mean"] = dist.get("jsd_mean")

        row["_source_file"] = Path(fpath).name
        rows.append(row)

    if parse_errors:
        print(f"WARNING: {len(parse_errors)} files failed to parse:")
        for fp, err in parse_errors[:10]:
            print(f"  {fp}: {err}")

    df = pd.DataFrame(rows)
    return df, parse_errors


def add_derived_columns(df):
    """Add method_family and other derived columns."""
    df["method_family"] = df["method"].apply(
        lambda m: "DL" if m in DL_METHODS else "Non-DL"
    )
    df["ratio_pct"] = (df["ratio"] * 100).astype(int)
    df["setting"] = (
        df["mechanism"] + "_" + df["pattern"] + "_" + df["ratio_pct"].astype(str) + "pct"
    )
    # Cast seed to int
    df["seed"] = df["seed"].astype(int)
    return df


def validate_completeness(df):
    """Check for missing/duplicate combinations."""
    key_cols = ["dataset", "method", "mechanism", "pattern", "ratio", "seed"]
    issues = []

    # Expected combinations
    expected = set(product(DATASETS, METHODS, MECHANISMS, PATTERNS, RATES, SEEDS))
    actual = set(df[key_cols].itertuples(index=False, name=None))

    missing = expected - actual
    extra = actual - expected

    # Duplicates
    dupes = df[df.duplicated(subset=key_cols, keep=False)]

    if missing:
        issues.append(f"Missing combinations: {len(missing)}")
    if extra:
        issues.append(f"Extra/unexpected combinations: {len(extra)}")
    if len(dupes) > 0:
        issues.append(f"Duplicate combinations: {len(dupes)}")

    # Metric null checks
    metric_cols = [
        "imp_num_rmse", "imp_num_mae", "imp_cat_accuracy", "imp_cat_macro_f1",
        "ds_accuracy", "ds_f1_weighted", "ds_f1_macro", "ds_roc_auc",
        "ds_rmse", "ds_mae", "ds_r2",
        "dist_wasserstein_mean", "dist_jsd_mean", "runtime_seconds"
    ]
    metric_coverage = {}
    for col in metric_cols:
        n_total = len(df)
        n_notna = df[col].notna().sum()
        metric_coverage[col] = {"total": n_total, "available": int(n_notna),
                                 "missing": int(n_total - n_notna),
                                 "coverage_pct": round(100 * n_notna / n_total, 2)}

    return {
        "n_expected": len(expected),
        "n_actual": len(actual),
        "n_files": len(df),
        "missing_combos": sorted(list(missing))[:20],
        "n_missing": len(missing),
        "extra_combos": sorted(list(extra))[:20],
        "n_extra": len(extra),
        "n_duplicates": len(dupes),
        "duplicate_rows": dupes[key_cols].to_dict("records")[:10] if len(dupes) > 0 else [],
        "metric_coverage": metric_coverage,
        "issues": issues,
    }


def generate_schema_report(df, validation):
    """Generate schema_report.md"""
    lines = [
        "# Schema Report",
        "",
        "## Data Structure",
        f"- Total records: {len(df)}",
        f"- Expected records: {EXPECTED_TOTAL}",
        f"- Unique datasets: {df['dataset'].nunique()} — {sorted(df['dataset'].unique())}",
        f"- Unique methods: {df['method'].nunique()} — {sorted(df['method'].unique())}",
        f"- Unique mechanisms: {df['mechanism'].nunique()} — {sorted(df['mechanism'].unique())}",
        f"- Unique patterns: {df['pattern'].nunique()} — {sorted(df['pattern'].unique())}",
        f"- Unique rates: {sorted(df['ratio'].unique())}",
        f"- Unique seeds: {sorted(df['seed'].unique())}",
        "",
        "## Column Schema",
        "",
        "| Column | Dtype | Non-Null Count | Null Count | Coverage % |",
        "|--------|-------|----------------|------------|------------|",
    ]
    for col in df.columns:
        nn = df[col].notna().sum()
        nu = df[col].isna().sum()
        pct = round(100 * nn / len(df), 1)
        lines.append(f"| {col} | {df[col].dtype} | {nn} | {nu} | {pct}% |")

    lines += [
        "",
        "## Nested JSON Structure (flattened)",
        "```",
        "root",
        "├── dataset_name -> dataset",
        "├── method_name -> method",
        "├── mechanism",
        "├── pattern",
        "├── missing_rate -> ratio",
        "├── seed",
        "├── n_samples, n_features",
        "├── runtime_seconds",
        "├── imputation_quality",
        "│   ├── numerical.rmse -> imp_num_rmse",
        "│   ├── numerical.mae -> imp_num_mae",
        "│   ├── categorical.accuracy -> imp_cat_accuracy",
        "│   └── categorical.macro_f1 -> imp_cat_macro_f1",
        "├── downstream_performance",
        "│   ├── task_type",
        "│   ├── model -> downstream_model",
        "│   └── metrics",
        "│       ├── accuracy -> ds_accuracy",
        "│       ├── f1_weighted -> ds_f1_weighted",
        "│       ├── f1_macro -> ds_f1_macro",
        "│       ├── roc_auc -> ds_roc_auc",
        "│       ├── rmse -> ds_rmse",
        "│       ├── mae -> ds_mae",
        "│       └── r2 -> ds_r2",
        "└── distribution_preservation",
        "    ├── wasserstein_distance_mean -> dist_wasserstein_mean",
        "    └── jsd_mean -> dist_jsd_mean",
        "```",
        "",
        "## Notes",
        "- Categorical imputation quality is null for pure-numerical datasets",
        "- Classification datasets have ds_accuracy/f1/roc_auc; regression datasets have ds_rmse/mae/r2",
        "- Some downstream metrics may be null (e.g., roc_auc for multi-class without probabilities)",
        "- JSD is null for datasets without categorical features",
    ]

    report = "\n".join(lines)
    out_path = REPORTS_DIR / "schema_report.md"
    out_path.write_text(report)
    print(f"Written: {out_path}")


def generate_validation_report(df, validation):
    """Generate data_validation_report.md"""
    v = validation
    lines = [
        "# Data Validation Report",
        "",
        "## Completeness Check",
        f"- Expected unique combinations: {v['n_expected']}",
        f"- Actual unique combinations: {v['n_actual']}",
        f"- Total file records loaded: {v['n_files']}",
        f"- Missing combinations: {v['n_missing']}",
        f"- Extra combinations: {v['n_extra']}",
        f"- Duplicate combinations: {v['n_duplicates']}",
        "",
    ]

    if v["n_missing"] > 0:
        lines.append("### Missing Combinations (first 20)")
        for m in v["missing_combos"]:
            lines.append(f"  - {m}")
        lines.append("")

    if v["n_extra"] > 0:
        lines.append("### Extra Combinations (first 20)")
        for e in v["extra_combos"]:
            lines.append(f"  - {e}")
        lines.append("")

    if v["n_duplicates"] > 0:
        lines.append("### Duplicate Rows (first 10)")
        for d in v["duplicate_rows"]:
            lines.append(f"  - {d}")
        lines.append("")

    if not v["issues"]:
        lines.append("**✓ All validation checks passed. Dataset is complete with no duplicates.**")
    else:
        lines.append("### Issues Found")
        for issue in v["issues"]:
            lines.append(f"- ⚠ {issue}")

    lines += [
        "",
        "## Metric Coverage",
        "",
        "| Metric | Available | Missing | Coverage % |",
        "|--------|-----------|---------|------------|",
    ]
    for metric, info in v["metric_coverage"].items():
        lines.append(
            f"| {metric} | {info['available']} | {info['missing']} | {info['coverage_pct']}% |"
        )

    lines += [
        "",
        "## Per-Method Completeness",
        "",
        "| Method | Records | Expected | Complete |",
        "|--------|---------|----------|----------|",
    ]
    expected_per_method = len(DATASETS) * len(MECHANISMS) * len(PATTERNS) * len(RATES) * len(SEEDS)
    for method in sorted(METHODS):
        n = len(df[df["method"] == method])
        complete = "✓" if n == expected_per_method else f"✗ ({n}/{expected_per_method})"
        lines.append(f"| {method} | {n} | {expected_per_method} | {complete} |")

    lines += [
        "",
        "## Per-Dataset Completeness",
        "",
        "| Dataset | Records | Expected | Complete |",
        "|---------|---------|----------|----------|",
    ]
    expected_per_ds = len(METHODS) * len(MECHANISMS) * len(PATTERNS) * len(RATES) * len(SEEDS)
    for ds in sorted(DATASETS):
        n = len(df[df["dataset"] == ds])
        complete = "✓" if n == expected_per_ds else f"✗ ({n}/{expected_per_ds})"
        lines.append(f"| {ds} | {n} | {expected_per_ds} | {complete} |")

    report = "\n".join(lines)
    out_path = REPORTS_DIR / "data_validation_report.md"
    out_path.write_text(report)
    print(f"Written: {out_path}")


def main():
    print("=" * 60)
    print("Step 1: Data Consolidation & Validation")
    print("=" * 60)

    # Load
    df, parse_errors = load_all_results()
    print(f"Loaded {len(df)} records.")

    # Standardize & derive
    df = add_derived_columns(df)

    # Validate
    validation = validate_completeness(df)
    print(f"Validation: {len(validation['issues'])} issues found.")
    for issue in validation["issues"]:
        print(f"  - {issue}")
    if not validation["issues"]:
        print("  All checks passed!")

    # Save
    df.to_csv(AGGREGATED_CSV, index=False)
    print(f"Saved: {AGGREGATED_CSV}")
    try:
        df.to_parquet(AGGREGATED_PARQUET, index=False)
        print(f"Saved: {AGGREGATED_PARQUET}")
    except ImportError:
        print("Skipping parquet (pyarrow not installed)")

    # Reports
    generate_schema_report(df, validation)
    generate_validation_report(df, validation)

    print(f"\nTotal records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("Step 1 complete.\n")
    return df


if __name__ == "__main__":
    main()
