"""
Step 2: Exploratory audit.
Summarizes metric availability, per-method/dataset/setting statistics.
Outputs: exploratory_audit.md
"""
import pandas as pd
import numpy as np
from pathlib import Path

from paths import (
    AGGREGATED_PARQUET, AGGREGATED_CSV, REPORTS_DIR, TABLES_DIR,
    DATASETS, METHODS, NON_DL_METHODS, DL_METHODS,
    MECHANISMS, PATTERNS, RATES
)


def load_data():
    return pd.read_csv(AGGREGATED_CSV)


def metric_audit(df):
    """Identify all metric columns and their coverage."""
    metric_cols = [c for c in df.columns if c.startswith(("imp_", "ds_", "dist_", "runtime"))]
    rows = []
    for col in metric_cols:
        n_total = len(df)
        n_notna = df[col].notna().sum()
        mean_val = df[col].mean() if n_notna > 0 else np.nan
        std_val = df[col].std() if n_notna > 0 else np.nan
        rows.append({
            "metric": col,
            "total": n_total,
            "available": int(n_notna),
            "missing": int(n_total - n_notna),
            "coverage_pct": round(100 * n_notna / n_total, 2),
            "mean": round(mean_val, 6) if not np.isnan(mean_val) else None,
            "std": round(std_val, 6) if not np.isnan(std_val) else None,
            "min": round(df[col].min(), 6) if n_notna > 0 else None,
            "max": round(df[col].max(), 6) if n_notna > 0 else None,
        })
    return pd.DataFrame(rows)


def per_method_summary(df):
    """Per-method basic statistics for key metrics."""
    key_metrics = ["imp_num_rmse", "imp_num_mae", "ds_f1_weighted", "ds_rmse", "runtime_seconds"]
    results = []
    for method in sorted(METHODS):
        mdf = df[df["method"] == method]
        row = {"method": method, "family": "DL" if method in DL_METHODS else "Non-DL", "n_records": len(mdf)}
        for metric in key_metrics:
            vals = mdf[metric].dropna()
            if len(vals) > 0:
                row[f"{metric}_mean"] = round(vals.mean(), 6)
                row[f"{metric}_std"] = round(vals.std(), 6)
                row[f"{metric}_n"] = len(vals)
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
                row[f"{metric}_n"] = 0
        results.append(row)
    return pd.DataFrame(results)


def per_dataset_summary(df):
    """Per-dataset basic statistics."""
    key_metrics = ["imp_num_rmse", "ds_f1_weighted", "ds_rmse", "runtime_seconds"]
    results = []
    for ds in sorted(DATASETS):
        ddf = df[df["dataset"] == ds]
        row = {
            "dataset": ds,
            "task_type": ddf["task_type"].mode().iloc[0] if len(ddf) > 0 else "",
            "n_records": len(ddf),
            "n_samples": ddf["n_samples"].iloc[0] if len(ddf) > 0 else None,
            "n_features": ddf["n_features"].iloc[0] if len(ddf) > 0 else None,
        }
        for metric in key_metrics:
            vals = ddf[metric].dropna()
            if len(vals) > 0:
                row[f"{metric}_mean"] = round(vals.mean(), 6)
                row[f"{metric}_std"] = round(vals.std(), 6)
        results.append(row)
    return pd.DataFrame(results)


def per_setting_summary(df):
    """Per missing-setting statistics."""
    results = []
    for mech in MECHANISMS:
        for pat in PATTERNS:
            for rate in RATES:
                mask = (df["mechanism"] == mech) & (df["pattern"] == pat) & (df["ratio"] == rate)
                sdf = df[mask]
                results.append({
                    "mechanism": mech,
                    "pattern": pat,
                    "ratio": rate,
                    "n_records": len(sdf),
                    "imp_num_rmse_mean": round(sdf["imp_num_rmse"].dropna().mean(), 6) if sdf["imp_num_rmse"].notna().any() else None,
                    "runtime_mean": round(sdf["runtime_seconds"].dropna().mean(), 4) if sdf["runtime_seconds"].notna().any() else None,
                })
    return pd.DataFrame(results)


def generate_audit_report(df, metric_df, method_df, dataset_df, setting_df):
    lines = [
        "# Exploratory Audit Report",
        "",
        f"## Overview",
        f"- Total records: {len(df)}",
        f"- Datasets: {df['dataset'].nunique()}",
        f"- Methods: {df['method'].nunique()}",
        f"- Missing settings: {df['setting'].nunique()}",
        f"- Seeds: {df['seed'].nunique()}",
        "",
        "## Metric Inventory & Coverage",
        "",
        metric_df.to_markdown(index=False),
        "",
        "## Per-Method Summary",
        "",
        method_df.to_markdown(index=False),
        "",
        "## Per-Dataset Summary",
        "",
        dataset_df.to_markdown(index=False),
        "",
        "## Per-Setting Summary",
        "",
        setting_df.to_markdown(index=False),
        "",
        "## Notes",
        "- `imp_cat_accuracy` and `imp_cat_macro_f1` are only available for mixed-type datasets (adult, autism, bank, titanic)",
        "- `ds_roc_auc` is available for binary classification datasets only",
        "- `ds_rmse/mae/r2` is available for regression datasets only (concrete, wine)",
        "- `ds_accuracy/f1_weighted/f1_macro` is available for classification datasets",
        "- `dist_jsd_mean` is primarily available for mixed-type datasets",
        "- Some downstream metrics may be empty when the model fails to produce predictions",
    ]
    report = "\n".join(lines)
    out_path = REPORTS_DIR / "exploratory_audit.md"
    out_path.write_text(report)
    print(f"Written: {out_path}")


def main():
    print("=" * 60)
    print("Step 2: Exploratory Audit")
    print("=" * 60)

    df = load_data()
    print(f"Loaded {len(df)} records.")

    metric_df = metric_audit(df)
    print("\nMetric coverage:")
    print(metric_df[["metric", "available", "coverage_pct"]].to_string(index=False))

    method_df = per_method_summary(df)
    dataset_df = per_dataset_summary(df)
    setting_df = per_setting_summary(df)

    # Save tables
    method_df.to_csv(TABLES_DIR / "per_method_summary.csv", index=False)
    dataset_df.to_csv(TABLES_DIR / "per_dataset_summary.csv", index=False)
    setting_df.to_csv(TABLES_DIR / "per_setting_summary.csv", index=False)
    metric_df.to_csv(TABLES_DIR / "metric_coverage.csv", index=False)

    generate_audit_report(df, metric_df, method_df, dataset_df, setting_df)
    print("\nStep 2 complete.\n")


if __name__ == "__main__":
    main()
