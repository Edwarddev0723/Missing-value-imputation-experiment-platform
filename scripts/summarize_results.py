"""Aggregate raw experiment JSON files into summary CSV tables.

Usage:
    python -m scripts.summarize_results [--input results/raw] [--output results/aggregated]

Reads all JSON files from the input directory, aggregates mean ± std across seeds,
and produces three CSVs:
    - imputation_quality.csv
    - downstream_performance.csv
    - distribution_preservation.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_all_results(input_dir: Path) -> list[dict]:
    """Load all raw JSON result files."""
    results = []
    for json_path in sorted(input_dir.glob("*.json")):
        with open(json_path) as f:
            results.append(json.load(f))
    return results


def flatten_result(r: dict) -> dict:
    """Flatten a single spec-compliant JSON into a flat row."""
    iq = r.get("imputation_quality", {})
    dp = r.get("downstream_performance", {})
    dist = r.get("distribution_preservation", {})
    return {
        "dataset_id": r.get("dataset_id", ""),
        "dataset_name": r.get("dataset_name", ""),
        "method_id": r.get("method_id", ""),
        "method_name": r.get("method_name", ""),
        "mechanism": r.get("mechanism", ""),
        "pattern": r.get("pattern", ""),
        "missing_rate": r.get("missing_rate", 0),
        "seed": r.get("seed", 0),
        "n_samples": r.get("n_samples", 0),
        "n_features": r.get("n_features", 0),
        "runtime_seconds": r.get("runtime_seconds", 0),
        # Imputation quality
        "numerical_rmse": iq.get("numerical", {}).get("rmse"),
        "numerical_mae": iq.get("numerical", {}).get("mae"),
        "categorical_accuracy": iq.get("categorical", {}).get("accuracy"),
        "categorical_macro_f1": iq.get("categorical", {}).get("macro_f1"),
        # Downstream
        "downstream_task_type": dp.get("task_type", ""),
        **{f"downstream_{k}": v for k, v in dp.get("metrics", {}).items()},
        # Distribution
        "wasserstein_distance_mean": dist.get("wasserstein_distance_mean"),
        "jsd_mean": dist.get("jsd_mean"),
    }


def aggregate(df: pd.DataFrame, value_cols: list[str], group_cols: list[str]) -> pd.DataFrame:
    """Group by (dataset, method, mechanism, pattern, rate) and compute mean ± std over seeds."""
    agg_dict = {}
    for col in value_cols:
        if col in df.columns:
            agg_dict[col] = ["mean", "std"]
    agg_dict["seed"] = "count"

    grouped = df.groupby(group_cols, dropna=False).agg(agg_dict)
    # Flatten multi-level columns
    grouped.columns = ["_".join(c).strip("_") for c in grouped.columns]
    grouped = grouped.rename(columns={"seed_count": "n_seeds"})
    return grouped.reset_index()


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results.")
    parser.add_argument("--input", default="results/raw", help="Directory with raw JSON results")
    parser.add_argument("--output", default="results/aggregated", help="Output directory for CSVs")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_all_results(input_dir)
    if not results:
        print(f"No JSON files found in {input_dir}")
        return

    rows = [flatten_result(r) for r in results]
    df = pd.DataFrame(rows)

    group_cols = ["dataset_id", "dataset_name", "method_id", "method_name", "mechanism", "pattern", "missing_rate"]

    # Imputation quality
    iq_cols = ["numerical_rmse", "numerical_mae", "categorical_accuracy", "categorical_macro_f1"]
    iq_agg = aggregate(df, iq_cols, group_cols)
    iq_agg.to_csv(output_dir / "imputation_quality.csv", index=False)
    print(f"Saved {output_dir / 'imputation_quality.csv'} ({len(iq_agg)} rows)")

    # Downstream performance — gather all downstream_ columns dynamically
    ds_cols = [c for c in df.columns if c.startswith("downstream_") and c != "downstream_task_type"]
    ds_agg = aggregate(df, ds_cols, group_cols)
    ds_agg.to_csv(output_dir / "downstream_performance.csv", index=False)
    print(f"Saved {output_dir / 'downstream_performance.csv'} ({len(ds_agg)} rows)")

    # Distribution preservation
    dist_cols = ["wasserstein_distance_mean", "jsd_mean"]
    dist_agg = aggregate(df, dist_cols, group_cols)
    dist_agg.to_csv(output_dir / "distribution_preservation.csv", index=False)
    print(f"Saved {output_dir / 'distribution_preservation.csv'} ({len(dist_agg)} rows)")


if __name__ == "__main__":
    main()
