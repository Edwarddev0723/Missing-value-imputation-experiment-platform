"""Summary table generation from experiment results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import ensure_dir, load_results_csv
from src.utils.logging import get_logger

logger = get_logger(__name__)


def generate_summary_table(
    results_csv: str = "results/tables/batch_summary.csv",
    output_dir: str = "results/tables",
) -> pd.DataFrame:
    """Generate a pivot-style summary table from batch results.

    Rows: methods, Columns: datasets × metrics.
    """
    df = load_results_csv(results_csv)
    ensure_dir(output_dir)

    # Filter completed runs
    df = df[df["status"] == "completed"]

    if df.empty:
        logger.warning("No completed experiments found.")
        return df

    # Identify metric columns
    imp_cols = [c for c in df.columns if c.startswith("imp_")]
    ds_cols = [c for c in df.columns if c.startswith("ds_")]
    metric_cols = imp_cols + ds_cols

    if not metric_cols:
        logger.warning("No metric columns found.")
        return df

    # Group by dataset, method, mechanism and compute mean ± std across seeds
    group_cols = ["dataset_name", "method_name", "missing_mechanism", "missing_rate"]
    existing_group = [c for c in group_cols if c in df.columns]

    summary = df.groupby(existing_group)[metric_cols].agg(["mean", "std"]).reset_index()

    # Flatten multi-level columns
    summary.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in summary.columns
    ]

    # Save
    output_path = Path(output_dir) / "summary_aggregated.csv"
    summary.to_csv(output_path, index=False)
    logger.info(f"Summary table saved to {output_path}")

    return summary


# TODO: LaTeX table export
# TODO: Recommendation matrix generation
