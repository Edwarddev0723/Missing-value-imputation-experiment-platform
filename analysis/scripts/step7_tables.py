"""
Step 7: Generate paper-ready tables in CSV, LaTeX, and Markdown.
"""
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from paths import (
    AGGREGATED_PARQUET, AGGREGATED_CSV, TABLES_DIR, STATS_DIR,
    METHODS, NON_DL_METHODS, DL_METHODS, DATASETS,
    MECHANISMS, RATES, DATASET_TRACK_CONFIG
)


def load_data():
    return pd.read_csv(AGGREGATED_CSV)


def pick_downstream_metric(row):
    if row["task_type"] == "classification":
        return row.get("ds_f1_weighted", np.nan)
    else:
        val = row.get("ds_rmse", np.nan)
        return -val if pd.notna(val) else np.nan


def compute_ranks(df, metric_col, higher_is_better):
    group_cols = ["dataset", "mechanism", "pattern", "ratio", "seed"]
    ascending = not higher_is_better
    df = df.copy()
    df["_rank"] = df.groupby(group_cols)[metric_col].rank(
        ascending=ascending, method="average", na_option="bottom"
    )
    return df


def save_table(df, name, index=False):
    """Save table in CSV, LaTeX, and Markdown."""
    df.to_csv(TABLES_DIR / f"{name}.csv", index=index)
    
    # LaTeX
    latex = df.to_latex(index=index, float_format="%.4f", escape=True)
    (TABLES_DIR / f"{name}.tex").write_text(latex)
    
    # Markdown
    md = df.to_markdown(index=index)
    (TABLES_DIR / f"{name}.md").write_text(md)
    
    print(f"  Saved: {name} (csv/tex/md)")


def table1_overall_summary(df):
    """Overall summary table."""
    overall = df.groupby("method").agg(
        Family=("method_family", "first"),
        Imp_Avg_Rank=("_rank", "mean"),
        Imp_Rank_Std=("_rank", "std"),
        DS_Avg_Rank=("_ds_rank", "mean"),
        Imp_RMSE_Mean=("imp_num_rmse", "mean"),
        Imp_RMSE_Std=("imp_num_rmse", "std"),
        Runtime_Mean=("runtime_seconds", "mean"),
        Runtime_Median=("runtime_seconds", "median"),
        N=("method", "count"),
    ).reset_index()
    overall.columns = ["Method", "Family", "Imp Avg Rank", "Imp Rank Std",
                        "DS Avg Rank", "Imp RMSE Mean", "Imp RMSE Std",
                        "Runtime Mean (s)", "Runtime Median (s)", "N"]
    overall = overall.sort_values("Imp Avg Rank")
    for c in overall.select_dtypes(include=[np.floating]).columns:
        overall[c] = overall[c].round(4)
    save_table(overall, "table1_overall_summary")
    return overall


def table2_dataset_best(df):
    """Best method per dataset."""
    ds_method = df.groupby(["dataset", "method"]).agg(
        imp_avg_rank=("_rank", "mean"),
        ds_avg_rank=("_ds_rank", "mean"),
        imp_rmse=("imp_num_rmse", "mean"),
    ).reset_index()
    
    best = ds_method.loc[ds_method.groupby("dataset")["imp_avg_rank"].idxmin()]
    best = best[["dataset", "method", "imp_avg_rank", "ds_avg_rank", "imp_rmse"]]
    best.columns = ["Dataset", "Best Method", "Imp Avg Rank", "DS Avg Rank", "Imp RMSE"]
    best = best.sort_values("Dataset")
    for c in best.select_dtypes(include=[np.floating]).columns:
        best[c] = best[c].round(4)
    save_table(best, "table2_dataset_best_method")
    return best


def table3_mechanism_comparison(df):
    """Per-mechanism method rankings."""
    tbl = df.groupby(["method", "mechanism"]).agg(
        imp_avg_rank=("_rank", "mean"),
    ).reset_index()
    pivot = tbl.pivot(index="method", columns="mechanism", values="imp_avg_rank")
    pivot = pivot[["MCAR", "MAR", "MNAR"]]
    pivot["Overall"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("Overall").round(2)
    pivot = pivot.reset_index()
    pivot.columns = ["Method", "MCAR", "MAR", "MNAR", "Overall"]
    save_table(pivot, "table3_mechanism_comparison")
    return pivot


def table4_ratio_comparison(df):
    """Per-ratio method rankings."""
    tbl = df.groupby(["method", "ratio"]).agg(
        imp_avg_rank=("_rank", "mean"),
    ).reset_index()
    pivot = tbl.pivot(index="method", columns="ratio", values="imp_avg_rank")
    pivot.columns = [f"{int(r*100)}%" for r in sorted(RATES)]
    pivot["Overall"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("Overall").round(2)
    pivot = pivot.reset_index()
    save_table(pivot, "table4_ratio_comparison")
    return pivot


def table5_dl_vs_nondl(df):
    """DL vs Non-DL summary."""
    tbl = df.groupby("method_family").agg(
        Imp_Avg_Rank=("_rank", "mean"),
        Imp_Rank_Std=("_rank", "std"),
        DS_Avg_Rank=("_ds_rank", "mean"),
        Imp_RMSE_Mean=("imp_num_rmse", "mean"),
        Runtime_Mean=("runtime_seconds", "mean"),
        Runtime_Median=("runtime_seconds", "median"),
        N=("method", "count"),
    ).reset_index()
    tbl.columns = ["Family", "Imp Avg Rank", "Imp Rank Std", "DS Avg Rank",
                    "Imp RMSE Mean", "Runtime Mean (s)", "Runtime Median (s)", "N"]
    for c in tbl.select_dtypes(include=[np.floating]).columns:
        tbl[c] = tbl[c].round(4)
    save_table(tbl, "table5_dl_vs_nondl")
    return tbl


def table6_top3(df):
    """Top-3 methods by various criteria."""
    results = {}
    
    # Overall rank
    overall = df.groupby("method")["_rank"].mean().sort_values()
    results["Overall"] = list(overall.head(3).index)
    
    # Numeric track
    with open(DATASET_TRACK_CONFIG) as f:
        tcfg = yaml.safe_load(f)
    num_ds = tcfg["tracks"]["pure_numerical"]["datasets"]
    mixed_ds = tcfg["tracks"]["mixed_type"]["datasets"]
    
    num_ranks = df[df["dataset"].isin(num_ds)].groupby("method")["_rank"].mean().sort_values()
    results["Numeric Track"] = list(num_ranks.head(3).index)
    
    mix_ranks = df[df["dataset"].isin(mixed_ds)].groupby("method")["_rank"].mean().sort_values()
    results["Mixed Track"] = list(mix_ranks.head(3).index)
    
    # MNAR only
    mnar_ranks = df[df["mechanism"] == "MNAR"].groupby("method")["_rank"].mean().sort_values()
    results["MNAR Only"] = list(mnar_ranks.head(3).index)
    
    # High missing (50%)
    high_ranks = df[df["ratio"] == 0.5].groupby("method")["_rank"].mean().sort_values()
    results["High Missing (50%)"] = list(high_ranks.head(3).index)
    
    tbl = pd.DataFrame(results).T
    tbl.columns = ["1st", "2nd", "3rd"]
    tbl.index.name = "Category"
    tbl = tbl.reset_index()
    save_table(tbl, "table6_top3_methods")
    return tbl


def table7_stability(df):
    """Stability ranking."""
    stab = df.groupby("method").agg(
        imp_rank_std=("_rank", "std"),
        ds_rank_std=("_ds_rank", "std"),
        imp_rmse_cv=("imp_num_rmse", lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan),
    ).reset_index()
    stab["Family"] = stab["method"].apply(lambda m: "DL" if m in DL_METHODS else "Non-DL")
    stab = stab.sort_values("imp_rank_std")
    stab.columns = ["Method", "Imp Rank Std", "DS Rank Std", "Imp RMSE CV", "Family"]
    for c in stab.select_dtypes(include=[np.floating]).columns:
        stab[c] = stab[c].round(4)
    save_table(stab, "table7_stability_ranking")
    return stab


def table8_runtime(df):
    """Runtime efficiency table."""
    rt = df.groupby("method").agg(
        runtime_mean=("runtime_seconds", "mean"),
        runtime_median=("runtime_seconds", "median"),
        runtime_std=("runtime_seconds", "std"),
        runtime_min=("runtime_seconds", "min"),
        runtime_max=("runtime_seconds", "max"),
    ).reset_index()
    rt["Family"] = rt["method"].apply(lambda m: "DL" if m in DL_METHODS else "Non-DL")
    rt = rt.sort_values("runtime_mean")
    rt.columns = ["Method", "Mean (s)", "Median (s)", "Std (s)", "Min (s)", "Max (s)", "Family"]
    for c in rt.select_dtypes(include=[np.floating]).columns:
        rt[c] = rt[c].round(4)
    save_table(rt, "table8_runtime_efficiency")
    return rt


def main():
    print("=" * 60)
    print("Step 7: Paper-Ready Tables")
    print("=" * 60)

    df = load_data()
    df["ds_metric"] = df.apply(pick_downstream_metric, axis=1)
    df["method_family"] = df["method"].apply(lambda m: "DL" if m in DL_METHODS else "Non-DL")
    df = compute_ranks(df, "imp_num_rmse", higher_is_better=False)
    
    # Downstream ranks
    group_cols = ["dataset", "mechanism", "pattern", "ratio", "seed"]
    df["_ds_rank"] = df.groupby(group_cols)["ds_metric"].rank(
        ascending=False, method="average", na_option="bottom"
    )
    
    table1_overall_summary(df)
    table2_dataset_best(df)
    table3_mechanism_comparison(df)
    table4_ratio_comparison(df)
    table5_dl_vs_nondl(df)
    table6_top3(df)
    table7_stability(df)
    table8_runtime(df)
    
    print(f"\nAll tables saved to: {TABLES_DIR}")
    print("Step 7 complete.\n")


if __name__ == "__main__":
    main()
