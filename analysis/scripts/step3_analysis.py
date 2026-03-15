"""
Step 3: Stratified analysis with within-dataset normalization.
Computes standardized ranks, normalized scores, and stratified comparisons.
"""
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from paths import (
    AGGREGATED_PARQUET, AGGREGATED_CSV, TABLES_DIR, REPORTS_DIR, STATS_DIR,
    DATASETS, METHODS, NON_DL_METHODS, DL_METHODS,
    MECHANISMS, PATTERNS, RATES, SEEDS,
    DATASET_TRACK_CONFIG, METHOD_ELIGIBILITY_CONFIG
)


def load_data():
    return pd.read_csv(AGGREGATED_CSV)


def load_configs():
    with open(DATASET_TRACK_CONFIG) as f:
        track_cfg = yaml.safe_load(f)
    with open(METHOD_ELIGIBILITY_CONFIG) as f:
        method_cfg = yaml.safe_load(f)
    return track_cfg, method_cfg


def pick_downstream_metric(row):
    """Return the primary downstream metric value for a row."""
    if row["task_type"] == "classification":
        return row.get("ds_f1_weighted", np.nan)
    else:
        # For regression, negate RMSE so higher = better (for ranking consistency)
        val = row.get("ds_rmse", np.nan)
        return -val if pd.notna(val) else np.nan


def compute_within_dataset_ranks(df, metric_col, higher_is_better=True):
    """
    For each (dataset, mechanism, pattern, ratio, seed),
    rank methods from 1 (best) to n_methods (worst).
    """
    group_cols = ["dataset", "mechanism", "pattern", "ratio", "seed"]
    ascending = not higher_is_better
    return df.groupby(group_cols)[metric_col].rank(
        ascending=ascending, method="average", na_option="bottom"
    )


def compute_within_dataset_normalized_score(df, metric_col, higher_is_better=True):
    """
    Min-max normalize within each (dataset, mechanism, pattern, ratio, seed) group.
    Score is 0-1 where 1 = best.
    """
    group_cols = ["dataset", "mechanism", "pattern", "ratio", "seed"]
    grouped = df.groupby(group_cols)[metric_col]
    vmin = grouped.transform("min")
    vmax = grouped.transform("max")
    rng = vmax - vmin
    vals = df[metric_col].astype(float)
    if higher_is_better:
        score = (vals - vmin) / rng
    else:
        score = (vmax - vals) / rng
    score = score.fillna(0.5)
    # Where range is 0, set to 0.5
    score = score.where(rng > 0, 0.5)
    return score


def overall_analysis(df):
    """Section A: Overall as-executed analysis."""
    print("\n--- A. Overall Analysis ---")
    
    # Compute ranks on imputation RMSE (lower is better)
    df["imp_rank"] = compute_within_dataset_ranks(df, "imp_num_rmse", higher_is_better=False)
    df["imp_norm_score"] = compute_within_dataset_normalized_score(df, "imp_num_rmse", higher_is_better=False)
    
    # For downstream: use task-appropriate metric
    df["ds_metric"] = df.apply(pick_downstream_metric, axis=1)
    df["ds_rank"] = compute_within_dataset_ranks(df, "ds_metric", higher_is_better=True)
    df["ds_norm_score"] = compute_within_dataset_normalized_score(df, "ds_metric", higher_is_better=True)
    
    # Overall average rank per method
    overall_imp = df.groupby("method").agg(
        imp_avg_rank=("imp_rank", "mean"),
        imp_std_rank=("imp_rank", "std"),
        imp_avg_norm=("imp_norm_score", "mean"),
        ds_avg_rank=("ds_rank", "mean"),
        ds_std_rank=("ds_rank", "std"),
        ds_avg_norm=("ds_norm_score", "mean"),
        runtime_mean=("runtime_seconds", "mean"),
        runtime_median=("runtime_seconds", "median"),
        n_records=("method", "count"),
    ).reset_index()
    
    overall_imp["family"] = overall_imp["method"].apply(
        lambda m: "DL" if m in DL_METHODS else "Non-DL"
    )
    overall_imp = overall_imp.sort_values("imp_avg_rank")
    
    # Round for readability
    for c in ["imp_avg_rank", "imp_std_rank", "imp_avg_norm", "ds_avg_rank", "ds_std_rank", "ds_avg_norm", "runtime_mean", "runtime_median"]:
        overall_imp[c] = overall_imp[c].round(4)
    
    overall_imp.to_csv(TABLES_DIR / "overall_method_ranking.csv", index=False)
    print(f"  Overall method ranking saved. Top 5 by imp_avg_rank:")
    print(overall_imp[["method", "family", "imp_avg_rank", "ds_avg_rank", "runtime_mean"]].head().to_string(index=False))
    
    return df, overall_imp


def dataset_wise_analysis(df):
    """Section B: Dataset-wise analysis."""
    print("\n--- B. Dataset-wise Analysis ---")
    
    # Per dataset x method: average rank across settings and seeds
    ds_method = df.groupby(["dataset", "method"]).agg(
        imp_avg_rank=("imp_rank", "mean"),
        imp_avg_norm=("imp_norm_score", "mean"),
        ds_avg_rank=("ds_rank", "mean"),
        ds_avg_norm=("ds_norm_score", "mean"),
        imp_rmse_mean=("imp_num_rmse", "mean"),
        imp_rmse_std=("imp_num_rmse", "std"),
        runtime_mean=("runtime_seconds", "mean"),
    ).reset_index()
    ds_method["family"] = ds_method["method"].apply(lambda m: "DL" if m in DL_METHODS else "Non-DL")
    
    for c in ds_method.select_dtypes(include=[np.floating]).columns:
        ds_method[c] = ds_method[c].round(4)
    
    ds_method.to_csv(TABLES_DIR / "dataset_method_ranking.csv", index=False)
    
    # Best method per dataset
    best_per_ds = ds_method.loc[ds_method.groupby("dataset")["imp_avg_rank"].idxmin()]
    best_per_ds = best_per_ds[["dataset", "method", "family", "imp_avg_rank", "ds_avg_rank"]].sort_values("dataset")
    best_per_ds.to_csv(TABLES_DIR / "best_method_per_dataset.csv", index=False)
    print("  Best method per dataset (by imputation rank):")
    print(best_per_ds.to_string(index=False))
    
    # Effect of ratio per dataset
    ratio_effect = df.groupby(["dataset", "method", "ratio"]).agg(
        imp_rmse_mean=("imp_num_rmse", "mean"),
        imp_rank_mean=("imp_rank", "mean"),
    ).reset_index()
    ratio_effect.to_csv(TABLES_DIR / "ratio_effect_per_dataset.csv", index=False)
    
    # Effect of mechanism per dataset
    mech_effect = df.groupby(["dataset", "method", "mechanism"]).agg(
        imp_rmse_mean=("imp_num_rmse", "mean"),
        imp_rank_mean=("imp_rank", "mean"),
    ).reset_index()
    mech_effect.to_csv(TABLES_DIR / "mechanism_effect_per_dataset.csv", index=False)
    
    # Effect of pattern per dataset
    pattern_effect = df.groupby(["dataset", "method", "pattern"]).agg(
        imp_rmse_mean=("imp_num_rmse", "mean"),
        imp_rank_mean=("imp_rank", "mean"),
    ).reset_index()
    pattern_effect.to_csv(TABLES_DIR / "pattern_effect_per_dataset.csv", index=False)
    
    return ds_method


def method_family_analysis(df, overall_imp):
    """Section C: Method family analysis."""
    print("\n--- C. Method Family Analysis ---")
    
    family_summary = df.groupby("method_family").agg(
        imp_avg_rank=("imp_rank", "mean"),
        imp_std_rank=("imp_rank", "std"),
        imp_avg_norm=("imp_norm_score", "mean"),
        ds_avg_rank=("ds_rank", "mean"),
        ds_avg_norm=("ds_norm_score", "mean"),
        runtime_mean=("runtime_seconds", "mean"),
        runtime_median=("runtime_seconds", "median"),
    ).reset_index()
    
    for c in family_summary.select_dtypes(include=[np.floating]).columns:
        family_summary[c] = family_summary[c].round(4)
    
    family_summary.to_csv(TABLES_DIR / "family_comparison.csv", index=False)
    print("  Family comparison:")
    print(family_summary.to_string(index=False))
    
    # Stability: std of rank across all settings
    stability = df.groupby("method").agg(
        imp_rank_std=("imp_rank", "std"),
        imp_norm_std=("imp_norm_score", "std"),
        ds_rank_std=("ds_rank", "std"),
    ).reset_index()
    stability["family"] = stability["method"].apply(lambda m: "DL" if m in DL_METHODS else "Non-DL")
    stability = stability.sort_values("imp_rank_std")
    for c in stability.select_dtypes(include=[np.floating]).columns:
        stability[c] = stability[c].round(4)
    stability.to_csv(TABLES_DIR / "stability_ranking.csv", index=False)
    print("\n  Most stable methods (lowest rank std):")
    print(stability[["method", "family", "imp_rank_std", "ds_rank_std"]].head(5).to_string(index=False))
    
    # High missing rate resilience (ratio=0.5)
    high_miss = df[df["ratio"] == 0.5].groupby("method").agg(
        imp_avg_rank=("imp_rank", "mean"),
        ds_avg_rank=("ds_rank", "mean"),
    ).reset_index().sort_values("imp_avg_rank")
    high_miss["family"] = high_miss["method"].apply(lambda m: "DL" if m in DL_METHODS else "Non-DL")
    for c in high_miss.select_dtypes(include=[np.floating]).columns:
        high_miss[c] = high_miss[c].round(4)
    high_miss.to_csv(TABLES_DIR / "high_missing_rate_ranking.csv", index=False)
    print("\n  Best methods at 50% missing rate:")
    print(high_miss.head(5).to_string(index=False))
    
    # MNAR sensitivity
    mnar_ranks = df[df["mechanism"] == "MNAR"].groupby("method").agg(
        imp_avg_rank_mnar=("imp_rank", "mean"),
    ).reset_index()
    non_mnar_ranks = df[df["mechanism"] != "MNAR"].groupby("method").agg(
        imp_avg_rank_other=("imp_rank", "mean"),
    ).reset_index()
    mnar_sensitivity = mnar_ranks.merge(non_mnar_ranks, on="method")
    mnar_sensitivity["mnar_degradation"] = mnar_sensitivity["imp_avg_rank_mnar"] - mnar_sensitivity["imp_avg_rank_other"]
    mnar_sensitivity["family"] = mnar_sensitivity["method"].apply(lambda m: "DL" if m in DL_METHODS else "Non-DL")
    mnar_sensitivity = mnar_sensitivity.sort_values("mnar_degradation", ascending=False)
    for c in mnar_sensitivity.select_dtypes(include=[np.floating]).columns:
        mnar_sensitivity[c] = mnar_sensitivity[c].round(4)
    mnar_sensitivity.to_csv(TABLES_DIR / "mnar_sensitivity.csv", index=False)
    
    return family_summary, stability


def interaction_analysis(df):
    """Section D: Interaction analysis."""
    print("\n--- D. Interaction Analysis ---")
    
    interactions = {
        "method_x_mechanism": ["method", "mechanism"],
        "method_x_ratio": ["method", "ratio"],
        "method_x_pattern": ["method", "pattern"],
        "method_x_dataset": ["method", "dataset"],
        "family_x_ratio": ["method_family", "ratio"],
        "family_x_mechanism": ["method_family", "mechanism"],
    }
    
    for name, group_cols in interactions.items():
        tbl = df.groupby(group_cols).agg(
            imp_avg_rank=("imp_rank", "mean"),
            imp_avg_norm=("imp_norm_score", "mean"),
            ds_avg_rank=("ds_rank", "mean"),
            ds_avg_norm=("ds_norm_score", "mean"),
            n=("method", "count"),
        ).reset_index()
        for c in tbl.select_dtypes(include=[np.floating]).columns:
            tbl[c] = tbl[c].round(4)
        tbl.to_csv(TABLES_DIR / f"interaction_{name}.csv", index=False)
        print(f"  Saved: interaction_{name}.csv")


def track_analysis(df):
    """Step 4: Compatibility-aware fair analysis by track."""
    print("\n--- Step 4: Track Analysis ---")
    
    track_cfg, method_cfg = load_configs()
    
    tracks = track_cfg["tracks"]
    eligibility = method_cfg["track_eligibility"]
    
    for track_name, track_info in tracks.items():
        track_datasets = track_info["datasets"]
        eligible_methods = [m for m, info in eligibility.items() if track_name in info["tracks"]]
        
        track_df = df[(df["dataset"].isin(track_datasets)) & (df["method"].isin(eligible_methods))]
        
        if len(track_df) == 0:
            print(f"  {track_name}: No data (skipping)")
            continue
        
        # Re-compute ranks within track
        track_df = track_df.copy()
        track_df["track_imp_rank"] = compute_within_dataset_ranks(track_df, "imp_num_rmse", higher_is_better=False)
        track_df["track_ds_rank"] = compute_within_dataset_ranks(track_df, "ds_metric", higher_is_better=True)
        
        track_summary = track_df.groupby("method").agg(
            imp_avg_rank=("track_imp_rank", "mean"),
            imp_std_rank=("track_imp_rank", "std"),
            ds_avg_rank=("track_ds_rank", "mean"),
            ds_std_rank=("track_ds_rank", "std"),
            runtime_mean=("runtime_seconds", "mean"),
            n_records=("method", "count"),
        ).reset_index()
        track_summary["family"] = track_summary["method"].apply(lambda m: "DL" if m in DL_METHODS else "Non-DL")
        track_summary = track_summary.sort_values("imp_avg_rank")
        for c in track_summary.select_dtypes(include=[np.floating]).columns:
            track_summary[c] = track_summary[c].round(4)
        
        track_summary.to_csv(TABLES_DIR / f"track_{track_name}_ranking.csv", index=False)
        print(f"  {track_name} track: {len(track_datasets)} datasets, {len(eligible_methods)} methods, {len(track_df)} records")
        print(track_summary[["method", "family", "imp_avg_rank", "ds_avg_rank"]].head(5).to_string(index=False))
        print()


def degradation_analysis(df):
    """Compute degradation from lowest to highest missing rate."""
    print("\n--- Degradation Analysis ---")
    
    low_rate = df[df["ratio"] == 0.1].groupby("method")["imp_num_rmse"].mean()
    high_rate = df[df["ratio"] == 0.5].groupby("method")["imp_num_rmse"].mean()
    
    degrad = pd.DataFrame({
        "method": low_rate.index,
        "rmse_at_10pct": low_rate.values,
        "rmse_at_50pct": high_rate.reindex(low_rate.index).values,
    })
    degrad["abs_degradation"] = degrad["rmse_at_50pct"] - degrad["rmse_at_10pct"]
    degrad["rel_degradation_pct"] = ((degrad["rmse_at_50pct"] - degrad["rmse_at_10pct"]) / degrad["rmse_at_10pct"] * 100)
    degrad["family"] = degrad["method"].apply(lambda m: "DL" if m in DL_METHODS else "Non-DL")
    degrad = degrad.sort_values("rel_degradation_pct")
    for c in degrad.select_dtypes(include=[np.floating]).columns:
        degrad[c] = degrad[c].round(4)
    degrad.to_csv(TABLES_DIR / "degradation_analysis.csv", index=False)
    print("  Degradation (10% -> 50% missing):")
    print(degrad.to_string(index=False))
    
    return degrad


def main():
    print("=" * 60)
    print("Steps 3-4: Stratified & Track Analysis")
    print("=" * 60)

    df = load_data()
    
    # Compute downstream unified metric
    df["ds_metric"] = df.apply(pick_downstream_metric, axis=1)
    
    # A. Overall
    df, overall_imp = overall_analysis(df)
    
    # B. Dataset-wise
    ds_method = dataset_wise_analysis(df)
    
    # C. Method family
    family_summary, stability = method_family_analysis(df, overall_imp)
    
    # D. Interactions
    interaction_analysis(df)
    
    # Step 4: Track analysis
    track_analysis(df)
    
    # Degradation
    degradation_analysis(df)
    
    # Save enriched dataframe
    df.to_csv(STATS_DIR / "enriched_results.csv", index=False)
    try:
        df.to_parquet(STATS_DIR / "enriched_results.parquet", index=False)
    except ImportError:
        pass
    
    print("\nSteps 3-4 complete.\n")
    return df


if __name__ == "__main__":
    main()
