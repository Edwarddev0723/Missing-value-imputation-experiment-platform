"""
Step 5: Statistical testing — Friedman test, pairwise Wilcoxon, effect sizes.
"""
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
from pathlib import Path

from paths import (
    AGGREGATED_PARQUET, AGGREGATED_CSV, TABLES_DIR, STATS_DIR, REPORTS_DIR,
    METHODS, NON_DL_METHODS, DL_METHODS
)


def load_data():
    df = pd.read_csv(AGGREGATED_CSV)
    return df


def pick_downstream_metric(row):
    if row["task_type"] == "classification":
        return row.get("ds_f1_weighted", np.nan)
    else:
        val = row.get("ds_rmse", np.nan)
        return -val if pd.notna(val) else np.nan


def compute_ranks(df, metric_col, higher_is_better):
    """Rank methods within each experimental condition."""
    group_cols = ["dataset", "mechanism", "pattern", "ratio", "seed"]
    ascending = not higher_is_better
    df = df.copy()
    df["_rank"] = df.groupby(group_cols)[metric_col].rank(
        ascending=ascending, method="average", na_option="bottom"
    )
    return df


def friedman_test(df, rank_col="_rank"):
    """
    Friedman test: are there significant differences among methods?
    Requires: each 'block' (setting) has all methods measured.
    """
    group_cols = ["dataset", "mechanism", "pattern", "ratio", "seed"]
    
    # Pivot: each row = one experimental condition, columns = methods
    pivot = df.pivot_table(
        index=group_cols, columns="method", values=rank_col, aggfunc="mean"
    ).dropna()
    
    if pivot.shape[1] < 3:
        return None, None, pivot
    
    # Friedman test
    method_arrays = [pivot[m].values for m in pivot.columns]
    stat, p_value = stats.friedmanchisquare(*method_arrays)
    
    return stat, p_value, pivot


def pairwise_wilcoxon(pivot, alpha=0.05, correction="holm"):
    """Pairwise Wilcoxon signed-rank tests with multiple comparison correction."""
    methods = list(pivot.columns)
    results = []
    
    for m1, m2 in combinations(methods, 2):
        d1 = pivot[m1].values
        d2 = pivot[m2].values
        
        # Remove pairs with NaN
        mask = ~(np.isnan(d1) | np.isnan(d2))
        d1_clean = d1[mask]
        d2_clean = d2[mask]
        
        if len(d1_clean) < 10:
            continue
        
        # Wilcoxon signed-rank test
        try:
            stat, p_value = stats.wilcoxon(d1_clean, d2_clean, alternative="two-sided")
        except ValueError:
            # All differences are zero
            stat, p_value = 0, 1.0
        
        # Effect size: r = Z / sqrt(N)
        n = len(d1_clean)
        z = stats.norm.ppf(1 - p_value / 2) if p_value < 1 else 0
        effect_r = z / np.sqrt(n)
        
        # Mean rank difference
        mean_diff = d1_clean.mean() - d2_clean.mean()
        
        results.append({
            "method_1": m1,
            "method_2": m2,
            "mean_rank_1": round(d1_clean.mean(), 4),
            "mean_rank_2": round(d2_clean.mean(), 4),
            "mean_rank_diff": round(mean_diff, 4),
            "wilcoxon_stat": round(stat, 2),
            "p_value_raw": p_value,
            "effect_size_r": round(effect_r, 4),
            "n_pairs": n,
            "winner": m1 if mean_diff < 0 else m2,
        })
    
    result_df = pd.DataFrame(results)
    
    if len(result_df) == 0:
        return result_df
    
    # Multiple comparison correction
    if correction == "holm":
        result_df = holm_correction(result_df)
    elif correction == "bh":
        result_df = bh_correction(result_df)
    
    result_df["significant"] = result_df["p_adjusted"] < alpha
    result_df = result_df.sort_values("p_adjusted")
    
    return result_df


def holm_correction(df):
    """Holm-Bonferroni correction."""
    df = df.sort_values("p_value_raw").reset_index(drop=True)
    n = len(df)
    adjusted = []
    for i in range(n):
        adj_p = min(df.loc[i, "p_value_raw"] * (n - i), 1.0)
        adjusted.append(adj_p)
    # Enforce monotonicity
    for i in range(1, n):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])
    df["p_adjusted"] = adjusted
    return df


def bh_correction(df):
    """Benjamini-Hochberg correction."""
    df = df.sort_values("p_value_raw").reset_index(drop=True)
    n = len(df)
    adjusted = []
    for i in range(n):
        adj_p = min(df.loc[i, "p_value_raw"] * n / (i + 1), 1.0)
        adjusted.append(adj_p)
    # Enforce monotonicity (reverse)
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    df["p_adjusted"] = adjusted
    return df


def key_comparisons(pairwise_df):
    """Extract key comparison results."""
    comparisons = [
        ("missforest", "xgboost", "missforest vs xgboost"),
        ("gain", "tabcsdi", "gain vs tabcsdi"),
        ("gain", "miracle", "gain vs miracle"),
        ("tabcsdi", "miracle", "tabcsdi vs miracle"),
    ]
    
    # Best non-DL vs best DL
    key_results = []
    
    for m1, m2, label in comparisons:
        row = pairwise_df[
            ((pairwise_df["method_1"] == m1) & (pairwise_df["method_2"] == m2)) |
            ((pairwise_df["method_1"] == m2) & (pairwise_df["method_2"] == m1))
        ]
        if len(row) > 0:
            r = row.iloc[0]
            key_results.append({
                "comparison": label,
                "winner": r["winner"],
                "mean_rank_diff": abs(r["mean_rank_diff"]),
                "p_adjusted": r["p_adjusted"],
                "significant": r["significant"],
                "effect_size_r": r["effect_size_r"],
            })
    
    return pd.DataFrame(key_results)


def baseline_gap_analysis(df):
    """Compare simple baselines (mean/median) vs advanced methods."""
    group_cols = ["dataset", "mechanism", "pattern", "ratio", "seed"]
    
    baseline_methods = ["mean", "median"]
    advanced_methods = [m for m in METHODS if m not in baseline_methods]
    
    results = []
    for adv in advanced_methods:
        for base in baseline_methods:
            base_vals = df[df["method"] == base].set_index(group_cols)["_rank"]
            adv_vals = df[df["method"] == adv].set_index(group_cols)["_rank"]
            
            common = base_vals.index.intersection(adv_vals.index)
            if len(common) < 10:
                continue
            
            b = base_vals.loc[common].values
            a = adv_vals.loc[common].values
            
            # Percentage of conditions where advanced beats baseline
            win_pct = np.mean(a < b) * 100
            
            results.append({
                "baseline": base,
                "advanced": adv,
                "advanced_family": "DL" if adv in DL_METHODS else "Non-DL",
                "baseline_avg_rank": round(b.mean(), 4),
                "advanced_avg_rank": round(a.mean(), 4),
                "rank_improvement": round(b.mean() - a.mean(), 4),
                "win_pct": round(win_pct, 1),
                "n_conditions": len(common),
            })
    
    return pd.DataFrame(results).sort_values("rank_improvement", ascending=False)


def generate_stats_report(friedman_stat, friedman_p, pairwise_df, key_comp_df, baseline_df):
    """Generate statistical testing report."""
    lines = [
        "# Statistical Testing Report",
        "",
        "## 1. Friedman Test (Global)",
        "",
        f"- Test statistic: {friedman_stat:.4f}" if friedman_stat else "- Test not applicable",
        f"- p-value: {friedman_p:.2e}" if friedman_p else "",
        "",
    ]
    
    if friedman_p is not None and friedman_p < 0.05:
        lines.append("**Result: Significant differences exist among methods (p < 0.05). Proceeding to pairwise comparisons.**")
    elif friedman_p is not None:
        lines.append("**Result: No significant differences among methods (p >= 0.05).**")
    
    lines += [
        "",
        "## 2. Pairwise Wilcoxon Signed-Rank Tests",
        f"- Total comparisons: {len(pairwise_df)}",
        f"- Significant after Holm correction (α=0.05): {pairwise_df['significant'].sum()}",
        "",
        "### Top 20 Most Significant Differences",
        "",
    ]
    
    top20 = pairwise_df.head(20)[["method_1", "method_2", "mean_rank_diff", "p_adjusted", "significant", "effect_size_r", "winner"]]
    for c in top20.select_dtypes(include=[np.floating]).columns:
        top20[c] = top20[c].round(4)
    lines.append(top20.to_markdown(index=False))
    
    lines += [
        "",
        "## 3. Key Comparisons",
        "",
    ]
    if len(key_comp_df) > 0:
        lines.append(key_comp_df.to_markdown(index=False))
    else:
        lines.append("No key comparisons available.")
    
    lines += [
        "",
        "## 4. Baseline Gap Analysis",
        "How much do advanced methods improve over simple baselines (mean/median)?",
        "",
    ]
    if len(baseline_df) > 0:
        top_baseline = baseline_df.head(20)
        for c in top_baseline.select_dtypes(include=[np.floating]).columns:
            top_baseline[c] = top_baseline[c].round(4)
        lines.append(top_baseline.to_markdown(index=False))
    
    lines += [
        "",
        "## Methodology Notes",
        "- Ranks computed within each (dataset, mechanism, pattern, ratio, seed) condition",
        "- Friedman test checks overall differences across all methods",
        "- Wilcoxon signed-rank test for pairwise comparisons (non-parametric, paired)",
        "- Holm-Bonferroni correction for multiple comparisons",
        "- Effect size r = Z / sqrt(N), interpreted as: small (0.1), medium (0.3), large (0.5)",
    ]
    
    report = "\n".join(lines)
    out_path = REPORTS_DIR / "statistical_testing_report.md"
    out_path.write_text(report)
    print(f"Written: {out_path}")


def main():
    print("=" * 60)
    print("Step 5: Statistical Testing")
    print("=" * 60)

    df = load_data()
    df["ds_metric"] = df.apply(pick_downstream_metric, axis=1)
    
    # Compute ranks on imputation RMSE
    df = compute_ranks(df, "imp_num_rmse", higher_is_better=False)
    
    # Friedman test
    print("\nFriedman test on imputation ranks...")
    friedman_stat, friedman_p, pivot = friedman_test(df)
    if friedman_stat is not None:
        print(f"  χ² = {friedman_stat:.4f}, p = {friedman_p:.2e}")
    
    # Pairwise Wilcoxon
    print("\nPairwise Wilcoxon tests...")
    pairwise_df = pairwise_wilcoxon(pivot)
    n_sig = pairwise_df["significant"].sum() if len(pairwise_df) > 0 else 0
    print(f"  {len(pairwise_df)} comparisons, {n_sig} significant after Holm correction")
    
    # Save
    pairwise_df.to_csv(STATS_DIR / "pairwise_wilcoxon_imputation.csv", index=False)
    
    # Also do for downstream
    print("\nFriedman test on downstream ranks...")
    df_ds = compute_ranks(df, "ds_metric", higher_is_better=True)
    fs_ds, fp_ds, pivot_ds = friedman_test(df_ds)
    if fs_ds is not None:
        print(f"  χ² = {fs_ds:.4f}, p = {fp_ds:.2e}")
    pw_ds = pairwise_wilcoxon(pivot_ds)
    pw_ds.to_csv(STATS_DIR / "pairwise_wilcoxon_downstream.csv", index=False)
    
    # Key comparisons
    key_comp = key_comparisons(pairwise_df)
    key_comp.to_csv(STATS_DIR / "key_comparisons.csv", index=False)
    print("\nKey comparisons:")
    if len(key_comp) > 0:
        print(key_comp.to_string(index=False))
    
    # Baseline gap
    baseline_df = baseline_gap_analysis(df)
    baseline_df.to_csv(STATS_DIR / "baseline_gap_analysis.csv", index=False)
    
    # Report
    generate_stats_report(friedman_stat, friedman_p, pairwise_df, key_comp, baseline_df)
    
    print("\nStep 5 complete.\n")


if __name__ == "__main__":
    main()
