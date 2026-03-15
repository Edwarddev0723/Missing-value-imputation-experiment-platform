"""
Step 6: Publication-ready visualizations.
All figures saved as PNG and PDF.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

from paths import (
    AGGREGATED_PARQUET, AGGREGATED_CSV, TABLES_DIR, FIGURES_DIR, STATS_DIR,
    METHODS, NON_DL_METHODS, DL_METHODS,
    DATASETS, MECHANISMS, PATTERNS, RATES,
    FIGURE_DPI, FIGURE_FORMAT
)

# Publication style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": FIGURE_DPI,
    "savefig.dpi": FIGURE_DPI,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

# Consistent color palette
METHOD_ORDER = [
    "mean", "median", "em", "mice", "knn", "missforest", "softimpute", "xgboost",
    "mida", "miwae", "gain", "tabcsdi", "miracle"
]
DL_COLOR = "#e74c3c"
NONDL_COLOR = "#2980b9"
METHOD_COLORS = {}
nondl_palette = sns.color_palette("Blues_d", len(NON_DL_METHODS))
dl_palette = sns.color_palette("Reds_d", len(DL_METHODS))
for i, m in enumerate(NON_DL_METHODS):
    METHOD_COLORS[m] = nondl_palette[i]
for i, m in enumerate(DL_METHODS):
    METHOD_COLORS[m] = dl_palette[i]


def save_fig(fig, name):
    for fmt in FIGURE_FORMAT:
        fig.savefig(FIGURES_DIR / f"{name}.{fmt}", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


def load_data():
    return pd.read_csv(AGGREGATED_CSV)


def pick_downstream_metric(row):
    if row["task_type"] == "classification":
        return row.get("ds_f1_weighted", np.nan)
    else:
        val = row.get("ds_rmse", np.nan)
        return -val if pd.notna(val) else np.nan


def compute_within_dataset_ranks(df, metric_col, higher_is_better=True):
    group_cols = ["dataset", "mechanism", "pattern", "ratio", "seed"]
    ascending = not higher_is_better
    df = df.copy()
    df["_rank"] = df.groupby(group_cols)[metric_col].rank(
        ascending=ascending, method="average", na_option="bottom"
    )
    return df


def compute_norm_score(df, metric_col, higher_is_better=True):
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
    score = score.fillna(0.5).where(rng > 0, 0.5)
    df = df.copy()
    df["_norm"] = score
    return df


def fig1_heatmap_rank(df):
    """Heatmap: method × dataset average imputation rank."""
    pivot = df.groupby(["dataset", "method"])["_rank"].mean().unstack("method")
    pivot = pivot[METHOD_ORDER].reindex(sorted(DATASETS))
    
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r", ax=ax,
                linewidths=0.5, vmin=1, vmax=13, cbar_kws={"label": "Average Rank (lower=better)"})
    ax.set_title("Average Imputation Rank: Method × Dataset")
    ax.set_xlabel("Method")
    ax.set_ylabel("Dataset")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    save_fig(fig, "fig1_heatmap_method_dataset_rank")


def fig2_line_ratio(df):
    """Line plot: imputation RMSE vs missing ratio, per method, faceted by mechanism."""
    agg = df.groupby(["method", "mechanism", "ratio"])["imp_num_rmse"].mean().reset_index()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for i, mech in enumerate(MECHANISMS):
        ax = axes[i]
        mdf = agg[agg["mechanism"] == mech]
        for method in METHOD_ORDER:
            sub = mdf[mdf["method"] == method]
            if len(sub) > 0:
                ls = "-" if method in NON_DL_METHODS else "--"
                lw = 1.8 if method in NON_DL_METHODS else 2.2
                ax.plot(sub["ratio"], sub["imp_num_rmse"], marker="o", label=method,
                        color=METHOD_COLORS[method], linestyle=ls, linewidth=lw, markersize=4)
        ax.set_title(f"{mech}")
        ax.set_xlabel("Missing Ratio")
        if i == 0:
            ax.set_ylabel("Imputation RMSE (lower=better)")
        ax.set_xticks(RATES)
        ax.grid(alpha=0.3)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.12, 0.5), fontsize=9)
    fig.suptitle("Imputation RMSE vs Missing Ratio by Mechanism", fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, "fig2_line_ratio_by_mechanism")


def fig3_boxplot_rank(df):
    """Boxplot: rank distribution across all conditions for each method."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    plot_data = []
    for method in METHOD_ORDER:
        ranks = df[df["method"] == method]["_rank"]
        plot_data.append(ranks)
    
    bp = ax.boxplot(plot_data, labels=METHOD_ORDER, patch_artist=True, showfliers=False,
                     medianprops=dict(color="black", linewidth=1.5))
    
    for i, method in enumerate(METHOD_ORDER):
        color = METHOD_COLORS[method]
        bp["boxes"][i].set_facecolor(color)
        bp["boxes"][i].set_alpha(0.7)
    
    ax.set_ylabel("Rank (lower=better)")
    ax.set_title("Imputation Rank Distribution Across All Conditions")
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    
    # Add family legend
    dl_patch = mpatches.Patch(color=DL_COLOR, alpha=0.5, label="DL methods")
    nondl_patch = mpatches.Patch(color=NONDL_COLOR, alpha=0.5, label="Non-DL methods")
    ax.legend(handles=[nondl_patch, dl_patch])
    
    save_fig(fig, "fig3_boxplot_rank_distribution")


def fig4_bar_family(df):
    """Bar chart: DL vs Non-DL average rank & normalized score."""
    family_data = df.groupby("method_family").agg(
        imp_avg_rank=("_rank", "mean"),
        imp_rank_std=("_rank", "std"),
        imp_avg_norm=("_norm", "mean"),
    ).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    colors = [NONDL_COLOR, DL_COLOR]
    families = family_data["method_family"].tolist()
    
    ax1.bar(families, family_data["imp_avg_rank"], yerr=family_data["imp_rank_std"],
            color=colors, capsize=5, alpha=0.8)
    ax1.set_ylabel("Average Rank")
    ax1.set_title("Average Imputation Rank")
    ax1.grid(axis="y", alpha=0.3)
    
    ax2.bar(families, family_data["imp_avg_norm"], color=colors, alpha=0.8)
    ax2.set_ylabel("Normalized Score (higher=better)")
    ax2.set_title("Average Normalized Imputation Score")
    ax2.grid(axis="y", alpha=0.3)
    
    fig.suptitle("DL vs Non-DL Method Families", fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, "fig4_bar_dl_vs_nondl")


def fig5_scatter_perf_runtime(df):
    """Scatter: imputation rank vs runtime."""
    method_agg = df.groupby("method").agg(
        imp_avg_rank=("_rank", "mean"),
        runtime_mean=("runtime_seconds", "mean"),
    ).reset_index()
    method_agg["family"] = method_agg["method"].apply(lambda m: "DL" if m in DL_METHODS else "Non-DL")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    for _, row in method_agg.iterrows():
        color = DL_COLOR if row["family"] == "DL" else NONDL_COLOR
        marker = "^" if row["family"] == "DL" else "o"
        ax.scatter(row["runtime_mean"], row["imp_avg_rank"],
                   color=color, marker=marker, s=120, edgecolors="black", linewidths=0.5, zorder=5)
        ax.annotate(row["method"], (row["runtime_mean"], row["imp_avg_rank"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    
    ax.set_xlabel("Mean Runtime (seconds)")
    ax.set_ylabel("Average Imputation Rank (lower=better)")
    ax.set_title("Performance vs Runtime Trade-off")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    
    dl_patch = mpatches.Patch(color=DL_COLOR, label="DL")
    nondl_patch = mpatches.Patch(color=NONDL_COLOR, label="Non-DL")
    ax.legend(handles=[nondl_patch, dl_patch])
    
    save_fig(fig, "fig6_scatter_perf_vs_runtime")


def fig6_stability(df):
    """Stability: mean ± std of RMSE across seeds for each method."""
    agg = df.groupby(["method", "dataset", "mechanism", "pattern", "ratio"])["imp_num_rmse"].agg(["mean", "std"]).reset_index()
    method_stab = agg.groupby("method").agg(
        rmse_mean=("mean", "mean"),
        rmse_std_mean=("std", "mean"),
    ).reset_index()
    method_stab["family"] = method_stab["method"].apply(lambda m: "DL" if m in DL_METHODS else "Non-DL")
    method_stab = method_stab.set_index("method").loc[METHOD_ORDER].reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(method_stab))
    colors = [METHOD_COLORS[m] for m in method_stab["method"]]
    ax.bar(x, method_stab["rmse_mean"], yerr=method_stab["rmse_std_mean"],
           color=colors, capsize=4, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(method_stab["method"], rotation=45, ha="right")
    ax.set_ylabel("Imputation RMSE (mean ± cross-seed std)")
    ax.set_title("Method Stability: Cross-Seed Variability")
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, "fig7_stability_rmse")


def fig7_degradation(df):
    """Degradation: performance change from 10% to 50% missing ratio."""
    low = df[df["ratio"] == 0.1].groupby("method")["_norm"].mean()
    high = df[df["ratio"] == 0.5].groupby("method")["_norm"].mean()
    
    degrad = pd.DataFrame({
        "method": low.index,
        "score_10pct": low.values,
        "score_50pct": high.reindex(low.index).values,
    })
    degrad["degradation"] = degrad["score_10pct"] - degrad["score_50pct"]
    degrad = degrad.set_index("method").loc[METHOD_ORDER].reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(degrad))
    width = 0.35
    colors10 = [METHOD_COLORS[m] for m in degrad["method"]]
    
    ax.bar([xi - width/2 for xi in x], degrad["score_10pct"], width,
           color=colors10, alpha=0.9, label="10% missing", edgecolor="black", linewidth=0.5)
    ax.bar([xi + width/2 for xi in x], degrad["score_50pct"], width,
           color=colors10, alpha=0.4, label="50% missing", edgecolor="black", linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(degrad["method"], rotation=45, ha="right")
    ax.set_ylabel("Normalized Score (higher=better)")
    ax.set_title("Performance Degradation: 10% → 50% Missing Ratio")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, "fig8_degradation_plot")


def fig8_heatmap_norm_score(df):
    """Heatmap: method × dataset normalized imputation score."""
    pivot = df.groupby(["dataset", "method"])["_norm"].mean().unstack("method")
    pivot = pivot[METHOD_ORDER].reindex(sorted(DATASETS))
    
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax,
                linewidths=0.5, vmin=0, vmax=1, cbar_kws={"label": "Normalized Score (higher=better)"})
    ax.set_title("Normalized Imputation Score: Method × Dataset")
    ax.set_xlabel("Method")
    ax.set_ylabel("Dataset")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    save_fig(fig, "fig1b_heatmap_method_dataset_normscore")


def fig9_mechanism_comparison(df):
    """Grouped bar: method ranks under each mechanism."""
    mech_agg = df.groupby(["method", "mechanism"])["_rank"].mean().reset_index()
    mech_pivot = mech_agg.pivot(index="method", columns="mechanism", values="_rank")
    mech_pivot = mech_pivot.loc[METHOD_ORDER]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(METHOD_ORDER))
    width = 0.25
    
    for i, mech in enumerate(MECHANISMS):
        offset = (i - 1) * width
        vals = mech_pivot[mech].values
        ax.bar(x + offset, vals, width, label=mech, alpha=0.8, edgecolor="black", linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_ORDER, rotation=45, ha="right")
    ax.set_ylabel("Average Rank (lower=better)")
    ax.set_title("Method Rankings by Missing Mechanism")
    ax.legend(title="Mechanism")
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, "fig9_mechanism_comparison")


def fig10_violin_rank(df):
    """Violin plot: rank distributions across all conditions."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    plot_df = df[["method", "_rank"]].copy()
    plot_df["method"] = pd.Categorical(plot_df["method"], categories=METHOD_ORDER, ordered=True)
    
    parts = ax.violinplot(
        [df[df["method"] == m]["_rank"].values for m in METHOD_ORDER],
        positions=range(len(METHOD_ORDER)),
        showmeans=True, showmedians=True
    )
    
    for i, pc in enumerate(parts["bodies"]):
        method = METHOD_ORDER[i]
        pc.set_facecolor(METHOD_COLORS[method])
        pc.set_alpha(0.7)
    
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels(METHOD_ORDER, rotation=45, ha="right")
    ax.set_ylabel("Rank (lower=better)")
    ax.set_title("Imputation Rank Distribution (Violin)")
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, "fig10_violin_rank")


def main():
    print("=" * 60)
    print("Step 6: Visualizations")
    print("=" * 60)

    df = load_data()
    df["ds_metric"] = df.apply(pick_downstream_metric, axis=1)
    df["method_family"] = df["method"].apply(lambda m: "DL" if m in DL_METHODS else "Non-DL")
    
    # Compute ranks and normalized scores
    df = compute_within_dataset_ranks(df, "imp_num_rmse", higher_is_better=False)
    df = compute_norm_score(df, "imp_num_rmse", higher_is_better=False)
    
    print("Generating figures...")
    fig1_heatmap_rank(df)
    fig8_heatmap_norm_score(df)
    fig2_line_ratio(df)
    fig3_boxplot_rank(df)
    fig4_bar_family(df)
    fig5_scatter_perf_runtime(df)
    fig6_stability(df)
    fig7_degradation(df)
    fig9_mechanism_comparison(df)
    fig10_violin_rank(df)
    
    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print("Step 6 complete.\n")


if __name__ == "__main__":
    main()
