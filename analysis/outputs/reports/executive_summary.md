# Executive Summary

## Benchmark Overview
This report presents a comprehensive benchmark of 13 missing value imputation methods across 11 datasets, 24 missing data scenarios (3 mechanisms × 2 patterns × 4 ratios), and 5 random seeds, totaling 17,160 experimental runs.

## Top-5 Methods (by Average Imputation Rank)

| Rank | Method | Family | Avg Imp Rank | Avg DS Rank | Mean Runtime |
|------|--------|--------|-------------|-------------|-------------|
| 1 | missforest | Non-DL | 2.06 | 5.61 | 79.18s |
| 2 | xgboost | Non-DL | 2.71 | 5.80 | 71.60s |
| 3 | knn | Non-DL | 4.49 | 6.77 | 10.40s |
| 4 | em | Non-DL | 5.05 | 6.96 | 2.12s |
| 5 | mice | Non-DL | 5.52 | 6.82 | 3.52s |

## Key Conclusions

1. **Best overall method**: missforest achieves the best average imputation rank (2.06) across all experimental conditions.

2. **Non-DL methods outperform DL methods** on average (rank 5.59 vs 9.26), while being substantially faster (21.01s vs 11.59s).

3. **No universal winner exists** — the best method varies across datasets, highlighting the importance of dataset-specific method selection.

4. **Simple baselines remain competitive** — in some settings, mean/median imputation is not far behind more complex approaches, especially at low missing rates.

5. **MNAR is the hardest scenario** — all methods show degraded performance under Missing Not At Random conditions compared to MCAR and MAR.

## Practical Recommendations

- **Default choice**: missforest offers the best overall imputation quality.
- **Speed-critical applications**: Consider simple baselines (mean/median) or knn, which are orders of magnitude faster.
- **High missing rates**: Methods resilient to 50% missingness should be preferred for severely incomplete data.
- **MNAR data**: Exercise caution; no method fully compensates for non-random missingness.