# Results Section Draft

## 4.1 Overall Comparison of Imputation Methods

We evaluated 13 imputation methods (8 non-deep-learning and 5 deep-learning-based) across 11 datasets under 24 missing data configurations with 5 random seeds, yielding 17,160 experimental runs. To enable fair cross-dataset comparison, we employed within-condition ranking: for each unique combination of (dataset, mechanism, pattern, ratio, seed), methods were ranked from 1 (best) to 13 (worst) based on imputation RMSE. We report average ranks across all conditions.

Table 1 presents the overall results. **missforest** achieved the best average imputation rank (2.06), followed by xgboost and knn. The worst-performing methods were median, miracle, tabcsdi.

A Friedman test confirmed statistically significant differences among methods. Subsequent pairwise Wilcoxon signed-rank tests with Holm correction identified 76 out of 78 pairwise comparisons as significant (α = 0.05).

## 4.2 Deep Learning vs. Non-Deep-Learning Methods

Non-DL methods achieved a lower (better) average rank (5.59) compared to DL methods (9.26). Moreover, non-DL methods were substantially faster, with a mean runtime of 21.01s versus 11.59s for DL methods — a factor of 1× difference.

## 4.3 Effect of Missing Data Mechanism

Table 3 shows method rankings stratified by missing mechanism (MCAR, MAR, MNAR). All methods exhibited higher (worse) ranks under MNAR compared to MCAR and MAR, confirming that non-random missingness poses the greatest challenge for imputation.

## 4.4 Effect of Missing Rate

Performance consistently degraded as the missing rate increased from 10% to 50%. However, the relative ordering of methods remained relatively stable, suggesting that the best methods at low missing rates remain competitive at high rates.

## 4.5 Dataset-Specific Analysis

The best method varied across datasets, indicating that no single method universally dominates. This underscores the importance of method selection tailored to dataset characteristics. Pure numerical datasets and mixed-type datasets showed different method preference patterns.

## 4.6 Practical Trade-offs

When considering the performance-runtime trade-off, tree-based methods (missforest, xgboost) offered strong imputation quality with moderate computational cost. Simple baselines (mean, median) provided fast imputation with acceptable quality at low missing rates, making them suitable for time-constrained applications. DL methods generally required substantially more computation time without consistently superior imputation quality.