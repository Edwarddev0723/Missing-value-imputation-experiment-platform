# Statistical Testing Report

## 1. Friedman Test (Global)

- Test statistic: 8769.7153


**Result: Significant differences exist among methods (p < 0.05). Proceeding to pairwise comparisons.**

## 2. Pairwise Wilcoxon Signed-Rank Tests
- Total comparisons: 78
- Significant after Holm correction (α=0.05): 76

### Top 20 Most Significant Differences

| method_1   | method_2   |   mean_rank_diff |   p_adjusted | significant   |   effect_size_r | winner     |
|:-----------|:-----------|-----------------:|-------------:|:--------------|----------------:|:-----------|
| missforest | tabcsdi    |          -9.5939 |            0 | True          |             inf | missforest |
| tabcsdi    | xgboost    |           8.9473 |            0 | True          |             inf | xgboost    |
| miracle    | xgboost    |           7.9466 |            0 | True          |             inf | xgboost    |
| miracle    | missforest |           8.5932 |            0 | True          |             inf | missforest |
| knn        | tabcsdi    |          -7.1689 |            0 | True          |             inf | knn        |
| median     | missforest |           7.1591 |            0 | True          |             inf | missforest |
| gain       | xgboost    |           5.9258 |            0 | True          |             inf | xgboost    |
| median     | xgboost    |           6.5125 |            0 | True          |             inf | xgboost    |
| gain       | missforest |           6.5723 |            0 | True          |             inf | missforest |
| em         | tabcsdi    |          -6.6061 |            0 | True          |             inf | em         |
| softimpute | xgboost    |           5.6977 |            0 | True          |             inf | xgboost    |
| knn        | miracle    |          -6.1682 |            0 | True          |             inf | knn        |
| mice       | tabcsdi    |          -6.1341 |            0 | True          |             inf | mice       |
| missforest | softimpute |          -6.3443 |            0 | True          |             inf | missforest |
| mice       | miracle    |          -5.1333 |            0 | True          |             inf | mice       |
| miwae      | xgboost    |           4.9833 |            0 | True          |             inf | xgboost    |
| mida       | missforest |           5.6008 |            0 | True          |             inf | missforest |
| mean       | tabcsdi    |          -4.4242 |            0 | True          |             inf | mean       |
| missforest | miwae      |          -5.6299 |            0 | True          |             inf | missforest |
| mida       | xgboost    |           4.9542 |            0 | True          |             inf | xgboost    |

## 3. Key Comparisons

| comparison            | winner     |   mean_rank_diff |   p_adjusted | significant   |   effect_size_r |
|:----------------------|:-----------|-----------------:|-------------:|:--------------|----------------:|
| missforest vs xgboost | missforest |           0.6466 | 9.28e-33     | True          |             inf |
| gain vs tabcsdi       | gain       |           3.0216 | 1.34961e-135 | True          |             inf |
| gain vs miracle       | gain       |           2.0208 | 5.45426e-65  | True          |             inf |
| tabcsdi vs miracle    | miracle    |           1.0008 | 5.72492e-16  | True          |             inf |

## 4. Baseline Gap Analysis
How much do advanced methods improve over simple baselines (mean/median)?

| baseline   | advanced   | advanced_family   |   baseline_avg_rank |   advanced_avg_rank |   rank_improvement |   win_pct |   n_conditions |
|:-----------|:-----------|:------------------|--------------------:|--------------------:|-------------------:|----------:|---------------:|
| median     | missforest | Non-DL            |              9.222  |              2.0629 |             7.1591 |      94.1 |           1320 |
| median     | xgboost    | Non-DL            |              9.222  |              2.7095 |             6.5125 |      95.8 |           1320 |
| mean       | missforest | Non-DL            |              7.2326 |              2.0629 |             5.1697 |      92.7 |           1320 |
| median     | knn        | Non-DL            |              9.222  |              4.4879 |             4.7341 |      84.3 |           1320 |
| mean       | xgboost    | Non-DL            |              7.2326 |              2.7095 |             4.5231 |      89.8 |           1320 |
| median     | em         | Non-DL            |              9.222  |              5.0508 |             4.1712 |      83.5 |           1320 |
| median     | mice       | Non-DL            |              9.222  |              5.5227 |             3.6992 |      78.3 |           1320 |
| mean       | knn        | Non-DL            |              7.2326 |              4.4879 |             2.7447 |      78.9 |           1320 |
| mean       | em         | Non-DL            |              7.2326 |              5.0508 |             2.1818 |      77.3 |           1320 |
| mean       | mice       | Non-DL            |              7.2326 |              5.5227 |             1.7098 |      71.6 |           1320 |
| median     | mida       | DL                |              9.222  |              7.6636 |             1.5583 |      62.6 |           1320 |
| median     | miwae      | DL                |              9.222  |              7.6928 |             1.5292 |      64.3 |           1320 |
| median     | softimpute | Non-DL            |              9.222  |              8.4072 |             0.8148 |      60.1 |           1320 |
| median     | gain       | DL                |              9.222  |              8.6352 |             0.5867 |      61.8 |           1320 |
| mean       | mida       | DL                |              7.2326 |              7.6636 |            -0.4311 |      48.9 |           1320 |
| mean       | miwae      | DL                |              7.2326 |              7.6928 |            -0.4602 |      48.4 |           1320 |
| mean       | softimpute | Non-DL            |              7.2326 |              8.4072 |            -1.1746 |      39.7 |           1320 |
| mean       | gain       | DL                |              7.2326 |              8.6352 |            -1.4027 |      39.8 |           1320 |
| median     | miracle    | DL                |              9.222  |             10.6561 |            -1.4341 |      35.2 |           1320 |
| median     | tabcsdi    | DL                |              9.222  |             11.6568 |            -2.4348 |      17.3 |           1320 |

## Methodology Notes
- Ranks computed within each (dataset, mechanism, pattern, ratio, seed) condition
- Friedman test checks overall differences across all methods
- Wilcoxon signed-rank test for pairwise comparisons (non-parametric, paired)
- Holm-Bonferroni correction for multiple comparisons
- Effect size r = Z / sqrt(N), interpreted as: small (0.1), medium (0.3), large (0.5)