# Key Findings

## 1. Overall Best Methods
- **Best overall method by imputation rank: missforest** (avg rank: 2.06)
- Top-3 overall: missforest, xgboost, knn
- Worst overall method: tabcsdi (avg rank: 11.66)

## 2. DL vs Non-DL Comparison
- DL: Imp avg rank = 9.26, DS avg rank = 7.68, Runtime mean = 11.59s
- Non-DL: Imp avg rank = 5.59, DS avg rank = 6.57, Runtime mean = 21.01s
- **Non-DL methods collectively outperform DL methods** (avg rank 5.59 vs 9.26)

## 3. Stability
- Most stable method (lowest rank std): **xgboost** (std = 1.75)
- Least stable method: **em** (std = 2.94)

## 4. High Missing Rate Resilience
- Top-3 at 50% missing: missforest, xgboost, knn

## 5. MNAR Sensitivity
- Most sensitive to MNAR: **mean** (rank degradation: 1.30)
- Least sensitive to MNAR: **softimpute** (rank degradation: -1.11)

## 6. Dataset-Dependent Winners
- Number of distinct best methods across 11 datasets: 3
  - missforest: best on 9 dataset(s) (bank, breast_cancer, concrete, letter, mammographic, spambase, titanic, wine, yeast)
  - xgboost: best on 1 dataset(s) (adult)
  - mice: best on 1 dataset(s) (autism)

## 7. Baseline Gap
- Average rank improvement over mean imputation: 0.46
- Biggest improvement over mean: **missforest** (rank improvement: 5.17)
- Methods with negligible improvement over mean baseline: mida, miwae, softimpute, gain, miracle, tabcsdi

## 8. Degradation Under High Missing Rates
- Least degraded (10%→50%): **tabcsdi** (-19.7% RMSE increase)
- Most degraded: **xgboost** (12.0% RMSE increase)

## 9. Mechanism Impact
- Best method under MCAR: **missforest** (rank: 2.04)
- Best method under MNAR: **missforest** (rank: 1.65)

## 10. Runtime Efficiency
- Fastest: **mean** (0.0023s mean)
- Slowest: **missforest** (79.18s mean)
- Speed ratio (slowest/fastest): 34428x
