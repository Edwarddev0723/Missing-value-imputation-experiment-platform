# Data Validation Report

## Completeness Check
- Expected unique combinations: 17160
- Actual unique combinations: 17160
- Total file records loaded: 17160
- Missing combinations: 0
- Extra combinations: 0
- Duplicate combinations: 0

**✓ All validation checks passed. Dataset is complete with no duplicates.**

## Metric Coverage

| Metric | Available | Missing | Coverage % |
|--------|-----------|---------|------------|
| imp_num_rmse | 17160 | 0 | 100.0% |
| imp_num_mae | 17160 | 0 | 100.0% |
| imp_cat_accuracy | 3120 | 14040 | 18.18% |
| imp_cat_macro_f1 | 3120 | 14040 | 18.18% |
| ds_accuracy | 13439 | 3721 | 78.32% |
| ds_f1_weighted | 13439 | 3721 | 78.32% |
| ds_f1_macro | 13439 | 3721 | 78.32% |
| ds_roc_auc | 9360 | 7800 | 54.55% |
| ds_rmse | 1560 | 15600 | 9.09% |
| ds_mae | 1560 | 15600 | 9.09% |
| ds_r2 | 1560 | 15600 | 9.09% |
| dist_wasserstein_mean | 17160 | 0 | 100.0% |
| dist_jsd_mean | 6240 | 10920 | 36.36% |
| runtime_seconds | 17160 | 0 | 100.0% |

## Per-Method Completeness

| Method | Records | Expected | Complete |
|--------|---------|----------|----------|
| em | 1320 | 1320 | ✓ |
| gain | 1320 | 1320 | ✓ |
| knn | 1320 | 1320 | ✓ |
| mean | 1320 | 1320 | ✓ |
| median | 1320 | 1320 | ✓ |
| mice | 1320 | 1320 | ✓ |
| mida | 1320 | 1320 | ✓ |
| miracle | 1320 | 1320 | ✓ |
| missforest | 1320 | 1320 | ✓ |
| miwae | 1320 | 1320 | ✓ |
| softimpute | 1320 | 1320 | ✓ |
| tabcsdi | 1320 | 1320 | ✓ |
| xgboost | 1320 | 1320 | ✓ |

## Per-Dataset Completeness

| Dataset | Records | Expected | Complete |
|---------|---------|----------|----------|
| adult | 1560 | 1560 | ✓ |
| autism | 1560 | 1560 | ✓ |
| bank | 1560 | 1560 | ✓ |
| breast_cancer | 1560 | 1560 | ✓ |
| concrete | 1560 | 1560 | ✓ |
| letter | 1560 | 1560 | ✓ |
| mammographic | 1560 | 1560 | ✓ |
| spambase | 1560 | 1560 | ✓ |
| titanic | 1560 | 1560 | ✓ |
| wine | 1560 | 1560 | ✓ |
| yeast | 1560 | 1560 | ✓ |