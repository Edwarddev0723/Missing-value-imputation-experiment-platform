# Schema Report

## Data Structure
- Total records: 17160
- Expected records: 17160
- Unique datasets: 11 — ['adult', 'autism', 'bank', 'breast_cancer', 'concrete', 'letter', 'mammographic', 'spambase', 'titanic', 'wine', 'yeast']
- Unique methods: 13 — ['em', 'gain', 'knn', 'mean', 'median', 'mice', 'mida', 'miracle', 'missforest', 'miwae', 'softimpute', 'tabcsdi', 'xgboost']
- Unique mechanisms: 3 — ['MAR', 'MCAR', 'MNAR']
- Unique patterns: 2 — ['general', 'univariate']
- Unique rates: [np.float64(0.1), np.float64(0.2), np.float64(0.3), np.float64(0.5)]
- Unique seeds: [np.int64(42), np.int64(123), np.int64(456), np.int64(789), np.int64(1024)]

## Column Schema

| Column | Dtype | Non-Null Count | Null Count | Coverage % |
|--------|-------|----------------|------------|------------|
| dataset | str | 17160 | 0 | 100.0% |
| dataset_id | str | 17160 | 0 | 100.0% |
| method | str | 17160 | 0 | 100.0% |
| mechanism | str | 17160 | 0 | 100.0% |
| pattern | str | 17160 | 0 | 100.0% |
| ratio | float64 | 17160 | 0 | 100.0% |
| seed | int64 | 17160 | 0 | 100.0% |
| n_samples | int64 | 17160 | 0 | 100.0% |
| n_features | int64 | 17160 | 0 | 100.0% |
| runtime_seconds | float64 | 17160 | 0 | 100.0% |
| imp_num_rmse | float64 | 17160 | 0 | 100.0% |
| imp_num_mae | float64 | 17160 | 0 | 100.0% |
| imp_cat_accuracy | float64 | 3120 | 14040 | 18.2% |
| imp_cat_macro_f1 | float64 | 3120 | 14040 | 18.2% |
| task_type | str | 17160 | 0 | 100.0% |
| downstream_model | str | 17160 | 0 | 100.0% |
| ds_accuracy | float64 | 13439 | 3721 | 78.3% |
| ds_f1_weighted | float64 | 13439 | 3721 | 78.3% |
| ds_f1_macro | float64 | 13439 | 3721 | 78.3% |
| ds_roc_auc | float64 | 9360 | 7800 | 54.5% |
| ds_rmse | float64 | 1560 | 15600 | 9.1% |
| ds_mae | float64 | 1560 | 15600 | 9.1% |
| ds_r2 | float64 | 1560 | 15600 | 9.1% |
| dist_wasserstein_mean | float64 | 17160 | 0 | 100.0% |
| dist_jsd_mean | float64 | 6240 | 10920 | 36.4% |
| _source_file | str | 17160 | 0 | 100.0% |
| method_family | str | 17160 | 0 | 100.0% |
| ratio_pct | int64 | 17160 | 0 | 100.0% |
| setting | str | 17160 | 0 | 100.0% |

## Nested JSON Structure (flattened)
```
root
├── dataset_name -> dataset
├── method_name -> method
├── mechanism
├── pattern
├── missing_rate -> ratio
├── seed
├── n_samples, n_features
├── runtime_seconds
├── imputation_quality
│   ├── numerical.rmse -> imp_num_rmse
│   ├── numerical.mae -> imp_num_mae
│   ├── categorical.accuracy -> imp_cat_accuracy
│   └── categorical.macro_f1 -> imp_cat_macro_f1
├── downstream_performance
│   ├── task_type
│   ├── model -> downstream_model
│   └── metrics
│       ├── accuracy -> ds_accuracy
│       ├── f1_weighted -> ds_f1_weighted
│       ├── f1_macro -> ds_f1_macro
│       ├── roc_auc -> ds_roc_auc
│       ├── rmse -> ds_rmse
│       ├── mae -> ds_mae
│       └── r2 -> ds_r2
└── distribution_preservation
    ├── wasserstein_distance_mean -> dist_wasserstein_mean
    └── jsd_mean -> dist_jsd_mean
```

## Notes
- Categorical imputation quality is null for pure-numerical datasets
- Classification datasets have ds_accuracy/f1/roc_auc; regression datasets have ds_rmse/mae/r2
- Some downstream metrics may be null (e.g., roc_auc for multi-class without probabilities)
- JSD is null for datasets without categorical features