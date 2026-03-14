# Missing Value Imputation Benchmark — Experiment Specification

> **目的**：針對統計 / 機器學習 / 深度學習三大類共 14 種 missing value imputation 方法，在 11 個 tabular datasets 上進行系統性比較實驗，結果將寫入綜述論文的實驗章節。
>
> **硬體環境**：MacBook Pro M4 Pro, 48 GB RAM（無 CUDA，深度學習方法使用 MPS 或 CPU）

---

## 1. Methods（14 種方法）

### 1.1 Statistical Methods（4 種）

| ID | Method | 說明 | 適用變數類型 |
|----|--------|------|-------------|
| `S1` | **Mean Imputation** | 以該欄位的算術平均值填補 | Numerical only |
| `S2` | **Median Imputation** | 以該欄位的中位數填補 | Numerical only |
| `S3` | **Expectation-Maximization (EM)** | 迭代式最大期望演算法 | Numerical only |
| `S4` | **MICE (Multiple Imputation by Chained Equations)** | 多重鏈式方程填補，每個變數以條件模型依序填補並迭代 | Numerical + Categorical |

> **注意**：S1、S2、S3 僅適用於數值型變數。在 Mixed datasets 上，類別型變數需使用 Mode Imputation 作為 fallback，並在論文中明確說明。

### 1.2 Machine Learning Methods（4 種）

| ID | Method | 說明 | 適用變數類型 |
|----|--------|------|-------------|
| `M1` | **KNN Imputation** | 基於 K-Nearest Neighbors 的加權填補 | Numerical + Categorical |
| `M2` | **MissForest** | 基於 Random Forest 的迭代式填補 | Numerical + Categorical |
| `M3` | **SoftImpute** | 基於矩陣補全的核範數正則化方法 | Numerical only |
| `M4` | **XGBoost Imputation** | 以 XGBoost 為 base learner 的迭代式填補 | Numerical + Categorical |

> **注意**：S3 和 M3 僅適用於數值型變數。在 Mixed datasets 上需先將類別變數 one-hot encoding 後再進行填補，填補完畢後再轉回原始類別。

### 1.3 Deep Learning Methods（5 種）

| ID | Method | 簡稱 | 論文 | 適用變數類型 |
|----|--------|------|------|-------------|
| `D1` | **Missing Data Imputation Denoising Autoencoder** | MIDA | Gondara & Wang, 2018 | Numerical + Categorical |
| `D2` | **Missing data Importance-Weighted Autoencoder** | MIWAE | Mattei & Friston, 2019 | Numerical (原始設計) |
| `D3` | **Generative Adversarial Imputation Nets** | GAIN | Yoon et al., 2018 | Numerical + Categorical |
| `D4` | **Conditional Score-based Diffusion Models for Tabular data** | TabCSDI | Zheng & Charoenphakdee, 2022 | Numerical + Categorical |
| `D5` | **Missing data Imputation Refinement And Causal LEarning** | MIRACLE | Kyono et al., 2021 | Numerical + Categorical |

---

## 2. Datasets（11 個）

### 2.1 Pure Numerical Datasets（6 個）

| ID | Dataset | Features | Samples | Source | URL | 備註 |
|----|---------|----------|---------|--------|-----|------|
| `PN1` | Wine Quality (White) | 12 | 4,898 | UCI | https://archive.ics.uci.edu/dataset/186/wine+quality | 中型 benchmark |
| `PN2` | Concrete Compressive Strength | 8 | 1,030 | UCI | https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength | 小型 regression dataset |
| `PN3` | Spambase | 57 | 4,601 | UCI | https://archive.ics.uci.edu/dataset/94/spambase | 高維度測試 |
| `PN4` | Yeast | 8 | 1,484 | UCI | https://archive.ics.uci.edu/dataset/110/yeast | feature 結構簡單 |
| `PN5` | Breast Cancer Wisconsin (Original) | 9 | 699 | UCI | https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original | 經典小型 dataset |
| `PN6` | Letter Recognition | 16 | 20,000 | UCI | https://archive.ics.uci.edu/dataset/59/letter+recognition | 大型 dataset |

### 2.2 Mixed (Numerical + Categorical) Datasets（5 個）

| ID | Dataset | Categorical | Numerical | Samples | Source | URL | 備註 |
|----|---------|-------------|-----------|---------|--------|-----|------|
| `MX1` | Adult (Census Income) | 9 | 6 | 48,842 | UCI | https://archive.ics.uci.edu/dataset/2/adult | 大型 mixed |
| `MX2` | Bank Marketing | 9 | 7 | 45,211 | UCI | https://archive.ics.uci.edu/dataset/222/bank+marketing | 大型 mixed |
| `MX3` | Titanic | 5 | 5 | 891 | Kaggle | https://www.kaggle.com/c/titanic | 小型 mixed |
| `MX4` | Mammographic Masses | 3 | 2 | 961 | UCI | https://archive.ics.uci.edu/dataset/161/mammographic+mass | 小型 mixed |
| `MX5` | Autism (ASD Screening) | 15 | 6 | 704 | UCI | https://archive.ics.uci.edu/dataset/419/autism+screening+adult | categorical 佔比高 |

### 2.3 資料前處理規則

1. **移除原始缺失值**：所有 dataset 先刪除含有天然缺失值的 rows，得到完整子集作為 ground truth。
2. **Breast Cancer Wisconsin**：原始資料有 16 筆缺失值，需先移除。
3. **Titanic**：Age、Cabin、Embarked 欄位有天然缺失，需先移除含缺失的 rows。
4. **標準化**：數值型欄位進行 Min-Max Scaling 到 [0, 1]（在製造缺失之前），以確保 RMSE 跨 dataset 可比較。類別型欄位進行 Label Encoding。
5. **Train/Test Split**：先做 80/20 split，imputation 僅在 training set 上 fit，test set 上 transform（避免 data leakage）。

---

## 3. Missing Data Simulation

### 3.1 Missing Mechanisms（3 種）

#### MCAR (Missing Completely At Random)
- 每個值以固定機率 `p = missing_rate` 被獨立隨機遮罩
- 實作：對每個元素以 Bernoulli(p) 決定是否設為 NaN

```python
mask = np.random.binomial(1, missing_rate, size=data.shape).astype(bool)
data_missing = data.copy()
data_missing[mask] = np.nan
```

#### MAR (Missing At Random)
- 缺失與否取決於其他已觀測變數的值
- 實作方式：選定一個「決定變數」X_j，根據 X_j 的分位數設定「目標變數」X_k 的缺失機率

```python
# 對目標欄位 X_k，依據觀測欄位 X_j 的值決定缺失機率
# X_j 值越高 → X_k 缺失機率越高（使用 logistic function）
def generate_mar_mask(data, target_col, observed_col, missing_rate):
    """
    用 observed_col 的值透過 logistic function 產生 target_col 的缺失機率。
    校準 logistic 的 intercept 使得整體缺失比例 ≈ missing_rate。
    """
    x = data[:, observed_col]
    x_standardized = (x - x.mean()) / (x.std() + 1e-8)
    # 用 binary search 校準 intercept 使整體缺失率 ≈ missing_rate
    # prob = sigmoid(slope * x_standardized + intercept)
    # slope 固定為 2.0，調整 intercept
    ...
```

#### MNAR (Missing Not At Random)
- 缺失與否取決於該欄位自身的值
- 實作方式：該欄位值越大（或越小），缺失機率越高

```python
# 對目標欄位 X_k，其自身值越高 → 缺失機率越高
def generate_mnar_mask(data, target_col, missing_rate):
    """
    用 target_col 自身的值透過 logistic function 產生缺失機率。
    校準方式同 MAR。
    """
    x = data[:, target_col]
    x_standardized = (x - x.mean()) / (x.std() + 1e-8)
    ...
```

### 3.2 Missing Patterns（2 種）

#### Univariate Missing Pattern
- **僅一個欄位**被施加缺失值，其餘欄位完整
- 選擇哪個欄位：選擇該 dataset 中 variance 最大的數值欄位（或對於 mixed dataset，分別測試一個數值欄位和一個類別欄位）
- 用途：最簡單的場景，作為 sanity check

#### General Missing Pattern
- **所有欄位**都可能出現缺失值
- 每個欄位獨立按照指定的 missing mechanism 和 missing rate 施加缺失
- 約束條件：確保每筆 row 至少有一個欄位是完整的（避免全空 row）
- 用途：最接近真實場景

### 3.3 Missing Rates（4 種）

```python
MISSING_RATES = [0.10, 0.20, 0.30, 0.50]
```

### 3.4 Random Seeds

```python
SEEDS = [42, 123, 456, 789, 1024]
```

每組 (dataset, method, mechanism, pattern, rate) 的設定使用以上 5 個 seed 重複實驗，報告 mean ± std。

---

## 4. Evaluation Metrics

### 4.1 Imputation Quality（直接衡量填補精度）

#### 數值型變數

| Metric | 公式 | 說明 |
|--------|------|------|
| **RMSE** | `sqrt(mean((x_true - x_imputed)²))` | 僅計算被遮罩位置的值，Root Mean Squared Error |
| **MAE** | `mean(abs(x_true - x_imputed))` | 僅計算被遮罩位置的值，Mean Absolute Error |

```python
# 僅在 missing mask 為 True 的位置計算
rmse = np.sqrt(np.mean((X_true[mask] - X_imputed[mask]) ** 2))
mae = np.mean(np.abs(X_true[mask] - X_imputed[mask]))
```

#### 類別型變數（僅 Mixed datasets）

| Metric | 說明 |
|--------|------|
| **Accuracy** | 被遮罩位置的類別填補正確率 |
| **Macro-F1** | 被遮罩位置的 Macro-F1 Score（處理類別不平衡） |

```python
from sklearn.metrics import accuracy_score, f1_score
accuracy = accuracy_score(X_true_cat[mask_cat], X_imputed_cat[mask_cat])
macro_f1 = f1_score(X_true_cat[mask_cat], X_imputed_cat[mask_cat], average='macro')
```

### 4.2 Downstream Task Performance（間接衡量填補品質）

在 imputed data 上訓練下游分類 / 回歸模型，評估預測效能。

#### 下游模型
- 統一使用 **Random Forest** 作為下游模型（避免下游模型選擇影響比較公平性）
- 超參數：`n_estimators=100, random_state=seed`

#### 下游任務與指標

| Dataset | Task Type | Target Variable | Downstream Metrics |
|---------|-----------|----------------|-------------------|
| `PN1` Wine Quality | Regression | quality | R², RMSE |
| `PN2` Concrete | Regression | compressive_strength | R², RMSE |
| `PN3` Spambase | Binary Classification | is_spam | Accuracy, AUC-ROC |
| `PN4` Yeast | Multi-class Classification | localization_site | Accuracy, Macro-F1 |
| `PN5` Breast Cancer | Binary Classification | class | Accuracy, AUC-ROC |
| `PN6` Letter Recognition | Multi-class Classification | letter | Accuracy, Macro-F1 |
| `MX1` Adult | Binary Classification | income >50K | Accuracy, AUC-ROC |
| `MX2` Bank Marketing | Binary Classification | deposit (y) | Accuracy, AUC-ROC |
| `MX3` Titanic | Binary Classification | survived | Accuracy, AUC-ROC |
| `MX4` Mammographic | Binary Classification | severity | Accuracy, AUC-ROC |
| `MX5` Autism | Binary Classification | ASD_class | Accuracy, AUC-ROC |

#### 下游 Baseline
- **Oracle Baseline**：使用完整資料（無缺失）訓練的下游模型表現，作為上界參考
- **Drop Rows Baseline**：直接刪除含缺失 rows 後訓練的下游模型表現

### 4.3 Distribution Preservation（分佈保真度）

衡量 imputed 資料的分佈是否與原始完整資料的分佈一致。

#### 數值型變數

| Metric | 說明 |
|--------|------|
| **Wasserstein Distance** | 衡量兩分佈之間的「搬土距離」，值越小表示分佈越接近 |

```python
from scipy.stats import wasserstein_distance
# 對每個數值欄位分別計算，最後取平均
wd = np.mean([
    wasserstein_distance(X_true[:, j], X_imputed[:, j])
    for j in numerical_columns
])
```

#### 類別型變數（僅 Mixed datasets）

| Metric | 說明 |
|--------|------|
| **Jensen-Shannon Divergence (JSD)** | KL Divergence 的對稱版本，衡量兩個類別分佈的差異，值域 [0, 1] |

```python
from scipy.spatial.distance import jensenshannon
# 對每個類別欄位計算分佈後取 JSD，最後取平均
jsd = np.mean([
    jensenshannon(
        np.bincount(X_true_cat[:, j], minlength=n_classes),
        np.bincount(X_imputed_cat[:, j], minlength=n_classes)
    )
    for j in categorical_columns
])
```

---

## 5. Experiment Matrix

### 5.1 完整實驗矩陣

```
14 methods × 11 datasets × 3 mechanisms × 2 patterns × 4 rates × 5 seeds
= 14 × 11 × 3 × 2 × 4 × 5
= 18,480 experimental runs
```

### 5.2 執行順序建議

為了儘早發現 bug 並逐步驗證，建議以下執行順序：

```
Phase 1 — Smoke Test（快速驗證 pipeline 正確性）
  └─ Dataset: PN5 (Breast Cancer, 最小)
  └─ Mechanism: MCAR only
  └─ Pattern: General only
  └─ Rate: 20% only
  └─ Seed: 42 only
  └─ Methods: Mean, KNN, GAIN (各類各一)
  └─ 預期 runs: 3，預期時間: < 10 分鐘

Phase 2 — 單一 Dataset 全方法驗證
  └─ Dataset: PN5 (Breast Cancer)
  └─ 所有 mechanisms × patterns × rates × seeds
  └─ 所有 14 methods
  └─ 預期 runs: 14 × 3 × 2 × 4 × 5 = 1,680
  └─ 目的：確認所有方法在小 dataset 上能正常跑完

Phase 3 — Pure Numerical Datasets 全量
  └─ Datasets: PN1–PN6
  └─ 所有設定
  └─ 預期 runs: 14 × 6 × 3 × 2 × 4 × 5 = 10,080

Phase 4 — Mixed Datasets 全量
  └─ Datasets: MX1–MX5
  └─ 所有設定
  └─ 預期 runs: 14 × 5 × 3 × 2 × 4 × 5 = 8,400
```

### 5.3 效能預估與注意事項

| Method | 預估每次 run 時間 (小型 dataset) | 預估每次 run 時間 (大型 dataset, Adult 48K) | 備註 |
|--------|-------------------------------|------------------------------------------|------|
| Mean/Median | < 1 秒 | < 1 秒 | — |
| EM | < 5 秒 | < 30 秒 | — |
| MICE | 10–30 秒 | 5–15 分鐘 | iterations=10 |
| KNN | 5–10 秒 | 10–30 分鐘 | O(n²) pairwise distance |
| MissForest | 10–60 秒 | 10–30 分鐘 | iterations=10 |
| SoftImpute | 5–10 秒 | 1–5 分鐘 | — |
| XGBoost | 5–15 秒 | 5–15 分鐘 | — |
| MIDA | 30–120 秒 | 5–20 分鐘 | MPS/CPU |
| MIWAE | 30–120 秒 | 5–20 分鐘 | MPS/CPU |
| GAIN | 60–180 秒 | 10–30 分鐘 | MPS/CPU |
| TabCSDI | 60–300 秒 | 15–60 分鐘 | Diffusion model, 較慢 |
| MIRACLE | 60–180 秒 | 10–30 分鐘 | MPS/CPU |

> **預估總時間**：約 3–7 天（取決於平行處理策略）

---

## 6. Hyperparameter Settings

為確保公平比較，所有方法使用以下統一或建議的超參數設定。

### 6.1 Statistical Methods

```yaml
Mean:
  # 無超參數

Median:
  # 無超參數

EM:
  max_iterations: 100
  convergence_threshold: 1e-4

MICE:
  max_iterations: 10
  n_imputations: 5       # 多重填補取平均
  estimator: "BayesianRidge"  # sklearn default
```

### 6.2 Machine Learning Methods

```yaml
KNN:
  n_neighbors: 5
  weights: "distance"

MissForest:
  n_estimators: 100
  max_iterations: 10
  convergence_threshold: 1e-4

SoftImpute:
  max_rank: min(n_features, 50)
  max_iterations: 100
  convergence_threshold: 1e-4

XGBoost:
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
  max_iterations: 10      # 迭代填補次數
```

### 6.3 Deep Learning Methods

```yaml
# 所有 DL 方法共通設定
common:
  device: "mps"            # Apple Silicon; fallback to "cpu"
  batch_size: 256          # 大型 dataset 可調到 512
  early_stopping_patience: 10

MIDA:
  hidden_layers: [256, 128, 256]
  epochs: 200
  learning_rate: 1e-3
  corruption_rate: 0.5
  optimizer: "Adam"

MIWAE:
  latent_dim: 32
  n_hidden: 128
  n_samples: 20            # importance weight samples
  epochs: 200
  learning_rate: 1e-3
  optimizer: "Adam"

GAIN:
  generator_hidden: [256, 128]
  discriminator_hidden: [256, 128]
  epochs: 300
  learning_rate: 1e-3
  alpha: 100               # hint rate
  optimizer: "Adam"

TabCSDI:
  diffusion_steps: 50
  hidden_dim: 128
  n_layers: 4
  epochs: 200
  learning_rate: 1e-3
  beta_schedule: "linear"
  # 類別變數編碼：one-hot encoding (主實驗)
  # 額外消融實驗可測試 analog bits 和 feature tokenization
  optimizer: "Adam"

MIRACLE:
  hidden_dim: 128
  n_layers: 3
  epochs: 200
  learning_rate: 1e-3
  lambda_reg: 1.0          # causal regularization weight
  optimizer: "Adam"
```

---

## 7. Output Format

### 7.1 結果儲存結構

```
results/
├── raw/
│   └── {dataset_id}_{method_id}_{mechanism}_{pattern}_{rate}_{seed}.json
├── aggregated/
│   ├── imputation_quality.csv      # RMSE, MAE, Accuracy, Macro-F1
│   ├── downstream_performance.csv  # R², RMSE, Accuracy, AUC-ROC, Macro-F1
│   └── distribution_preservation.csv  # Wasserstein Distance, JSD
└── figures/
    ├── heatmap_rmse_by_method_dataset.png
    ├── lineplot_rmse_vs_missingrate.png
    └── ...
```

### 7.2 Raw JSON 格式

每次 run 輸出一個 JSON 檔案：

```json
{
  "dataset_id": "PN1",
  "dataset_name": "Wine Quality (White)",
  "method_id": "D3",
  "method_name": "GAIN",
  "mechanism": "MCAR",
  "pattern": "general",
  "missing_rate": 0.2,
  "seed": 42,
  "n_samples": 4898,
  "n_features": 12,
  "runtime_seconds": 125.3,
  "imputation_quality": {
    "numerical": {
      "rmse": 0.0832,
      "mae": 0.0614
    },
    "categorical": {
      "accuracy": null,
      "macro_f1": null
    }
  },
  "downstream_performance": {
    "task_type": "regression",
    "model": "RandomForest",
    "metrics": {
      "r2": 0.421,
      "rmse": 0.753
    }
  },
  "distribution_preservation": {
    "wasserstein_distance_mean": 0.0123,
    "jsd_mean": null
  }
}
```

### 7.3 Aggregated CSV 格式

匯總後的 CSV 每行代表一組 `(dataset, method, mechanism, pattern, rate)` 的 5 次 seed 統計：

```csv
dataset_id,method_id,mechanism,pattern,rate,rmse_mean,rmse_std,mae_mean,mae_std,cat_acc_mean,cat_acc_std,cat_f1_mean,cat_f1_std,downstream_metric1_mean,downstream_metric1_std,...,wd_mean,wd_std,jsd_mean,jsd_std,runtime_mean,runtime_std
PN1,D3,MCAR,general,0.2,0.0832,0.0041,0.0614,0.0029,,,,,0.421,0.015,...,0.0123,0.0008,,,125.3,8.7
```

---

## 8. Reproducibility Checklist

- [ ] 所有 random seed 固定（NumPy, PyTorch, Python random）
- [ ] 資料前處理流程（刪除原始缺失 → split → scaling）在所有方法間一致
- [ ] Missing mask 在同一組 (dataset, mechanism, pattern, rate, seed) 下對所有方法完全相同
- [ ] Min-Max Scaling 的 fit 僅用 training set 的統計值
- [ ] 下游模型的超參數在所有實驗中固定
- [ ] Deep Learning 方法在 MPS 不穩定時自動 fallback 到 CPU
- [ ] 所有結果以 JSON 逐次儲存，避免中途 crash 丟失資料
- [ ] 提供 checkpoint / resume 機制：跳過已存在結果的 (dataset, method, mechanism, pattern, rate, seed) 組合

---

## 9. Key Implementation Notes

### 9.1 Missing Mask 共用

**關鍵**：同一組 `(dataset, mechanism, pattern, rate, seed)` 下，所有 14 個方法必須使用**完全相同的 missing mask**。

```python
# 先生成 mask，再分別套用到各方法
mask = generate_mask(data, mechanism, pattern, rate, seed)
for method in methods:
    result = method.impute(data, mask)
    evaluate(result, data, mask)
```

### 9.2 類別變數的特殊處理

| 方法 | 數值型 | 類別型 |
|------|--------|--------|
| Mean | 直接填均值 | Fallback: Mode |
| Median | 直接填中位數 | Fallback: Mode |
| EM | 直接適用 | Fallback: Mode |
| MICE | sklearn IterativeImputer | 可原生處理 |
| KNN | sklearn KNNImputer | 需先 encoding |
| MissForest | missforest 套件 | 原生支援 |
| SoftImpute | 直接適用 | 需先 one-hot，填補後取 argmax |
| XGBoost | XGBRegressor | XGBClassifier |
| MIDA | 一起輸入 | 需先 one-hot，填補後取 argmax |
| MIWAE | 一起輸入 | 需先 one-hot，填補後取 argmax |
| GAIN | 一起輸入 | 需先 one-hot，填補後取 argmax |
| TabCSDI | 原生支援 | 原生支援（one-hot / analog bits / FT） |
| MIRACLE | 一起輸入 | 需先 encoding |

### 9.3 Suggested Python Libraries

```python
# Data handling
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Statistical methods
from sklearn.impute import SimpleImputer           # Mean, Median
# EM: 使用 sklearn.mixture.GaussianMixture 或自行實作
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer         # MICE

# ML methods
from sklearn.impute import KNNImputer               # KNN
# MissForest: pip install missforest 或 missingpy
# SoftImpute: pip install fancyimpute
# XGBoost: 自行實作 iterative imputation with xgboost

# DL methods
# MIDA: 自行實作或 github repo
# MIWAE: github.com/pamattei/miwae
# GAIN: github.com/jsyoon0823/GAIN
# TabCSDI: github.com/pfnet-research/TabCSDI
# MIRACLE: github.com/vanderschaarlab/MIRACLE

# Evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

# Downstream model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
```

---

## 10. Ablation Studies（消融實驗，選做）

如果時間與篇幅允許，以下消融實驗可為論文增加深度：

1. **TabCSDI 類別編碼比較**：在 Mixed datasets 上比較 one-hot / analog bits / feature tokenization 三種編碼方式對 imputation quality 的影響。

2. **KNN 的 K 值敏感度**：測試 K = {3, 5, 7, 10, 15} 對 imputation quality 的影響。

3. **DL 方法訓練樣本量敏感度**：在 Adult dataset 上，分別用 {1K, 5K, 10K, 20K, 48K} 筆資料訓練，觀察各 DL 方法的效能與樣本量的關係。

4. **計算時間比較**：以 bar chart 呈現各方法在不同規模 dataset 上的平均 runtime。
