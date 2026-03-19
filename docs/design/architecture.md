# Missing Value Imputation 實驗平台軟體架構文檔

## 1. 文件目的

本文件旨在規劃一套可重複、可擴充、可比較的 Missing Value Imputation 實驗平台，用於系統化比較傳統統計方法、機器學習方法與深度學習方法在不同資料集、缺失機制與缺失比例下的表現。平台設計需滿足以下需求：

1. 在尚未完全確認最終資料集前，仍可先建立通用實驗基礎設施。
2. 可快速接入 UCI 與其他 tabular dataset。
3. 可統一管理缺失值注入、資料前處理、模型訓練、補值、評估與結果彙整流程。
4. 可支援研究用途所需的可重現性、批次實驗、自動報表與圖表輸出。
5. 可支援後續論文撰寫所需的表格、消融實驗、案例分析與附錄素材產出。

---

## 2. 平台目標

### 2.1 核心目標

本平台的核心目標不是只執行單次補值實驗，而是建立一個標準化實驗框架，使不同方法能在一致條件下被公平比較。平台應能統一處理以下變因：

- 資料集差異
- 特徵型態差異（numerical / categorical / mixed）
- 缺失機制差異（MCAR / MAR / MNAR）
- 缺失比例差異
- 方法類型差異（統計 / ML / DL）
- 隨機種子與重複次數差異

### 2.2 非功能目標

平台除了正確性之外，還需具備以下特性：

- **Reproducibility**：相同設定與隨機種子可重現結果。
- **Scalability**：可從少量方法擴充至大量方法與資料集。
- **Extensibility**：新增方法、資料集與評估指標不需大改主程式。
- **Traceability**：每次實驗都有完整設定、日誌與結果記錄。
- **Automation**：可一鍵執行批次實驗並自動匯總輸出。
- **Application Alignment**：實驗設計需盡可能貼近真實應用情境，而非僅停留在人工缺失下的理想化比較。
- **Practical Transferability**：平台輸出需可被其他研究者或應用團隊直接套用至自身資料情境，例如可明確回答「在何種資料條件下應優先選擇哪一類補值方法」。

---

## 3. 研究導向需求

本平台預期支援以下研究活動：

1. 比較不同補值方法在多個 UCI 與 tabular benchmark 資料集上的整體表現。
2. 分析不同缺失機制（MCAR / MAR / MNAR）下方法表現差異。
3. 分析不同缺失比例下的穩定性與退化趨勢。
4. 比較方法在 numerical、categorical 與 mixed-type 資料上的適用性。
5. 產出 downstream task evaluation，例如分類或回歸任務中補值後的預測表現。
6. 支援論文中的 ablation study，例如是否做標準化、是否使用缺失指示器、不同超參數設定等。
7. 支援誤差分析與案例視覺化。
8. 支援**更貼近真實場景的 benchmark protocol**，例如：
   - 訓練集與測試集缺失模式不同
   - 缺失比例隨欄位而非全域均勻分布
   - 特定族群或特定欄位更易缺失
   - 原始資料本身已存在真實缺失，再疊加人工缺失進行 controlled evaluation
9. 產出可泛化的實務建議，例如：
   - 小樣本 mixed-type 資料優先方法
   - 高缺失率情境之穩健方法
   - 計算資源有限時的最佳性價比方法
   - 對下游預測任務最友善的方法類型

---

## 4. 方法範圍

### 4.1 傳統統計方法

- Mean / Median / Mode Imputation
- KNN Imputation
- MICE
- EM-based Imputation（可選）
- Matrix Completion / softImpute

### 4.2 機器學習方法

- MissForest
- XGBoost-based Imputation
- Iterative Imputer with tree/boosting regressors
- 其他樹模型或集成方法（可擴充）

### 4.3 深度學習方法

- Autoencoder-based Imputation
- GAIN
- MIWAE
- MIRACLE
- CSDI
- 後續可擴充 diffusion / transformer-based variants

---

## 5. 整體架構概觀

平台採用模組化分層架構，分為以下七層：

1. **Dataset Layer**：資料集載入與欄位型態定義。
2. **Preprocessing Layer**：資料清理、型態轉換、編碼、標準化。
3. **Missingness Layer**：人工缺失注入與 mask 管理。
4. **Method Layer**：各類補值方法的統一介面與實作。
5. **Evaluation Layer**：補值品質與下游任務效能評估。
6. **Experiment Orchestration Layer**：批次實驗排程、設定管理、結果收集。
7. **Reporting Layer**：表格、圖表、案例分析與研究素材輸出。

系統設計理念為：

- 所有方法共享同一份資料切分與缺失 mask。
- 所有方法透過統一 API 被呼叫。
- 所有結果以結構化格式保存，便於後續彙總與論文撰寫。

---

## 6. 核心模組設計

## 6.1 Dataset Layer

### 職責

- 載入 UCI 或本地資料集
- 定義欄位型態與 target 欄位
- 統一輸出 DataFrame 與 metadata

### 建議設計

每個資料集實作為一個 adapter，遵循統一介面：

- `load()`：載入原始資料
- `get_features()`：取得特徵欄位
- `get_target()`：取得下游任務標籤
- `get_feature_types()`：回傳 numerical / categorical / binary / ordinal 資訊
- `get_splits()`：提供 train / validation / test 切分

### 資料集 metadata 建議欄位

- dataset_name
- n_samples
- n_features
- task_type
- feature_type_summary
- target_type
- original_missing_rate
- source

---

## 6.2 Preprocessing Layer

### 職責

- 缺失注入前的基礎清理
- 型態統一與編碼流程
- 為不同方法產生相容的輸入格式

### 設計原則

不同補值方法對資料型態要求不同，因此平台不應只有單一 preprocessing pipeline，而應採用兩段式設計：

1. **Canonical preprocessing**：資料清理、欄位型態識別、切分、基礎 schema 定義。
2. **Method-specific preprocessing**：根據方法需求決定 encoding、normalization、tensorization。

### 關鍵元件

- Numerical scaler
- Categorical encoder
- Mixed-type transformer
- Missing indicator generator
- Data tensor builder

---

## 6.3 Missingness Layer

### 職責

- 人工注入缺失值
- 儲存 ground-truth 與 mask
- 管理不同缺失機制與比例設定
- 模擬更接近真實應用場景的缺失分布

### 支援模式

- MCAR
- MAR
- MNAR
- Hybrid missingness（不同欄位採不同機制）
- Scenario-based missingness（依應用場景定義缺失規則）

### 設計原則

此層是整個平台最關鍵的公平比較基礎。所有方法必須使用同一組缺失注入結果，以確保方法之間比較公平。

但若研究目標希望更貼近真實應用，平台不應只停留在隨機遮罩，而應提供多層次 missingness protocol：

1. **Controlled Benchmark Protocol**：標準 MCAR / MAR / MNAR 設定，用於公平比較。
2. **Application-like Protocol**：模擬真實資料收集場景，例如欄位敏感性差異、族群偏差、值依賴缺失。
3. **Robustness Protocol**：訓練與測試採不同缺失模式，用於檢驗方法在 distribution shift 下的泛化能力。

### 核心輸出

- `X_complete`：完整原始資料
- `X_missing`：注入缺失後資料
- `missing_mask`：缺失位置 mask
- `observed_mask`：可觀測位置 mask
- `missing_config`：缺失機制與比例設定
- `scenario_tag`：場景標籤（例如 medical-like、survey-like、sensor-like）

### 補充設計

MNAR 注入建議支援 rule-based 策略，例如：

- 值越高越容易缺失
- 某群體在特定欄位更可能缺失
- 根據 target 或 proxy variable 產生缺失傾向
- 某些欄位具有不同的 base missing rate

這將有利於後續研究中討論不同方法對 realistic missingness 的適應能力，並讓最終結果不只回答「哪個方法平均最好」，而是回答「哪類方法在何種應用情境更值得採用」。

---

## 6.4 Method Layer

### 職責

- 封裝所有補值方法
- 對外提供統一 fit / impute / save / load 介面
- 隔離不同第三方套件依賴

### 統一抽象介面

每個方法都應繼承共同 base class，例如：

- `fit(X_train, metadata)`
- `impute(X_missing)`
- `fit_impute(X_train_missing, X_eval_missing=None)`
- `get_params()`
- `set_seed(seed)`

### 方法註冊機制

建議採用 registry 模式：

- `METHOD_REGISTRY["knn"]`
- `METHOD_REGISTRY["missforest"]`
- `METHOD_REGISTRY["gain"]`
- `METHOD_REGISTRY["csdi"]`

如此可讓實驗配置檔直接指定方法名稱，自動建立物件。

### 子模組分類

- `statistical/`
- `ml/`
- `dl/`

### 額外建議

每種方法應記錄：

- 適用資料型態
- 是否支援 mixed-type
- 是否需 GPU
- 是否可直接處理 categorical data
- 訓練時間與補值時間

這些資訊之後可以直接變成方法比較表。

---

## 6.5 Evaluation Layer

### 職責

- 衡量補值品質
- 評估下游任務影響
- 評估真實應用可遷移性
- 分析資源成本與穩定性

### 補值品質指標

對 numerical features：
- RMSE
- MAE
- NRMSE

對 categorical features：
- Accuracy
- F1-score
- Hamming loss（可選）

對 mixed-type 整體：
- feature-wise averaged score
- type-aware weighted score

### 下游任務評估

補值完成後可接一個固定 downstream model：

- classification：ROC-AUC, F1, Accuracy, PR-AUC
- regression：RMSE, MAE, R²

### 真實應用對齊評估

平台不應只報告補值誤差，還應評估方法是否能支持實務決策，因此建議加入：

- **Task Preservation**：補值後資料對下游任務的資訊保留程度
- **Ranking Consistency**：方法在不同資料集 / 機制下排名是否穩定
- **Robustness under Shift**：訓練與測試缺失模式不一致時的表現
- **Cost-effectiveness**：在時間與記憶體限制下的性價比
- **Failure Pattern Analysis**：方法在高缺失率、mixed-type 或 MNAR 下的失效模式

### 成本與穩定性評估

- training time
- inference / imputation time
- memory usage
- standard deviation across seeds
- convergence status

### 分析層級

- overall score per dataset
- score by missingness mechanism
- score by missing rate
- score by feature type
- score by method family
- score by scenario tag
- downstream-utility ranking

---

## 6.6 Experiment Orchestration Layer

### 職責

- 管理實驗設定
- 自動執行多組資料集與方法組合
- 保存結果與錯誤記錄
- 與 Weights & Biases（W&B）整合，統一管理實驗生命周期

### 設定檔驅動設計

建議所有實驗透過 YAML / JSON config 控制，避免寫死在 notebook 中。

### config 建議內容

- datasets
- split strategy
- missingness configs
- scenario tags
- methods
- hyperparameters
- seeds
- evaluation metrics
- output_dir
- downstream task settings
- wandb settings

### 執行單位設計

每個實驗執行單位可定義為：

`dataset × split × missing_mechanism × missing_rate × scenario × method × seed`

如此可完整展開成實驗矩陣，方便批量執行與後續聚合。

### 任務管理功能

- batch runner
- retry mechanism
- failure logging
- progress tracking
- experiment ID generation
- W&B run grouping
- W&B sweep support

### W&B 整合建議

平台應將 W&B 視為實驗控制中樞，而非僅為圖表工具。建議記錄以下資訊：

- config 與超參數
- dataset metadata
- missingness 設定
- 補值指標與下游任務指標
- training / imputation time
- 例外錯誤與狀態碼
- 方法輸出 artifact
- summary tables 與 figures

建議的 W&B 結構：

- **Project**：`missing-value-imputation-benchmark`
- **Group**：同一資料集或同一 scenario 的一批實驗
- **Run**：單一 `dataset × mechanism × rate × method × seed`
- **Artifact**：config、mask、summary table、plot、case study output

---

## 6.7 Reporting Layer

### 職責

- 將實驗結果整理為可讀、可發表、可引用的輸出

### 輸出格式

- CSV summary tables
- JSON detailed results
- Markdown reports
- LaTeX-ready tables
- plots (PNG / PDF)

### 圖表類型

- 不同方法在不同資料集上的總表
- missing rate vs performance 曲線
- MCAR / MAR / MNAR 分組比較圖
- runtime vs performance trade-off 圖
- feature-wise error heatmap
- critical difference diagram（可選）
- ranking table

### 案例分析輸出

- 單一資料集的補值前後樣本比較
- 某特徵欄位之實際值 vs 補值分布圖
- 不同方法在同一缺失位置的補值差異案例

---

## 7. 資料流程設計

平台建議採用以下標準流程：

1. 載入完整資料集
2. 定義欄位型態與任務資訊
3. 切分 train / validation / test
4. 在 train / validation / test 上依設定注入人工缺失
5. 保存完整值與缺失 mask
6. 呼叫補值方法進行 fit / impute
7. 對缺失位置進行補值品質評估
8. 將補值後資料送入固定 downstream model
9. 保存所有結果、圖表與案例分析
10. 匯總多次實驗產出最終報表

---

## 8. 建議專案目錄結構

```text
imputation-benchmark/
├─ configs/
│  ├─ datasets/
│  ├─ methods/
│  ├─ experiments/
│  └─ missingness/
├─ data/
│  ├─ raw/
│  ├─ processed/
│  ├─ masks/
│  └─ cache/
├─ src/
│  ├─ datasets/
│  ├─ preprocessing/
│  ├─ missingness/
│  ├─ methods/
│  │  ├─ statistical/
│  │  ├─ ml/
│  │  └─ dl/
│  ├─ evaluation/
│  ├─ downstream/
│  ├─ orchestration/
│  ├─ reporting/
│  └─ utils/
├─ scripts/
│  ├─ run_experiment.py
│  ├─ run_batch.py
│  ├─ summarize_results.py
│  └─ generate_report.py
├─ notebooks/
│  ├─ exploratory/
│  └─ case_studies/
├─ results/
│  ├─ tables/
│  ├─ figures/
│  ├─ logs/
│  └─ reports/
├─ tests/
├─ docs/
└─ README.md
```

---

## 9. 結果儲存設計

### 單次實驗建議保存欄位

- experiment_id
- dataset_name
- task_type
- method_name
- method_family
- missing_mechanism
- missing_rate
- seed
- feature_type_scope
- imputation_metrics
- downstream_metrics
- training_time
- inference_time
- resource_usage
- config_path
- status

### 儲存原則

- summary level：便於總覽與論文表格
- detailed level：便於誤差分析與附錄展示

---

## 10. 可重現性與實驗規範

### 必要規範

1. 統一隨機種子設定。
2. 同一資料集切分在不同方法間保持一致。
3. 同一缺失 mask 在不同方法間保持一致。
4. 所有方法使用統一評估函數。
5. 所有設定透過 config 檔保存。
6. 結果與版本資訊一併記錄。
7. 每個 run 必須同步記錄至 W&B，以確保結果可追蹤與可比較。
8. 必須區分 controlled benchmark 與 application-like benchmark，避免兩類結果混淆。

### 額外建議

- 使用 Hydra / OmegaConf 管理配置
- 使用 W&B 管理 run、group、sweep、artifact 與 dashboard
- 使用本地 SQLite / Parquet 作為離線結果備份
- 使用 DVC 或類似方式追蹤資料與結果版本

---

## 11. 擴充性設計

本平台後續應可輕易支援：

- 新資料集 adapter
- 新 missingness 生成策略
- 新補值方法 plugin
- 新下游任務模型
- 新評估指標
- 新報表模板

平台應避免在 main pipeline 中寫入過多 if-else，而應將方法差異封裝於 adapter、strategy 與 registry 中。

---

## 12. 建議技術選型

### 核心語言與框架

- Python
- pandas / numpy
- scikit-learn
- xgboost / lightgbm（如需）
- PyTorch
- PyYAML / OmegaConf / Hydra
- matplotlib / plotly（正式圖表）
- Weights & Biases（實驗追蹤、artifact 管理、dashboard）

### 本機硬體與執行環境建議

若平台部署於 MacBook Pro M4 Pro（48GB RAM），此硬體足以支撐大部分 tabular imputation benchmark 與中等規模深度學習方法開發。設計上可採以下策略：

- 優先支援 CPU 與 Apple Silicon MPS 後端
- 統計方法與樹模型預設走 CPU pipeline
- GAIN、MIWAE、Autoencoder 類方法可支援 MPS 加速
- 對 CSDI 等較重方法保留 batch size 與 feature size 控制機制
- 方法配置中加入 `device: cpu|mps` 選項
- 在 W&B 中記錄裝置資訊與資源消耗

### 工程管理

- Poetry 或 uv 管理依賴
- pytest 做單元測試
- pre-commit 做格式檢查
- logging 模組保存日誌

---

## 13. 最小可行版本（MVP）

### Phase 1：建立基礎骨架

- 完成 dataset adapter
- 完成 MCAR 注入
- 完成 Mean / KNN / MissForest / softImpute 的統一接口
- 完成 RMSE / MAE / Accuracy 評估
- 完成 CSV 結果匯總

### Phase 2：擴充研究功能

- 加入 MAR / MNAR
- 加入 XGBoost-based imputation
- 加入 downstream task evaluation
- 加入批次 config runner
- 加入圖表自動輸出

### Phase 3：擴充深度學習方法

- 加入 GAIN / MIWAE / CSDI
- 加入 GPU 任務支持
- 加入更完整案例分析
- 加入 LaTeX 表格輸出

---

## 14. 風險與注意事項

1. 不同方法對 categorical data 支援程度不同，需避免不公平比較。
2. MAR / MNAR 注入策略若設計不當，可能導致實驗不具說服力。
3. 深度學習方法訓練成本高，需在實驗矩陣設計上控制組合數量。
4. 若各方法 preprocessing 差異太大，可能降低比較公平性，因此需明確定義 canonical 與 method-specific preprocessing 的邊界。
5. 下游任務評估模型若不固定，可能混入額外變因。

---

## 15. 預期研究產出

透過本平台，可系統化產出以下研究成果：

1. 不同 missing value imputation 方法在多資料集上的主結果表。
2. 方法在不同缺失機制與缺失比例下的比較圖。
3. 方法在 mixed-type data 上的適用性分析。
4. runtime、resource 與 performance 的 trade-off 分析。
5. 下游分類或回歸任務之 performance comparison。
6. feature-wise error analysis。
7. 補值案例可視化圖與 qualitative comparison。
8. 論文中的 ablation study 與 appendix 素材。
9. **Application-oriented recommendation matrix**：例如在小樣本、高 MNAR、mixed-type、有限算力等情境下的建議方法。
10. **Scenario-based benchmark report**：針對 survey-like、medical-like、sensor-like 等情境輸出可泛用的選型建議。
11. **W&B dashboard**：用於持續監控實驗進度、比較 run、展示結果與保存 artifact。
12. **Reusable benchmark assets**：包含缺失注入設定、mask artifact、config template、實驗報表模板，便於他人複製至自身應用。

## 16. Repo Folder Structure 與檔案責任表

以下結構以「研究平台 + 可持續擴充 benchmark repo」為目標設計，兼顧可讀性、模組化與後續論文產出需求。

```text
imputation-benchmark/
├─ README.md
├─ pyproject.toml
├─ .env.example
├─ .gitignore
├─ configs/
│  ├─ base.yaml
│  ├─ datasets/
│  │  ├─ adult.yaml
│  │  ├─ heart.yaml
│  │  └─ wine.yaml
│  ├─ methods/
│  │  ├─ mean.yaml
│  │  ├─ knn.yaml
│  │  ├─ missforest.yaml
│  │  ├─ softimpute.yaml
│  │  ├─ gain.yaml
│  │  ├─ miwae.yaml
│  │  └─ csdi.yaml
│  ├─ missingness/
│  │  ├─ mcar_10.yaml
│  │  ├─ mar_30.yaml
│  │  ├─ mnar_50.yaml
│  │  └─ survey_like.yaml
│  ├─ downstream/
│  │  ├─ clf_default.yaml
│  │  └─ reg_default.yaml
│  └─ experiments/
│     ├─ benchmark_core.yaml
│     ├─ benchmark_application.yaml
│     └─ ablation_mnar.yaml
├─ data/
│  ├─ raw/
│  ├─ interim/
│  ├─ processed/
│  ├─ masks/
│  └─ cache/
├─ docs/
│  ├─ architecture.md
│  ├─ config_schema.md
│  ├─ api_design.md
│  └─ experiment_protocol.md
├─ src/
│  ├─ __init__.py
│  ├─ registry.py
│  ├─ constants.py
│  ├─ schemas/
│  │  ├─ config_schema.py
│  │  ├─ result_schema.py
│  │  └─ metadata_schema.py
│  ├─ datasets/
│  │  ├─ base.py
│  │  ├─ adult.py
│  │  ├─ heart.py
│  │  └─ utils.py
│  ├─ preprocessing/
│  │  ├─ canonical.py
│  │  ├─ encoders.py
│  │  ├─ scalers.py
│  │  ├─ indicators.py
│  │  └─ builders.py
│  ├─ missingness/
│  │  ├─ base.py
│  │  ├─ mcar.py
│  │  ├─ mar.py
│  │  ├─ mnar.py
│  │  ├─ hybrid.py
│  │  └─ scenario.py
│  ├─ methods/
│  │  ├─ base.py
│  │  ├─ statistical/
│  │  │  ├─ mean.py
│  │  │  ├─ knn.py
│  │  │  ├─ mice.py
│  │  │  └─ softimpute.py
│  │  ├─ ml/
│  │  │  ├─ missforest.py
│  │  │  ├─ xgb_iterative.py
│  │  │  └─ sklearn_iterative.py
│  │  └─ dl/
│  │     ├─ autoencoder.py
│  │     ├─ gain.py
│  │     ├─ miwae.py
│  │     └─ csdi.py
│  ├─ evaluation/
│  │  ├─ imputation_metrics.py
│  │  ├─ downstream_metrics.py
│  │  ├─ robustness_metrics.py
│  │  └─ aggregators.py
│  ├─ downstream/
│  │  ├─ base.py
│  │  ├─ classifiers.py
│  │  └─ regressors.py
│  ├─ orchestration/
│  │  ├─ runner.py
│  │  ├─ batch_runner.py
│  │  ├─ pipeline.py
│  │  ├─ checkpoint.py
│  │  └─ errors.py
│  ├─ tracking/
│  │  ├─ wandb_logger.py
│  │  ├─ artifact_manager.py
│  │  └─ local_logger.py
│  ├─ reporting/
│  │  ├─ tables.py
│  │  ├─ plots.py
│  │  ├─ latex.py
│  │  └─ case_study.py
│  └─ utils/
│     ├─ seed.py
│     ├─ io.py
│     ├─ device.py
│     ├─ timing.py
│     └─ logging.py
├─ scripts/
│  ├─ run_experiment.py
│  ├─ run_batch.py
│  ├─ summarize_results.py
│  ├─ build_case_study.py
│  └─ sync_wandb_artifacts.py
├─ notebooks/
│  ├─ sanity_check.ipynb
│  ├─ method_debug.ipynb
│  └─ qualitative_analysis.ipynb
├─ results/
│  ├─ tables/
│  ├─ figures/
│  ├─ logs/
│  ├─ reports/
│  └─ exports/
└─ tests/
   ├─ test_datasets.py
   ├─ test_missingness.py
   ├─ test_methods.py
   ├─ test_pipeline.py
   └─ test_metrics.py
```

### 檔案責任表

#### 根目錄
- `README.md`：專案介紹、快速開始、實驗入口。
- `pyproject.toml`：依賴管理與套件設定。
- `.env.example`：W&B API key、資料路徑等環境變數樣板。

#### `configs/`
- 保存所有可重現實驗設定。
- `datasets/`：資料集來源、target 欄位、型態宣告。
- `methods/`：方法超參數與裝置設定。
- `missingness/`：缺失機制、比例、scenario 規則。
- `experiments/`：批次實驗組合入口。

#### `src/datasets/`
- 封裝所有資料集 adapter。
- 將不同資料來源轉成統一輸出格式。

#### `src/missingness/`
- 缺失注入的核心邏輯。
- 保證同一 mask 可被多方法共用。

#### `src/methods/`
- 每個補值方法都以獨立模組封裝。
- 提供統一 `fit / impute / fit_impute` 介面。

#### `src/evaluation/`
- 計算補值品質、下游任務效果、穩健性指標。

#### `src/orchestration/`
- 控制整條 pipeline 的執行順序。
- 負責批次展開實驗矩陣與錯誤處理。

#### `src/tracking/`
- 與 W&B、local logs、artifact storage 整合。

#### `src/reporting/`
- 將結果轉為 CSV、LaTeX、圖表與案例分析。

#### `tests/`
- 驗證資料載入、缺失注入、方法輸出與整條 pipeline 的一致性。

---

## 17. YAML Config Schema 設計

平台應採層級式 config 設計，使單次實驗與批次實驗都能透過組合組態完成。

### 17.1 Base Config 範例

```yaml
project:
  name: missing-value-imputation-benchmark
  task_name: benchmark_core
  seed: 42
  output_dir: results/

runtime:
  device: cpu
  num_workers: 4
  deterministic: true

tracking:
  use_wandb: true
  project: missing-value-imputation-benchmark
  entity: null
  group: adult_mcar10
  tags: [uci, benchmark, mcar]
  log_artifacts: true

experiment:
  repeat_seeds: [42, 52, 62]
  save_imputed_data: false
  save_masks: true
  save_case_studies: true
```

### 17.2 Dataset Config 範例

```yaml
dataset:
  name: adult
  source: uci
  path: data/raw/adult.csv
  target_col: income
  task_type: classification
  split:
    train_ratio: 0.7
    valid_ratio: 0.1
    test_ratio: 0.2
    stratify: true
  feature_types:
    numerical: [age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week]
    categorical: [workclass, education, marital_status, occupation, relationship, race, sex, native_country]
    binary: []
    ordinal: []
  preprocessing:
    drop_duplicates: true
    handle_original_missing: keep
    categorical_encoding: ordinal_for_tree
    numerical_scaling: standard_for_dl
```

### 17.3 Missingness Config 範例

```yaml
missingness:
  protocol: controlled
  mechanism: MCAR
  rate: 0.1
  apply_to: all_features
  seed: 123
  scenario_tag: benchmark_mcar
  options:
    per_feature_rate: false
    exclude_target: true
```

### 17.4 Application-like Missingness 範例

```yaml
missingness:
  protocol: application_like
  mechanism: hybrid
  rate: 0.3
  scenario_tag: survey_like
  apply_to: selected_features
  selected_features: [age, occupation, hours_per_week]
  rules:
    - type: value_dependent
      column: hours_per_week
      direction: high_more_missing
      strength: 0.7
    - type: group_dependent
      group_column: sex
      group_value: Female
      target_column: occupation
      extra_missing_rate: 0.15
    - type: base_rate_by_feature
      mapping:
        age: 0.10
        occupation: 0.35
        hours_per_week: 0.20
  options:
    exclude_target: true
    keep_original_missing: true
```

### 17.5 Method Config 範例

```yaml
method:
  name: missforest
  family: ml
  device: cpu
  random_state: 42
  params:
    n_estimators: 200
    max_depth: null
    max_iter: 10
    n_jobs: -1
  compatibility:
    supports_mixed_type: true
    requires_complete_target: true
```

```yaml
method:
  name: gain
  family: dl
  device: mps
  random_state: 42
  params:
    batch_size: 256
    hidden_dim: 128
    epochs: 200
    lr: 0.001
    hint_rate: 0.9
    alpha: 10.0
```

### 17.6 Downstream Config 範例

```yaml
downstream:
  enabled: true
  model_name: xgboost_classifier
  fixed_hparams: true
  params:
    n_estimators: 300
    max_depth: 6
    learning_rate: 0.05
  metrics: [roc_auc, f1, accuracy, pr_auc]
```

### 17.7 Evaluation Config 範例

```yaml
evaluation:
  imputation_metrics:
    numerical: [rmse, mae, nrmse]
    categorical: [accuracy, f1]
  robustness_metrics: [ranking_consistency, failure_rate]
  aggregate:
    by_feature_type: true
    by_missing_rate: true
    by_scenario_tag: true
```

### 17.8 Batch Experiment Config 範例

```yaml
batch:
  datasets: [adult, heart, wine]
  missingness_profiles: [mcar_10, mar_30, survey_like]
  methods: [mean, knn, missforest, softimpute, gain]
  seeds: [42, 52, 62]
  parallel:
    enabled: false
    max_concurrent_runs: 1
```

### 17.9 Schema 設計原則

1. **分層可覆寫**：base config 提供預設值，dataset / method / experiment 再覆蓋。
2. **方法與資料集解耦**：方法 config 不直接綁定某個資料集。
3. **scenario 可組合**：controlled 與 application-like protocol 可同時存在。
4. **追蹤欄位內建**：W&B 所需欄位直接在 config 中聲明。
5. **易於 sweep**：method params 與 missingness params 應易於被展開成搜尋空間。

---

## 18. 核心 Python Classes 與 API 介面設計

### 18.1 設計原則

平台應以「清楚的抽象介面 + 最少耦合」為原則。核心類別之間的依賴關係應保持單向：

- Dataset 提供資料
- Missingness 產生缺失版本
- Method 接收缺失資料並回傳補值結果
- Evaluation 消費 ground truth 與 imputed output
- Runner 負責協調整體流程

### 18.2 Dataset 抽象類別

```python
class BaseDataset:
    name: str

    def load(self):
        """Load raw dataset and return canonical dataframe."""

    def get_metadata(self) -> dict:
        """Return dataset metadata including feature types and task info."""

    def get_splits(self, seed: int):
        """Return train/valid/test split indices or dataframes."""
```

#### 建議職責
- 載入資料
- 宣告 feature types
- 提供 split
- 輸出 metadata

### 18.3 Missingness Generator 抽象類別

```python
class BaseMissingnessGenerator:
    name: str

    def generate(self, X, metadata, seed: int):
        """Return X_missing, missing_mask, observed_mask, missing_config."""
```

#### 具體子類
- `MCARGenerator`
- `MARGenerator`
- `MNARGenerator`
- `HybridScenarioGenerator`

### 18.4 Imputer 抽象類別

```python
class BaseImputer:
    name: str
    family: str

    def set_seed(self, seed: int) -> None:
        pass

    def fit(self, X_train, metadata: dict | None = None) -> None:
        pass

    def impute(self, X_missing):
        pass

    def fit_impute(self, X_train_missing, X_eval_missing=None, metadata: dict | None = None):
        pass

    def get_params(self) -> dict:
        pass
```

#### 介面設計重點
- 所有方法統一輸出 `X_imputed`
- 方法內部若需不同格式轉換，由 method wrapper 自行處理
- 深度學習方法可額外實作 `save_checkpoint()` / `load_checkpoint()`

### 18.5 Preprocessor 類別

```python
class Preprocessor:
    def fit(self, X_train, metadata: dict):
        pass

    def transform(self, X):
        pass

    def inverse_transform(self, X):
        pass
```

#### 說明
- canonical preprocessing 與 method-specific preprocessing 可拆為兩層物件
- 例如 `CanonicalPreprocessor`、`DLPreprocessor`

### 18.6 Evaluator 類別

```python
class ImputationEvaluator:
    def evaluate(self, X_true, X_imputed, missing_mask, metadata: dict) -> dict:
        pass

class DownstreamEvaluator:
    def fit_and_evaluate(self, X_train, y_train, X_test, y_test, task_type: str) -> dict:
        pass

class RobustnessEvaluator:
    def evaluate(self, run_results: list[dict]) -> dict:
        pass
```

### 18.7 Tracker 類別

```python
class ExperimentTracker:
    def start_run(self, config: dict):
        pass

    def log_metrics(self, metrics: dict, step: int | None = None):
        pass

    def log_artifact(self, path: str, artifact_type: str):
        pass

    def finish_run(self, status: str = "completed"):
        pass
```

#### 具體實作
- `WandbTracker`
- `LocalTracker`

### 18.8 Result Schema

```python
from dataclasses import dataclass

@dataclass
class ExperimentResult:
    experiment_id: str
    dataset_name: str
    method_name: str
    method_family: str
    missing_mechanism: str
    missing_rate: float
    scenario_tag: str
    seed: int
    imputation_metrics: dict
    downstream_metrics: dict
    runtime_metrics: dict
    status: str
```

### 18.9 Pipeline Runner

```python
class ExperimentRunner:
    def __init__(self, dataset, missingness_generator, imputer, evaluators, tracker, config):
        self.dataset = dataset
        self.missingness_generator = missingness_generator
        self.imputer = imputer
        self.evaluators = evaluators
        self.tracker = tracker
        self.config = config

    def run(self) -> dict:
        """Execute a single experiment and return structured results."""
```

### 18.10 Batch Runner

```python
class BatchExperimentRunner:
    def __init__(self, experiment_matrix: list[dict], factory):
        self.experiment_matrix = experiment_matrix
        self.factory = factory

    def run_all(self) -> list[dict]:
        pass
```

#### 職責
- 展開 `dataset × missingness × method × seed`
- 逐一建立物件並執行
- 收集所有 `ExperimentResult`

### 18.11 Factory / Registry 設計

```python
class ComponentFactory:
    def create_dataset(self, dataset_cfg: dict):
        pass

    def create_missingness(self, missingness_cfg: dict):
        pass

    def create_imputer(self, method_cfg: dict):
        pass

    def create_tracker(self, tracking_cfg: dict):
        pass
```

#### 說明
- 透過 registry/factory 避免在 runner 中寫大量 if-else
- 新增方法時，只需註冊一次即可被整個平台使用

### 18.12 建議的單次實驗 API 流程

```python
def run_single_experiment(config: dict) -> dict:
    dataset = factory.create_dataset(config["dataset"])
    missingness = factory.create_missingness(config["missingness"])
    imputer = factory.create_imputer(config["method"])
    tracker = factory.create_tracker(config["tracking"])

    runner = ExperimentRunner(
        dataset=dataset,
        missingness_generator=missingness,
        imputer=imputer,
        evaluators={
            "imputation": ImputationEvaluator(),
            "downstream": DownstreamEvaluator(),
            "robustness": RobustnessEvaluator(),
        },
        tracker=tracker,
        config=config,
    )
    return runner.run()
```

### 18.13 測試重點

平台落地時建議優先撰寫以下測試：

1. Dataset adapter 是否正確輸出 metadata
2. Missingness generator 是否能穩定生成相同 mask
3. 每個 imputer 是否遵守統一輸入輸出格式
4. Evaluation 指標是否只在缺失位置計算
5. Batch runner 是否能正確展開實驗矩陣
6. W&B tracker 是否能正確記錄 config、metric、artifact

---

## 19. 實作落地優先順序

### Phase A：先搭骨架
1. base config
2. dataset adapter interface
3. missingness generator interface
4. imputer base class
5. experiment runner
6. local result saving
7. W&B tracker

### Phase B：先完成 baseline benchmark
1. Adult / Heart / Wine 三個 dataset adapter
2. MCAR / MAR
3. Mean / KNN / MissForest / softImpute
4. imputation metrics + downstream metrics
5. summary table + basic plots

### Phase C：再擴充真實場景
1. survey-like / medical-like / sensor-like scenario generator
2. hybrid missingness
3. XGBoost-based method
4. GAIN / MIWAE
5. robustness under shift evaluation

### Phase D：再加入較重模型與完整論文輸出
1. CSDI
2. artifact-rich case study
3. LaTeX table export
4. recommendation matrix auto-generation

---

## 20. 結語

本文件至此已從概念架構進一步落成為可實作的工程文件，涵蓋了平台層級架構、資料流程、repo 結構、config schema 與核心 API 設計。下一步若要真正開始開發，最適合先建立 `base config + dataset adapter + missingness generator + BaseImputer + ExperimentRunner + WandbTracker` 這六個核心元件。一旦這條主幹完成，其餘方法與資料集都可在此骨架上持續擴充。

