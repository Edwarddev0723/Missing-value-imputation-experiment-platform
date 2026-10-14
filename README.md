# ImputeBench｜Missing Value Imputation Experiment Platform

表格缺失值補齊的研究與實驗平台，透過統一的 `fit()`／`impute()` 介面與 YAML 設定，比較統計方法、樹模型及 deep learning 方法，並同時觀察補齊品質、downstream 表現與計算成本。

作品集重點是**可擴充的實驗架構、受控缺失值模擬、評估定義與結果追溯**。Repository 保存 13 種方法、11 個 dataset identifiers、17,160 筆結果紀錄；目前部分紀錄的資料版本／split 條件不一致，因此排名應解讀為歷史產物的描述性分析，尚不能作為公平比較各演算法優劣的定論。

## 研究問題

1. 缺失值的生成方式、缺失比例及資料特性，如何影響補齊方法的表現？
2. 補齊誤差較低的方法，是否也能改善固定 downstream 模型的預測？
3. 如何在擴充方法與 datasets 時，保留一致的資料介面、評估定義及可核對的實驗產物？

## 專案實作範圍

- **實驗平台**：dataset／method／missingness registry 與 component factory，以 YAML 展開實驗組合；runner 儲存 raw JSON、完整設定及 runtime。
- **方法介面**：整合傳統補齊方法、迭代樹模型、autoencoder、GAN、diffusion 與 causal regularization 原型。
- **評估流程**：對人工遮蔽的位置計算補齊品質，搭配固定 RandomForest downstream 模型與分布距離。
- **結果分析**：彙整 JSON／CSV、檢查組合完整度、計算條件內排名，產生分層比較、統計檢定、表格及圖形。

以上為 repository 可檢查的實作範圍；方法名稱不代表已完成原論文的完整復現，實際部署與外部採用亦未驗證。

## 實驗架構

```mermaid
flowchart LR
    A["YAML 與 registry"] --> B["Dataset adapter 與 split"]
    B --> C["訓練資料 fit 前處理"]
    C --> D["人工缺失值與 mask"]
    D --> E["Imputer fit 與 train/test impute"]
    E --> F["Masked metrics 與固定 RF"]
    F --> G["JSON/CSV、排名與分析"]
```

[ExperimentRunner][runner] 負責單次流程；[batch configs][batch-config] 定義多方法、多條件與多 seed 組合。前處理在完整的訓練子集上 fit numerical MinMaxScaler 與 categorical OrdinalEncoder，再注入缺失值；test 使用同一組前處理參數。這是先有完整 GT、再模擬缺失的離線實驗設定。

## 已保存的實驗範圍

| 項目 | 結果 CSV 的實際範圍 |
|---|---|
| Non-DL 方法（8） | mean、median、em、mice、knn、missforest、softimpute、xgboost |
| DL 方法（5） | mida、miwae、gain、tabcsdi、miracle |
| Dataset identifiers（11） | adult、autism、bank、breast_cancer、concrete、letter、mammographic、spambase、titanic、wine、yeast |
| 缺失值生成器 | MCAR、MAR、MNAR；以下保留 repository 的機制名稱，MAR 實作限制見後文 |
| Patterns | general、univariate |
| 名目缺失比例 | 0.1、0.2、0.3、0.5 |
| Seeds | 42、123、456、789、1024 |
| 紀錄數 | 13 × 11 × 3 × 2 × 4 × 5 = **17,160** |
| 條件數 | 1,320 組，每組有 13 種方法的紀錄 |

Source 另有 `heart` adapter／config，但本份 [full_results.csv][results] 沒有 Heart 結果，不能將 12 個可用 adapters 寫成 12 個已完成 benchmark datasets。

Base／完整 batch 的預設 split 為 train/test = 80/20、valid = 0；單次示例 config 使用 70/10/20，且 missingness seed 為 123。這些是目前設定檔內容，不能代替每筆歷史紀錄的實際 split 證據。

`univariate` 會在各 split 的 numerical columns 中選擇最高變異數欄位；比例指向被選欄位的遮蔽機率，並非整張矩陣的缺失比例。`general` 對多欄注入缺失，並恢復全缺失列的一個值，因此名目比例也不等於實現的精確比例。

## 評估定義

| 面向 | 實作與解讀 |
|---|---|
| Numerical imputation | 在人工遮蔽的 numerical test positions 計算 RMSE／MAE；數值採用訓練資料 fit 的 MinMax scaling |
| Categorical imputation | 在人工遮蔽的 categorical positions 計算 Accuracy／Macro-F1；連續輸出先四捨五入，metrics 彙集各欄的 encoded labels |
| Downstream | 固定 RandomForest；batch 設定 `n_estimators=100`、`random_state=42`。分類排名用 weighted F1；迴歸排名用負 RMSE |
| Distribution | numerical Wasserstein distance 與 categorical Jensen–Shannon **distance**，使用完整欄位分布；原欄位名保留 `jsd_mean` |
| Runtime | `fit` 加上 train/test `impute` 的計時，不包含下載、前處理、downstream 訓練及整體分析 |
| Overall imputation rank | 每個 `dataset × mechanism × pattern × ratio × seed` 條件內，以 numerical RMSE 升序排名，ties 取平均，再跨條件取平均 |

因此，overall imputation rank 衡量 numerical 補齊表現，沒有合併 categorical Accuracy／F1，也不能直接當作 downstream 品質排名。分布指標的 `jsd_mean` 呼叫 [SciPy jensenshannon][js-doc]，回傳的是 divergence 的平方根。

## 可核對的歷史結果

以下數字來自已提交的 [leaderboard.csv][leaderboard]，已用 [full_results.csv][results] 及 [排名公式][ranking] 重算，所有 leaderboard numeric 欄位在四位小數上一致。本次沒有重新執行 imputer、模型訓練或完整 benchmark。

**比較條件限制：1,320 組條件中，有 123 組的 `n_samples`／`n_features` 在方法間不一致，其中包含全部 120 組 Yeast 條件。下表保留原始排名作為產物核對，不能據此宣稱已完成公平的 Non-DL／DL 比較。**

| 方法 | 平均條件內 rank ↓ | 跨條件 rank std | 平均 runtime（s） | 中位 runtime（s） |
|---|---:|---:|---:|---:|
| missforest | 2.0629 | 1.7946 | 79.1843 | 10.7274 |
| xgboost | 2.7095 | 1.7543 | 71.5997 | 2.0946 |
| knn | 4.4879 | 2.6413 | 10.3988 | 0.1243 |
| em | 5.0508 | 2.9380 | 2.1174 | 0.1166 |
| mice | 5.5227 | 2.8238 | 3.5210 | 0.5829 |
| mean | 7.2326 | 2.1856 | 0.0023 | 0.0015 |
| mida | 7.6636 | 2.7395 | 8.1252 | 2.4305 |
| miwae | 7.6928 | 2.9115 | 17.4217 | 5.6227 |
| softimpute | 8.4072 | 2.6676 | 1.2924 | 0.3580 |
| gain | 8.6352 | 2.5428 | 1.8795 | 0.8209 |
| median | 9.2220 | 2.5561 | 0.0032 | 0.0020 |
| miracle | 10.6561 | 2.5990 | 21.0012 | 3.6957 |
| tabcsdi | 11.6568 | 1.9324 | 9.4999 | 1.9572 |

每種方法別名有 1,320 筆紀錄；`xgboost` 的實際 backend 尚未逐筆確認，見後文。`rank std` 混合了 dataset、mechanism、pattern、ratio 與 seed 的變化，不是固定條件下的 seed stability 或信賴區間。Runtime 是歷史記錄值，缺少完整逐筆硬體／環境資訊，不能當成跨硬體速度比較。

可保留的描述性觀察：

- MissForest 的整體 numerical 平均 rank 最低；按原彙整方式，在 9／11 個 dataset identifiers 上取得最低平均 rank。Adult 的最佳平均 rank 為紀錄別名 `xgboost`，Autism 為 MICE。
- 原產物的 Non-DL／DL 平均 rank 分別為 5.5869／9.2609；這個差距受到資料版本、方法實作與訓練設定影響，不能外推為方法家族普遍優劣。
- Mean／Median 的歷史計時較低，適合作為成本基準；是否滿足特定資料集的品質需求，需要依任務指標判斷。

[Friedman 與 pairwise Wilcoxon 分析][stats-source] 以 1,320 個條件的排名為 blocks；78 組 pairwise tests 中，76 組在 Holm 校正、α=0.05 後顯著，已重算確認。但資料條件不一致、同 dataset 的重複 splits／settings 依賴性尚未處理，且已保存的 69 組 `effect_size_r` 為 `inf`，不宜將這些 p-values 或 effect sizes 當作確立演算法優劣的證據。

## 已知限制與未驗證項目

| 項目 | 目前證據與限制 |
|---|---|
| Cohort／split 一致性 | Yeast 的五種 DL 方法均記錄 `n_samples=1933, n_features=117`；多數 Non-DL 為 `1187, 8`，Mean 有一筆前者。Wine 的 KNN 有兩筆不同樣本數；Breast Cancer 的 MIDA 有一筆不同樣本數。目前 runner 的 `n_samples` 代表訓練集筆數。原始資料、split 與 mask identities 尚待逐筆對齊 |
| Dataset 身份與 target | `mammographic` loader 指向 [OpenML 310：mammography][openml-310]，與原規格的 Mammographic Masses 不同。Wine 的歷史紀錄有 12 個 features 且全缺 downstream metrics；target 是否正確排除、是否有標籤欄混入 features 的風險，尚待原始資料／欄位 manifest 核對 |
| Downstream coverage | 主要 task metric 可用 **14,999／17,160**；Wine 缺 1,560 筆，Yeast 缺 601 筆。原排名程式使用 `na_option="bottom"`，會把缺失 metrics 排入名次；Wine 全缺值的條件仍得到平均並列名次 7，因此不沿用 overall downstream rank 作為效能成果 |
| 方法與資料型別 | `missforest` 是 sklearn IterativeImputer + RandomForestRegressor，對 ordinal-encoded categoricals 也用迴歸。`xgboost` 在 import 失敗時會 fallback 至 HistGradientBoostingRegressor，歷史 backend 未逐筆確認。SoftImpute、TabCSDI、MIRACLE 標示為簡化實作；SoftImpute 在待補齊的 test matrix 上迭代 SVD，具有 transductive 特性 |
| 缺失值機制 | `MAR/general` 的 driver 也可能被遮蔽，程式未強制 driver 完全可觀測；機制名稱不能視為嚴格 MAR 已獲驗證。`univariate` 的選欄也會依 train／test 各自的變異數而不同 |
| 訓練與 runtime 可比性 | DL configs 指定 MPS，但 device helper 在 worker processes 強制 CPU、裝置不可用時也回退 CPU；上限為 100 或 150 epochs，使用 training-loss early stopping。沒有一致調參／計算預算的證據；Timer 未同步 accelerator，完整歷史裝置設定未保存於 raw JSON，耗時只作描述性資訊 |
| 再現性與測試 | 組合完整度、既有 CSV 的數值重算與 source inspection 已核對；完整訓練重跑、資料／mask hashes、原論文 fidelity 與 unit tests 本次未驗證。`set_global_seed` 同時呼叫 `torch.use_deterministic_algorithms(False)`，不能僅憑 seed 宣稱完全 deterministic |

## 安裝與使用

需求：Python ≥3.10；以下採用 `uv`。本次文件核對未執行全新環境安裝，也未重跑模型。

```bash
git clone https://github.com/Edwarddev0723/Missing-value-imputation-experiment-platform.git
cd Missing-value-imputation-experiment-platform

uv venv --python 3.11
uv pip install -e ".[dev]"
```

先閱讀既有結果，可用下列指令核對資料集、方法與唯一組合數，不會執行訓練：

```bash
uv run --no-sync python - <<'PY'
import csv
from collections import Counter

with open("benchmark/full_results.csv", newline="") as f:
    rows = list(csv.DictReader(f))
keys = ("dataset", "method", "mechanism", "pattern", "ratio", "seed")
unique = {tuple(r[k] for k in keys) for r in rows}
print("records:", len(rows), "unique combinations:", len(unique))
print("datasets:", dict(Counter(r["dataset"] for r in rows)))
print("methods:", dict(Counter(r["method"] for r in rows)))
PY
```

單次實驗入口為 `scripts/run_experiment.py`。若要嘗試 Adult + Mean，可先另存示例 config，將輸出放在新目錄，避免改寫既有結果：

```bash
mkdir -p work
uv run --no-sync python - <<'PY'
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("configs/experiments/single_adult_mean.yaml").read_text())
cfg["project"]["output_dir"] = "results_scratch/single_adult_mean/"
cfg["tracking"]["output_dir"] = "results_scratch/single_adult_mean/logs/"
with Path("work/single_adult_mean_portfolio.yaml").open("x") as f:
    yaml.safe_dump(cfg, f)
PY
uv run --no-sync python scripts/run_experiment.py \
  --config work/single_adult_mean_portfolio.yaml --no-wandb
```

這個示例會下載／讀取資料及執行一次離線實驗；其 split 與 missingness seed 不等於完整 benchmark。重新跑同一示例前，也需更換輸出目錄以保留舊結果。

XGBoost／DL 需要另裝 `.[ml]`／`.[dl]` extras，DL device 應依實際 CPU／CUDA／MPS 環境選擇。完整重跑的設定入口是 [Non-DL config][nondl-config] 與 [DL config][dl-config]；兩者合起來才涵蓋 13 種方法。重跑前應先處理前述 cohort／target 問題，並另存 configs、設定新的資料及結果目錄。

分析入口 `analysis/scripts/run_all.py` 會重寫 `analysis/outputs/`；重跑分析與 `pytest tests/ -v` 都應在獨立工作副本中執行。本次不以原 README 的靜態「40 passed」badge 作為已通過測試的證據。

## Repository 導覽

| 路徑 | 內容 |
|---|---|
| `src/datasets/` | Dataset adapters、metadata 與 split |
| `src/missingness/` | 生成器、選欄與 mask helpers |
| `src/methods/` | 統一 API 與方法實作 |
| `src/orchestration/` | 單次／batch runner |
| `src/evaluation/` | 補齊與 downstream metrics |
| `configs/` | 方法、資料與實驗設定 |
| `results/raw/`、`results/runs/` | 原始紀錄與完整 run 產物 |
| `benchmark/` | 已提交的 CSV、表格與圖形 |
| `analysis/` | 彙整、排名、檢定及報告程式 |
| `tests/`、`docs/` | 測試 source 與使用文件 |

## 後續研究與工程工作

優先建立 dataset version、target／feature schema、split IDs、mask hashes 與 source／environment manifest，再修正缺失值機制及 downstream 的 missing-metric 處理。完成條件對齊後，才進行受控重跑、方法 fidelity 核對、固定預算調參與跨 seed／dataset 的不確定性分析。

`pyproject.toml` 宣告 MIT，但此快照根目錄沒有 `LICENSE` 全文；授權文件仍需補齊。`CITATION.cff` 也保留 placeholder repository URL 與舊 dataset 數，引用本專案時請使用此 repository 的實際 URL 與 commit。

結果核對日期：2026-10-14；基準快照：`8faa48c7ef739175470c97cc52dd7f1e3dc477a9`。本 README 的核對範圍是已提交產物與 source，未建立新的 benchmark 結果。

[runner]: https://github.com/Edwarddev0723/Missing-value-imputation-experiment-platform/blob/8faa48c7ef739175470c97cc52dd7f1e3dc477a9/src/orchestration/runner.py
[batch-config]: https://github.com/Edwarddev0723/Missing-value-imputation-experiment-platform/blob/8faa48c7ef739175470c97cc52dd7f1e3dc477a9/configs/experiments/full_benchmark.yaml
[nondl-config]: https://github.com/Edwarddev0723/Missing-value-imputation-experiment-platform/blob/8faa48c7ef739175470c97cc52dd7f1e3dc477a9/configs/experiments/full_benchmark_nondl.yaml
[dl-config]: https://github.com/Edwarddev0723/Missing-value-imputation-experiment-platform/blob/8faa48c7ef739175470c97cc52dd7f1e3dc477a9/configs/experiments/full_benchmark_dl.yaml
[results]: https://github.com/Edwarddev0723/Missing-value-imputation-experiment-platform/blob/8faa48c7ef739175470c97cc52dd7f1e3dc477a9/benchmark/full_results.csv
[leaderboard]: https://github.com/Edwarddev0723/Missing-value-imputation-experiment-platform/blob/8faa48c7ef739175470c97cc52dd7f1e3dc477a9/benchmark/leaderboard.csv
[ranking]: https://github.com/Edwarddev0723/Missing-value-imputation-experiment-platform/blob/8faa48c7ef739175470c97cc52dd7f1e3dc477a9/analysis/scripts/step3_analysis.py
[stats-source]: https://github.com/Edwarddev0723/Missing-value-imputation-experiment-platform/blob/8faa48c7ef739175470c97cc52dd7f1e3dc477a9/analysis/scripts/step5_stats.py
[openml-310]: https://www.openml.org/api/v1/json/data/310
[js-doc]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html
