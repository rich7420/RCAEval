## GNN+KAN Python Package 規劃與實施藍圖

### 目標
- 將現有 GNN+KAN 方法模組化為可安裝的 Python Package（`gnnkan`），提升可用性與可維護性。
- 設計通用輸入介面，支援 RCA（微服務）、分子化學、量子物理等跨領域應用。
- 制定重構、測試、文件與發佈流程，確保品質與穩定性。

---

### 範圍
- 程式來源：`RCAEval/e2e/gnnkan.py`、`RCAEval/gnn_kan_module/**`、相關工具與配置（`config.py`、`training.py` 等）。
- 文檔來源：`dataflow_GNN+KAN.md`、`GNN_KAN_架構圖.md`、`GNN-KAN技術架構與理論說明.md`（將整理進 docs）。
- 不在此階段：重度演算法重寫；僅針對公開 API 與穩定性進行必要重構。

---

### 套件命名與最低需求
- 名稱：`gnnkan`
- Python：>= 3.9
- 主要依賴：`torch`、`numpy`、`pandas`、`scikit-learn`
- 選配依賴（extras）：`torch_geometric`、`matplotlib`、`tensorboard`

---

### 目錄結構（目標）
```
gnnkan/
  __init__.py                 # 暴露公開 API
  core/
    __init__.py
    base_classes.py           # 抽象基類（FeatureProcessor、GraphConstructor、KANProcessor）
    data_interface.py         # 統一資料介面（StandardizedData, UnifiedDataInterface）
    models.py                 # GNNKANModel / SimplifiedGNNKAN 等
  kan_components/
    __init__.py
    kan_layers.py             # KAN 層（Simplified/Advanced/Compatible）
    high_capacity_stable_kan.py
    gradient_stabilizer.py
    feature_extraction.py
  processors/
    __init__.py
    feature_extractors.py     # 多模態抽取
    feature_processing.py     # ICA/kPCA/統計等
    graph_constructors.py     # 圖構建（簡化/智能）
    utils.py                  # 安全操作/矩陣處理
  training/
    __init__.py
    training.py               # 訓練、損失、早停
  e2e/
    __init__.py
    gnnkan.py                 # 端到端流程（gnn_kan_rca / GNNKANEndToEnd）
  config/
    __init__.py
    config.py                 # Simplified/HighCapacity/Fast + Factory
  examples/
    rca_example.py
    molecular_example.py
    quantum_example.py
  tests/
    test_models.py
    test_processors.py
    test_e2e.py
  docs/
    index.md
    architecture.md
    api_reference.md
pyproject.toml or setup.py
README.md
LICENSE
```

對應遷移（規劃）：
- `RCAEval/gnn_kan_module/**` → `gnnkan/{core,processors,kan_components,training,config}/**`
- `RCAEval/e2e/gnnkan.py` → `gnnkan/e2e/gnnkan.py`
- 文檔 Markdown 整理到 `gnnkan/docs/` 並生成 API 參考

---

### 公開 API 設計（Stable v0.1）
- `gnnkan.get_gnn_kan_rca_method()` → 回傳可直接呼叫之 RCA 介面
- `gnnkan.GNNKANEndToEnd`：
  - `run_rca(data, inject_time=None, dataset=None, with_bg=False, **kwargs)`
  - `configure(**kwargs)`
- `gnnkan.ConfigFactory.create_config(config_type='simplified', **overrides)`
- `gnnkan.optimize_gnn_kan_input(data, feature_method='ica', target_dim=64, **kwargs)`（保留作為快速入口）

API 穩定性原則：
- 僅暴露高層 API；內部類別與函數以次要穩定性標記（可能變動）。
- 以 `typing` 強化參數型別；必要處加入 `@overload`。

---

### 輸入通用性標準（重點）

#### 標準資料物件
- 使用既有 `StandardizedData`（`data_interface.py`）作為核心承載。
- 強化欄位：
  - `data: np.ndarray | Any`（視領域而定）
  - `feature_names: List[str]`
  - `node_names: List[str]`
  - `data_type: DataType`（metrics/logs/traces/multimodal/graph/unknown）
  - `original_shape: Tuple[int, ...]`
  - `metadata: Dict[str, Any]`（time_index、units、graph_schema 等）

#### 自動偵測與標準化
- `UnifiedDataInterface.standardize_input(data, data_type='auto', node_names=None, inject_time=None)`：
  - metrics：DataFrame/ndarray，形狀 (T, F)
  - logs：list[str]/DataFrame[text]
  - traces：結構化 span 列表或 DataFrame，必要欄位：`trace_id`, `span_id`, `parent_id`, `service`, `start`, `end`
  - multimodal：dict[str, Any]，包含上述任一型別
  - graph：`(node_features: np.ndarray, edge_index: np.ndarray|Tensor[, edge_weight])`

#### 跨領域擴展規範
- 分子化學（molecular）：
  - `data_type='graph'`，`metadata={'node_attr_schema': 'atom', 'edge_attr_schema': 'bond'}`
  - 支援 RDKit 轉換器（可選外掛）
- 量子物理（quantum）：
  - `data_type='graph'`，`metadata={'node_attr_schema': 'qubit', 'edge_attr_schema': 'coupling'}`
  - 特徵包含能級、期望值；時間序列可走 metrics 流程

#### 驗證與錯誤處理
- `UnifiedDataInterface.validate_data(data)`：
  - 缺值/Inf 清理、類型校驗、維度一致性、名稱對齊
- 一致的例外類型：`InvalidInputError`, `SchemaMismatchError`

---

### 重構與簡化（高可用性導向）

1) 統一/去重
- 移除重複或近似重複方法：
  - `psm_metric_processing` 在 `feature_processing.py` 與 `advanced_processors.py` 的同名/相似函式合併為單一實作（留一處並導出）。
  - 日誌處理 `extract_log_features` 僅保留 `processors/log_processors.py` 作為權威入口。
- `feature_extractors.py` 與 `processors/*` 功能角色切清：抽象/流程在 processors，樣例/輔助在 examples 或 utils。

2) API 一致性
- 函式命名統一：`extract_*_features` → 僅保留 processors 版本；e2e 僅調用公開入口。
- 類別方法返回值統一為 `(np.ndarray, List[str])` 或 `StandardizedData`。

3) 型別與 Docstring
- 對外 API、核心模組新增完整型別註解與 docstring（numpy-style 或 Google-style）。
- 公開類的 `__init__` 與 `forward/process` 明確參數與回傳。

4) 數值穩定性
- 預設啟用 `GradientStabilizer` 中較保守的正則與裁剪；
- 將 NaN/Inf 檢測集中於 `safe_tensor_operation` 與模型 `forward` 前置衛檢。

5) 日誌與錯誤
- 對外以 `logging` 提供 `INFO/DEBUG/WARN`；移除 `print`。
- 統一錯誤訊息字典與錯誤碼（便於前端或上層系統捕捉）。

6) 相依縮減
- 將 `torch_geometric` 移至 extras（非 RCA 最小路徑時不強制）。

---

### 測試計畫
- 單元測試（pytest）：
  - `test_data_interface.py`：自動偵測與標準化、驗證錯誤路徑
  - `test_processors.py`：ICA/kPCA、logs/traces/metrics 處理
  - `test_models.py`：前向傳播、正則項不爆 NaN
- 整合測試：
  - `test_e2e.py`：端到端 `run_rca` 在合成/小型資料上的 precision@k 檢核
- 效能測試：
  - 以固定隨機種子比較迭代時間、GPU 記憶體曲線
- 覆蓋率目標：>= 80%

---

### 文件化計畫
- 使用者指南：`README.md` + `docs/index.md`（快速開始、常見問題）
- 理論與架構：整理 `GNN_KAN_架構圖.md` 與 `GNN-KAN技術架構與理論說明.md` 至 `docs/architecture.md`
- API 參考：以 docstring 生成 `docs/api_reference.md`
- 範例：`examples/rca_example.py`, `examples/molecular_example.py`, `examples/quantum_example.py`

---

### 發佈與相容性
- 版本：`0.1.0`（RCA 穩定 API），`0.2.x` 加入 molecules/quantum adapter，`1.0.0` 穩定跨域
- 打包：`pyproject.toml`（PEP 517/518），產生 wheel 與 sdist
- CI：GitHub Actions（Python 3.9/3.10/3.11；Linux/macOS；CPU/GPU matrix 可選）
- 授權：整合 `LICENSES/`，主專案以 Apache-2.0（建議）或沿用目前 `LICENSE`

---

### 跨領域擴展方案
- 抽象 `DomainAdapter` 介面：
  - `to_standardized_data(raw) -> StandardizedData`
  - `domain_specific_features(std: StandardizedData) -> (np.ndarray, List[str])`
- 分子化學：
  - `MoleculeAdapter`（可選：RDKit 解析 SMILES → 圖結構）
  - KAN 取代 GNN MLP，任務如屬性預測/毒性分類
- 量子物理：
  - `QuantumAdapter`（Qiskit/PennyLane 可選）
  - 圖為耦合拓撲；特徵為期望值/能階/熵

---

### 風險與緩解
- 數值風險：KAN 基函數與樣條階導致不穩 → 穩定化 + 單元測試守門
- 效能風險：大圖訓練慢 → 批次化消息傳遞、邊采樣、低秩近似
- 相容風險：不同 Python/Torch 版本 → CI 覆蓋 + 上限約束
- 文件風險：文檔滯後 → PR 檢查要求更新 docs

---

### 里程碑與任務清單

里程碑 M1（1-2 週）
- [ ] 決定套件命名與授權
- [ ] 定稿公開 API 清單與 `__init__.py` 出口
- [ ] 整理 `UnifiedDataInterface` 輸入標準與例外類別
- [ ] 去重與統一函式（psm_metric_processing、extract_*_features 集中）
- [ ] 最小單元測試通過（CPU）

里程碑 M2（2-4 週）
- [ ] 完成目錄遷移至 `gnnkan/`（不改邏輯）
- [ ] 完整 docstring 與 `docs/*` 初版
- [ ] 整合 CI（lint+test）與 wheels 打包
- [ ] `examples/rca_example.py` 可跑通

里程碑 M3（4-8 週）
- [ ] 分子化學 `MoleculeAdapter` 原型
- [ ] 量子物理 `QuantumAdapter` 原型
- [ ] 效能優化（資料管線、消息傳遞）
- [ ] 發佈 `0.1.0`

---

### 驗收準則（Definition of Done）
- `pip install .` 後可 `import gnnkan` 並跑通 `examples/rca_example.py`
- `run_rca` 對多型輸入（metrics/logs/traces/multimodal/graph）均可執行
- 單元/整合測試通過，覆蓋率 ≥ 80%
- docs 齊備（Quickstart/Architecture/API/Examples）
- CI 綠燈且可產生 wheel

---

### 快速使用（草案）
```python
from gnnkan import GNNKANEndToEnd, ConfigFactory

config = ConfigFactory.create_config('high_capacity')
runner = GNNKANEndToEnd(config)
results = runner.run_rca(data_dict, inject_time=1000)
print(results)
```

---

### 備註（需要注意/待決）
- `torch_geometric` 是否作為核心或 extras：建議 extras
- 高容量模型預設超參數（grid_size、spline_order）需保守，避免 NaN
- 版本策略：小版本包含新 adapter；重大變更（API）才升主版本 