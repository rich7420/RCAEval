## GNN+KAN 重構檢核清單（Refactor Checklist）

### 目標
提升高可用性：API 一致、輸入通用、數值穩定、依賴精簡、測試完善、文件齊備。

---

### A. 去重與單一權威（Single Source of Truth）
- [ ] `psm_metric_processing` 僅保留一處實作（統一於 `processors/feature_processing.py`），其他處改為導入
- [ ] `extract_log_features` 僅保留於 `processors/log_processors.py` 導出；移除/轉發在 `feature_extractors.py` 的重複
- [ ] `enhanced_trace_processing` 僅保留於 `processors/trace_processors.py`
- [ ] `create_model_with_config` 僅保留於 `training/training.py` 或 `core/models.py`（擇一），其餘改為調用

### B. API 命名與返回一致
- [ ] 函式命名統一：`extract_*_features`（metrics/logs/traces）
- [ ] 所有 processors `fit/transform/fit_transform/process` 返回 `(np.ndarray, List[str])`
- [ ] e2e 僅調用公開入口，不直接使用內部實作

### C. 型別與文件化
- [ ] 對外 API 全面加上 `typing` 註解
- [ ] 核心類/函數撰寫完整 docstring（範例/參數/返回/例外）
- [ ] Public API 清單集中於 `gnnkan/__init__.py`

### D. 數值穩定性
- [ ] 預設啟用 `GradientStabilizer` 保守超參數（l1_lambda、entropy_lambda、grad_clip）
- [ ] 在模型 `forward` 前加入 `validate_inputs` 與 NaN/Inf 檢查
- [ ] 訓練中記錄梯度範數與裁剪比率（logging）
- [ ] 覆寫可能產生 NaN 的基函數或啟用 fallback 分支

### E. 依賴與選配
- [ ] 將 `torch_geometric` 移至 extras：`pip install gnnkan[geom]`
- [ ] RDKit/Qiskit/PennyLane 等為 `molecular/quantum` extras
- [ ] 最小核心依賴：torch/numpy/pandas/scikit-learn

### F. 測試
- [ ] `test_data_interface.py`：自動偵測/標準化/驗證例外
- [ ] `test_processors.py`：ICA/kPCA、logs/traces/metrics 處理 I/O 契約
- [ ] `test_models.py`：前向傳播無 NaN/Inf、正則項有限
- [ ] `test_e2e.py`：跑通 `run_rca` 合成資料，檢查 precision@k 邏輯
- [ ] 覆蓋率 ≥ 80%

### G. 日誌與錯誤
- [ ] 統一使用 `logging`（不使用 print）
- [ ] 統一定義例外類別與錯誤碼；記錄於文件
- [ ] 關鍵路徑（標準化、訓練 epoch）提供 `INFO/DEBUG` 訊息

### H. CI 與版本
- [ ] GitHub Actions：3.9/3.10/3.11，Linux/macOS
- [ ] wheels + sdist 產物驗證
- [ ] Semantic Versioning；初版 `0.1.0`

### I. 檔案遷移（不改邏輯）
- [ ] `RCAEval/gnn_kan_module/**` → `gnnkan/{core,processors,kan_components,training,config}/**`
- [ ] `RCAEval/e2e/gnnkan.py` → `gnnkan/e2e/gnnkan.py`
- [ ] 文檔匯總進 `gnnkan/docs/`（本階段先連結現有）

### J. 驗收標準（DoD）
- [ ] `pip install -e .` 可用，`examples/rca_example.py` 跑通
- [ ] 多模態/跨域輸入能標準化並進入 pipeline
- [ ] 測試綠燈、CI 綠燈、產生 wheel 