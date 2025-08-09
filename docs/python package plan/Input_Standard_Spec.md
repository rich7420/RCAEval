## GNN+KAN 輸入通用性規格（Input Standardization Spec）

### 目的
定義跨領域、跨模態可重用的輸入標準，確保使用者以最小轉換成本將資料送入 GNN+KAN pipeline，同時保留強型別與嚴格驗證以提升可靠性。

---

### 標準資料物件：`StandardizedData`
- 位置：`gnnkan/core/data_interface.py`
- 欄位：
  - `data: np.ndarray | Any`（核心資料；對 graph 可為 tuple）
  - `feature_names: List[str]`
  - `node_names: List[str]`
  - `data_type: DataType` ∈ {`metrics`, `logs`, `traces`, `multimodal`, `graph`, `unknown`}
  - `original_shape: Tuple[int, ...]`
  - `metadata: Dict[str, Any]`（可包含：`time_index`, `units`, `service_graph`, `graph_schema`, `domain`, `inject_time` 等）
- 行為：
  - 在 `__post_init__` 自動處理 NaN/Inf，並對 `feature_names` 長度做基本一致性修正。

---

### 自動偵測與標準化：`UnifiedDataInterface.standardize_input`
- 介面：`standardize_input(data, data_type='auto', node_names=None, inject_time=None) -> StandardizedData`
- 偵測策略（`data_type='auto'`）：
  - metrics：`pd.DataFrame`/`np.ndarray`（形狀 (T, F) 或 (F,)）
  - logs：`List[str]` 或 `DataFrame[text]`
  - traces：包含 `trace_id, span_id, parent_id, service, start, end` 欄位的結構
  - multimodal：`Dict[str, Any]`，鍵包含 {metrics, logs, traces} 子集
  - graph：二元組/三元組 `(node_features, edge_index[, edge_weight])`

- 標準化結果：
  - metrics：`data.shape -> (T, F)`，`feature_names` 取自欄名，`node_names` 由欄名萃取服務名（若可），`metadata={'time_index': index}`
  - logs：`data.shape -> (N, )` 或稀疏向量；`feature_names` 為 token/特徵名；`node_names` 視 log 來源而定
  - traces：轉為節點/邊描述的中間表或特徵矩陣（由 processors 處理）；保留 `service_graph` 於 metadata
  - multimodal：各子模態分別標準化，於 metadata 合併記錄
  - graph：
    - `node_features: np.ndarray (num_nodes, feat_dim)`
    - `edge_index: np.ndarray|torch.Tensor (2, num_edges)`
    - `edge_weight: Optional[np.ndarray|torch.Tensor (num_edges,)]`
    - `node_names: List[str]` 必須提供或可由資料推斷

---

### 驗證規則：`UnifiedDataInterface.validate_data`
- 缺值處理：NaN → 0.0；Inf → {pos: 1.0, neg: -1.0}
- 維度一致性：
  - metrics：`len(feature_names) == F`；時間索引可選
  - graph：`edge_index.max() < num_nodes`；`edge_weight` 長度與邊數一致
- 名稱一致：`len(node_names) == num_nodes`（若可辨識）
- 失敗時擲出：`InvalidInputError`, `SchemaMismatchError`

---

### 處理器 I/O 契約
- 輸入：`StandardizedData` 或原始資料（將透過 `standardize_input` 包裝）
- 輸出：`Tuple[np.ndarray, List[str]]`（特徵矩陣, 特徵名）
- 多模態：回傳對齊後之特徵矩陣（必要時使用 `safe_feature_alignment`）

---

### 跨領域映射（Domain Metadata）
- 分子化學（molecular）：
  - `data_type='graph'`
  - `metadata={'domain': 'molecular', 'node_attr_schema': 'atom', 'edge_attr_schema': 'bond', 'units': {...}}`
  - 節點特徵：原子序數、雜化、芳香性等；邊特徵：鍵種類、階數
- 量子物理（quantum）：
  - `data_type='graph'`
  - `metadata={'domain': 'quantum', 'node_attr_schema': 'qubit', 'edge_attr_schema': 'coupling', 'units': {...}}`
  - 節點特徵：期望值/佔據數/局域可觀測量；邊特徵：耦合強度/閘深

---

### 相容性與遞移策略
- v0.1：
  - 高階 API 接受原始 Python 結構（DataFrame/Dict），內部自動標準化
  - 推薦使用 `StandardizedData` 以獲得最佳提示與錯誤訊息
- v0.2：
  - 對所有 processors 要求顯式支援 `StandardizedData`，原始資料通道標記為 deprecated

---

### 例外與錯誤訊息（建議）
- `InvalidInputError(code='shape_mismatch', message='Expected shape (T,F), got ...')`
- `SchemaMismatchError(code='missing_columns', message='Required columns: ...')`
- `DomainAdapterError(code='conversion_failed', message='RDKit parse failed: ...')`

---

### 最佳實務
- 明確提供 `node_names` 與 `feature_names`
- 對長時間序列先做降頻/對齊，避免巨量記憶體
- 多模態輸入時，確保各模態時間窗一致或提供對齊策略參數（`align='pad'|'interp'`） 