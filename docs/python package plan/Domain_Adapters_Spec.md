## GNN+KAN 跨領域 Adapter 規格（Domain Adapters Spec）

### 目的
提供統一的領域適配介面，將分子化學、量子物理等圖結構/多模態資料轉為 `StandardizedData`，並對接既有 processors 與 GNN+KAN 模型。

---

### 共用介面：`DomainAdapter`
- 方法：
  - `to_standardized_data(raw) -> StandardizedData`
  - `domain_specific_features(std: StandardizedData) -> Tuple[np.ndarray, List[str]]`
- 原則：
  - 嚴格填寫 `metadata.domain` 與 schema 描述
  - 回傳之特徵矩陣需對齊 `node_names`

---

### 分子化學：`MoleculeAdapter`
- 依賴（extras）：`rdkit`
- 輸入：SMILES 字串、RDKit Mol、或已存在的分子圖（節點/邊屬性）
- 轉換：
  - 節點（原子）特徵：原子序數、價態、雜化、芳香性、部分電荷
  - 邊（鍵）特徵：鍵種類、階數、是否芳香
  - 形成 `(node_features, edge_index[, edge_weight])`
- metadata：
  - `{'domain': 'molecular', 'node_attr_schema': 'atom', 'edge_attr_schema': 'bond'}`
- 特徵擴充（可選）：環結構、拓撲指標（如 betweenness）、訊息傳遞前置特徵

---

### 量子物理：`QuantumAdapter`
- 依賴（extras）：`qiskit` 或 `pennylane`
- 輸入：量子電路/哈密頓量/耦合圖；或實驗量測資料（時間序列）
- 轉換：
  - 節點（量子比特）特徵：期望值、佔據數、局域可觀測量
  - 邊（耦合）特徵：耦合強度、相互作用型別、閘深
  - 將時間序列特徵（metrics 流程）與拓撲特徵（graph 流程）融合
- metadata：
  - `{'domain': 'quantum', 'node_attr_schema': 'qubit', 'edge_attr_schema': 'coupling'}`

---

### 與處理器整合
- 標準化後可直接進入：
  - `processors/feature_processing.py`（ICA/kPCA/統計）
  - `processors/graph_constructors.py`（若需再構建或融合圖）
  - `core/models.py` 中 `SimplifiedGNNKAN` / `HighCapacityGNNKANEncoder`

---

### 使用範例（概念性）
```python
adapter = MoleculeAdapter()
std = adapter.to_standardized_data(smiles_list)
X, feat_names = adapter.domain_specific_features(std)

from gnnkan import GNNKANEndToEnd, ConfigFactory
cfg = ConfigFactory.create_config('simplified')
runner = GNNKANEndToEnd(cfg)
res = runner.run_rca({'graph': std}, with_bg=False)
```

---

### 測試建議
- 以 QM9/ESOL 小樣本驗證 `MoleculeAdapter` 端到端可跑
- 以 2D Ising/簡單耦合圖測 `QuantumAdapter`（合成資料即可）
- 檢查：節點數/特徵對齊、NaN/Inf、precision@k（如任務定義） 