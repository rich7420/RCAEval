# GNN+KAN 根因分析方法實現說明

## 1. 方法概述

本實現使用 **Kolmogorov-Arnold Networks (KAN) 取代傳統 GNN 中的 MLP 層**，構建了一個創新的根因分析框架。核心目標是通過 KAN 的可解釋性和表達能力，提升 GNN 在複雜系統根因分析中的準確率。

### 1.1 核心創新點

- **KAN 替代 MLP**：使用基於樣條函數的 KAN 層替代傳統的多層感知機
- **多模態特徵融合**：整合時間序列、日誌、拓撲等多種數據源
- **梯度穩定化**：實現穩定的 KAN 訓練過程
- **模組化設計**：完全模組化的架構便於擴展和維護

## 2. 模組化架構

### 2.1 入口點：`e2e/gnnkan.py`

這是整個系統的主入口點，提供：

```python
def gnn_kan_rca(data, inject_time=None, dataset=None, with_bg=False, **kwargs):
    """
    主要的 GNN-KAN RCA 方法
    
    Args:
        data: 輸入數據 (multimodal 或 單一模態)
        inject_time: 注入時間點
        dataset: 數據集名稱
        with_bg: 是否包含背景數據
        **kwargs: 其他參數
    
    Returns:
        dict: 包含 adj, node_names, ranks 的結果
    """
```

**主要功能流程**：
1. 配置管理 (`SimplifiedGNNKANConfig`)
2. 特徵提取 (`MultiModalFeatureExtractor`)
3. 圖構建 (`SimplifiedGraphConstructor`)
4. 模型初始化 (`GNNKANModel`)
5. 模型訓練 (`train_gnn_kan_model`)
6. PageRank 分析和結果輸出

### 2.2 依賴模組：`gnn_kan_module/`

#### 2.2.1 配置管理 (`config.py`)

```python
class SimplifiedGNNKANConfig:
    """簡化的GNN-KAN配置類 - 專注核心功能"""
    
    def __init__(self):
        # 🎯 核心KAN架構 - 保持用KAN取代MLP的核心價值
        self.input_dim = 64           # 簡化輸入維度
        self.hidden_dims = [128, 64]  # 簡化為2層隱藏層
        self.output_dim = 32          # 簡化輸出維度
        
        # 🔑 KAN設置 - 保持核心表達能力
        self.kan_grid_size = 5
        self.kan_spline_order = 3
        self.num_gnn_layers = 2
```

#### 2.2.2 特徵提取器 (`feature_extractors.py`)

**多模態特徵提取**：
- 時間序列特徵：STL 分解、滑動窗口統計
- 日誌特徵：錯誤模式、關鍵詞提取
- 拓撲特徵：圖結構分析、度中心性
- PSM 指標：服務性能指標

```python
class MultiModalFeatureExtractor:
    """多模態特徵提取器"""
    
    def extract_features(self, data, inject_time=None):
        """
        從多模態數據中提取特徵
        
        Returns:
            features: 統一的特徵矩陣
            node_names: 節點名稱列表
        """
```

#### 2.2.3 圖構建器 (`graph_constructors.py`)

```python
class SimplifiedGraphConstructor:
    """簡化圖構建器 - 高效準確的圖構建"""
    
    def build_graph(self, features, node_names):
        """
        根據特徵構建圖結構
        
        Returns:
            edge_index: 邊索引 [2, num_edges]
            edge_weights: 邊權重
        """
```

#### 2.2.4 KAN 組件 (`kan_components/`)

**核心 KAN 層實現**：

```python
class OptimizedGNNKANEncoder(nn.Module):
    """優化的 GNN-KAN 編碼器"""
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        # KAN 層替代傳統 MLP
        self.kan_layers = nn.ModuleList([
            UltraFastKANLayer(input_dim, hidden_dims[0]),
            UltraFastKANLayer(hidden_dims[0], hidden_dims[1]),
            UltraFastKANLayer(hidden_dims[1], output_dim)
        ])
```

**梯度穩定器**：
```python
class GradientStabilizer:
    """梯度穩定化器 - 確保 KAN 訓練穩定性"""
    
    def apply_gradient_clipping(self, model, clip_type='norm'):
        """應用梯度裁剪"""
    
    def check_numerical_stability(self, model):
        """數值穩定性檢查"""
```

#### 2.2.5 GNN-KAN 模型 (`models.py`)

```python
class GNNKANModel(nn.Module):
    """GNN-KAN 模型 - 結合 GNN 和 KAN 的優勢"""
    
    def __init__(self, config, num_nodes):
        # 特徵投影層
        self.feature_projection = nn.Linear(config.target_feature_dim, config.input_dim)
        
        # 🎯 核心創新：使用 KAN 編碼器替代傳統 MLP
        self.kan_encoder = OptimizedGNNKANEncoder(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim
        )
        
        # 鄰接矩陣預測層
        self.adj_predictor = nn.Linear(config.output_dim, num_nodes)
```

#### 2.2.6 訓練模組 (`training.py`)

```python
def train_gnn_kan_model(model, node_features, edge_index, config):
    """
    訓練 GNN-KAN 模型
    
    核心特點：
    1. 梯度穩定化：使用 GradientStabilizer 確保 KAN 訓練穩定
    2. 自適應學習率：動態調整學習率
    3. 早停機制：防止過擬合
    4. 設備自適應：自動 CPU/GPU 切換
    """
```

## 3. KAN 替代 MLP 的技術實現

### 3.1 傳統 GNN 中的 MLP 層

```python
# 傳統方式
class TraditionalGNN(nn.Module):
    def __init__(self):
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
```

### 3.2 KAN 替代實現

```python
# 創新方式：使用 KAN
class GNNKANModel(nn.Module):
    def __init__(self):
        self.kan_encoder = OptimizedGNNKANEncoder(
            input_dim=input_dim,
            hidden_dims=[hidden_dim],
            output_dim=output_dim
        )
```

### 3.3 KAN 層的核心優勢

1. **可解釋性**：基於樣條函數的激活函數可視化
2. **表達能力**：理論上可以學習任意連續函數
3. **參數效率**：相比深層 MLP 需要更少參數
4. **梯度流暢**：樣條函數的平滑性有助於梯度傳播

## 4. 數據流和交互邏輯

### 4.1 主要數據流

```
輸入數據 (data) 
    ↓
[MultiModalFeatureExtractor]
    ↓
統一特徵矩陣 (features)
    ↓
[SimplifiedGraphConstructor]
    ↓
圖結構 (edge_index, edge_weights)
    ↓
[GNNKANModel + KAN編碼器]
    ↓
節點嵌入 + 鄰接矩陣
    ↓
[PageRank 分析]
    ↓
根因排名結果
```

### 4.2 參數傳遞

- **配置統一**：所有模組共用 `SimplifiedGNNKANConfig`
- **設備管理**：自動 CPU/GPU 設備檢測和切換
- **維度對齊**：自動特徵維度對齊到 `target_feature_dim`

### 4.3 錯誤處理和回退機制

- **CUDA 錯誤**：自動回退到 CPU 訓練
- **數值不穩定**：梯度穩定化器介入
- **內存不足**：簡化模型結構

## 5. 核心技術特點

### 5.1 KAN 的數學基礎

KAN 使用可學習的單變量函數替代傳統的線性變換：

```
Traditional MLP: f(x) = σ(Wx + b)
KAN: f(x) = Σ φ(x_i) where φ is learnable spline function
```

### 5.2 梯度穩定化技術

```python
class GradientStabilizer:
    def compute_total_regularization_loss(self, model, base_loss):
        # L1 正則化：促進稀疏性
        l1_loss = self.l1_lambda * sum(param.abs().sum() for param in model.parameters())
        
        # 熵正則化：防止過擬合
        entropy_loss = self.entropy_lambda * self.compute_entropy_regularization(model)
        
        return base_loss + l1_loss + entropy_loss
```

### 5.3 多模態特徵融合

```python
def enhanced_feature_fusion(trace_features, metric_features, log_features, 
                          fusion_method='adaptive', target_dim=64):
    """
    智能自適應特徵融合
    - adaptive: 基於數據質量自適應權重
    - concatenate: 簡單拼接
    - weighted: 加權融合
    - attention: 注意力機制融合
    """
```

## 6. 性能優化策略

### 6.1 計算優化

- **向量化操作**：避免循環，使用批量操作
- **記憶體管理**：及時清理 GPU 記憶體
- **梯度累積**：處理大批量數據

### 6.2 數值穩定性

- **梯度裁剪**：防止梯度爆炸
- **自適應正則化**：動態調整正則化強度
- **早停機制**：防止過擬合

### 6.3 設備自適應

```python
# 自動設備選擇
device = 'cuda' if config.use_cuda and torch.cuda.is_available() else 'cpu'

# 錯誤回退機制
try:
    model = model.cuda()
except RuntimeError as e:
    print(f"CUDA 失敗: {e}，回退到 CPU")
    model = model.cpu()
```

## 7. 實驗驗證目標

### 7.1 主要目標

**證明 KAN 替代 MLP 在根因分析中的有效性**：
- 準確率提升：相比傳統 GNN 方法
- 可解釋性增強：可視化學習到的激活函數
- 參數效率：更少的參數達到更好的效果

### 7.2 評估指標

- **準確率**：Top-K 根因識別準確率
- **AUC**：異常檢測的 AUC 值  
- **訓練效率**：收斂速度和穩定性
- **可解釋性**：激活函數的可視化分析

## 8. 使用方式

### 8.1 基本使用

```python
from RCAEval.e2e.gnnkan import gnn_kan_rca

# 準備數據
data = {
    'metrics': pd.DataFrame(...),
    'traces': pd.DataFrame(...),
    'logs': pd.DataFrame(...)
}

# 執行根因分析
result = gnn_kan_rca(
    data=data,
    inject_time=fault_injection_time,
    epochs=50,
    **other_params
)

# 獲取結果
top_causes = result['ranks'][:5]  # 前5個根因
adjacency_matrix = result['adj']  # 學習到的鄰接矩陣
```

### 8.2 高級配置

```python
from RCAEval.gnn_kan_module import SimplifiedGNNKANConfig

# 創建自定義配置
config = SimplifiedGNNKANConfig()
config.epochs = 100
config.kan_grid_size = 10
config.learning_rate = 1e-5

# 使用自定義配置
result = gnn_kan_rca(data, config=config)
```

## 9. 總結

本實現成功實現了 **KAN 替代 GNN 中 MLP 層** 的創新方法，通過模組化設計確保了：

1. **功能完整性**：從數據輸入到結果輸出的完整流程
2. **技術創新性**：KAN 的有效集成和穩定訓練
3. **可擴展性**：模組化架構便於功能擴展
4. **實用性**：自動錯誤處理和設備適配

### 9.1 模組化完成狀態

✅ **入口點整合**：`e2e/gnnkan.py` 作為唯一主入口點  
✅ **依賴模組**：所有功能模組化到 `gnn_kan_module/`  
✅ **Import 修正**：消除循環導入和重複定義  
✅ **錯誤處理**：完善的錯誤處理和回退機制  
✅ **設備適配**：自動 CPU/GPU 切換  
✅ **參數統一**：統一的配置管理系統  

### 9.2 核心驗證目標

本框架的主要目標是**證明用 KAN 取代 GNN 中的 MLP 層是有效的方法（準確率比其他方法高）**。

關鍵驗證點：
- **準確率提升**：KAN 相比傳統 MLP 的性能優勢
- **參數效率**：更少參數達到更好效果
- **可解釋性**：KAN 激活函數的可視化分析
- **穩定性**：梯度穩定化技術確保訓練穩定

### 9.3 使用建議

建議在其他設備上測試時：
1. 確保 PyTorch 環境正確配置
2. 檢查 CUDA 可用性（如需 GPU 加速）
3. 使用提供的配置類進行參數調整
4. 監控訓練過程中的穩定性指標

這個框架為根因分析提供了一個強大而靈活的工具，同時驗證了 KAN 在圖神經網絡中的應用潛力。
