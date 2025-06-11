"""
E2E RCA Package
"""

# 導入主要函數 - 從實際存在的文件導入
try:
    # 從 gnnkan.py 導入主要函數（這個文件已經存在）
    from .gnnkan import gnn_kan_rca
    
    # 從 gnn_kan_module 導入配置和其他組件
    from RCAEval.gnn_kan_module import (
        SimplifiedGNNKANConfig as RCAConfig,
        MultiModalFeatureExtractor,
        SimplifiedGraphConstructor,
        GNNKANModel,
        train_gnn_kan_model
    )
    
    print("✓ E2E GNN-KAN modules loaded successfully")
    
    # 創建別名確保兼容性
    ModelConfig = SimplifiedGNNKANConfig = RCAConfig
    advanced_gnn_kan_rca = gnn_kan_rca
    run_gnn_kan_comparison = gnn_kan_rca
    run_gnn_kan_rca_pipeline = gnn_kan_rca
    
except ImportError as e:
    print(f"⚠️ Import error in E2E module: {e}")
    print("Creating fallback implementations...")
    
    # 導入正確的GNN-KAN實現
try:
    from .gnnkan import gnn_kan_rca, GNNKANEndToEnd
    
    # 創建別名確保向後兼容
    advanced_gnn_kan_rca = gnn_kan_rca
    
    def run_gnn_kan_comparison(data, inject_time=None, **kwargs):
        """運行GNN-KAN比較"""
        return gnn_kan_rca(data, inject_time, **kwargs)
    
    def run_gnn_kan_rca_pipeline(data, inject_time=None, **kwargs):
        """運行GNN-KAN RCA流水線"""
        return gnn_kan_rca(data, inject_time, **kwargs)
    
    # 從gnn_kan_module導入配置和組件
    from ..gnn_kan_module import (
        SimplifiedGNNKANConfig as RCAConfig,
        SimplifiedGNNKANConfig as ModelConfig,
        MultiModalFeatureExtractor,
        SimplifiedGraphConstructor,
        GNNKANModel,
        train_gnn_kan_model
    )
    
except ImportError as e:
    print(f"⚠️ GNN-KAN導入失敗: {e}")
    # 創建最小化的回退實現
    def gnn_kan_rca(data, inject_time=None, **kwargs):
        """回退實現"""
        print("⚠️ 使用回退實現")
        return {"adj": [], "node_names": [], "ranks": []}
    
    # 創建別名
    advanced_gnn_kan_rca = gnn_kan_rca
    run_gnn_kan_comparison = gnn_kan_rca  
    run_gnn_kan_rca_pipeline = gnn_kan_rca
    
    # 創建空的配置類
    class RCAConfig:
        def __init__(self):
            self.target_feature_dim = 64
            self.epochs = 30
    
    ModelConfig = RCAConfig
    MultiModalFeatureExtractor = None
    SimplifiedGraphConstructor = None
    GNNKANModel = None
    train_gnn_kan_model = None
    train_gnn_kan_model = None

# 確保向後兼容性 - 導出所有主要函數
__all__ = [
    'gnn_kan_rca',
    'run_gnn_kan_comparison',
    'advanced_gnn_kan_rca', 
    'run_gnn_kan_rca_pipeline',
    'RCAConfig',
    'ModelConfig',
    'MultiModalFeatureExtractor',
    'SimplifiedGraphConstructor',
    'GNNKANModel',
    'train_gnn_kan_model'
]




