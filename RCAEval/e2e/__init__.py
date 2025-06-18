"""
E2E RCA Package - 清理版本
統一導入路徑，消除重複定義
"""

# 🎯 主要函數導入 - 從gnnkan.py（主進入點）
try:
    from .gnnkan import gnn_kan_rca, GNNKANEndToEnd
    
    # 🔧 從gnn_kan_module導入依賴組件
    from ..gnn_kan_module import (
        SimplifiedGNNKANConfig,
        HighCapacityGNNKANConfig, 
        FastGNNKANConfig,
        MultiModalFeatureExtractor,
        SimplifiedGraphConstructor,
        GNNKANModel,
        train_gnn_kan_model
    )
    
    print("✅ E2E GNN-KAN模組載入成功 - 主進入點：gnnkan.py")
    
    # 🔧 創建兼容性別名（避免重複定義）
    RCAConfig = SimplifiedGNNKANConfig
    ModelConfig = SimplifiedGNNKANConfig
    advanced_gnn_kan_rca = gnn_kan_rca
    
    def run_gnn_kan_comparison(data, inject_time=None, **kwargs):
        """運行GNN-KAN比較 - 重定向到主函數"""
        return gnn_kan_rca(data, inject_time, **kwargs)
    
    def run_gnn_kan_rca_pipeline(data, inject_time=None, **kwargs):
        """運行GNN-KAN RCA流水線 - 重定向到主函數"""
        return gnn_kan_rca(data, inject_time, **kwargs)
    
    # 載入成功
    _import_success = True
    
except ImportError as e:
    print(f"❌ GNN-KAN導入失敗: {e}")
    print("🔧 創建最小回退實現...")
    
    # 🔧 最小回退實現
    def gnn_kan_rca(data, inject_time=None, **kwargs):
        """最小回退實現"""
        print("⚠️ 使用回退實現 - 請檢查gnn_kan_module安裝")
        return {
            "adj": [],
            "node_names": [], 
            "ranks": [],
            "error": "Import failed, using fallback"
        }
    
    # 創建別名
    advanced_gnn_kan_rca = gnn_kan_rca
    run_gnn_kan_comparison = gnn_kan_rca
    run_gnn_kan_rca_pipeline = gnn_kan_rca
    
    # 空配置類
    class RCAConfig:
        def __init__(self):
            self.target_feature_dim = 64
            self.epochs = 30
            self.feature_method = 'ica'
    
    ModelConfig = RCAConfig
    SimplifiedGNNKANConfig = RCAConfig
    HighCapacityGNNKANConfig = RCAConfig
    FastGNNKANConfig = RCAConfig
    
    # 空組件
    MultiModalFeatureExtractor = None
    SimplifiedGraphConstructor = None
    GNNKANModel = None
    train_gnn_kan_model = None
    GNNKANEndToEnd = None
    
    _import_success = False

# 🎯 統一導出清單
__all__ = [
    # 主要函數
    'gnn_kan_rca',
    'GNNKANEndToEnd',
    
    # 兼容性函數
    'run_gnn_kan_comparison',
    'advanced_gnn_kan_rca', 
    'run_gnn_kan_rca_pipeline',
    
    # 配置類
    'RCAConfig',
    'ModelConfig',
    'SimplifiedGNNKANConfig',
    'HighCapacityGNNKANConfig',
    'FastGNNKANConfig',
    
    # 核心組件
    'MultiModalFeatureExtractor',
    'SimplifiedGraphConstructor',
    'GNNKANModel',
    'train_gnn_kan_model'
]




