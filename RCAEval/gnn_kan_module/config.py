"""
簡化的 GNN-KAN 配置模組
移除過多可選項，專注核心功能
"""

import torch
import numpy as np


class GNNKANConfig:
    """
    簡化的 GNN-KAN 配置 - 移除過多可選項
    專注核心功能，確保方法結構清晰
    """
    
    def __init__(self):
        # 🎯 核心 KAN 配置 / Core KAN Configuration
        self.kan_grid_size = 8              # 增加KAN B-spline 網格大小提升表達能力 / Increase KAN B-spline grid size for better expressiveness
        self.kan_spline_order = 3           # 樣條階數 / Spline order
        self.learnable_activation = True    # 可學習激活函數 / Learnable activation functions
        self.kan_num_basis = 4              # KAN基函數數量 / Number of KAN basis functions
        
        # 🔧 特徵處理配置
        self.feature_method = 'enhanced_ica'  # 使用增強版 ICA
        self.target_feature_dim = 64        # 目標特徵維度
        
        # 🎯 GNN-KAN 架構配置
        self.input_dim = 64
        self.hidden_dims = [32, 16]         # 簡化的 2 層結構
        self.output_dim = 16
        self.num_gnn_layers = 2
        self.dropout = 0.2
        
        # 🚀 訓練配置
        self.learning_rate = 1e-3           # 較高學習率確保收斂
        self.weight_decay = 1e-4
        self.num_epochs = 100               # 較少輪數但更有效
        self.batch_size = 32
        self.patience = 25                  # 早停耐心值
        self.min_delta = 1e-5               # 最小改進閾值
        
        # 🔧 圖構建配置 / Graph Construction Configuration
        self.similarity_threshold = 0.5     # 提高閾值降低圖密度 / Increase threshold to reduce graph density
        self.max_edges_per_node = 4         # 減少每節點最大邊數 / Reduce max edges per node
        
        # 🔧 穩定性配置
        self.gradient_clip_norm = 1.0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_cuda = torch.cuda.is_available()
    
    def update_for_kan_purity(self):
        """更新配置以確保 KAN 純粹性"""
        pass  # 簡化版本不需要額外配置
        
    def get_device(self):
        """獲取計算設備"""
        return self.device


# 便捷的配置創建函數
def create_config(**kwargs):
    """創建配置並應用自定義參數"""
    config = GNNKANConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
    

# 兼容性別名
SimplifiedGNNKANConfig = GNNKANConfig