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
        #  大幅提升KAN表達能力
        self.kan_grid_size = 20              # 從8大幅增加到20
        self.kan_spline_order = 5            # 從3增加到5
        self.kan_num_basis = 12              # 從4增加到12
        self.learnable_activation = True
        
        # 🎯 基函數選擇配置 - Single source of truth for basis function selection
        self.basis_function = 'chebyshev'    # Default to chebyshev for backward compatibility
        self.basis_kwargs = {}               # Additional parameters for specific basis functions
        
        #  增加模型容量和深度
        self.input_dim = 128                 # 從64增加到128
        self.hidden_dims = [128, 96, 64]     # 從[32,16]增加到[128,96,64]
        self.output_dim = 96                 # 從16增加到96
        self.num_gnn_layers = 4              # 從2增加到4
        self.dropout = 0.1                   # 從0.2減少到0.1
        
        #  優化特徵處理
        self.feature_method = 'enhanced_ica' # 使用更強的特徵提取
        self.target_feature_dim = 128        # 從64增加到128
        
        # 🎯 調整訓練策略
        self.learning_rate = 2e-4            # 從1e-3調整到2e-4
        self.weight_decay = 5e-6             # 從1e-4減少到5e-6
        self.num_epochs = 400                # 從100大幅增加到400
        self.batch_size = 8                  # 從32減少到8，提高穩定性
        self.patience = 60                   # 從25增加到60
        self.min_delta = 1e-6                # 從1e-5減少到1e-6
        
        # 🎯 優化圖構建
        self.similarity_threshold = 0.2      # 從0.5大幅降低到0.2
        self.max_edges_per_node = 15         # 從4大幅增加到15
        
        #  穩定性配置
        self.gradient_clip_norm = 0.5        # 從1.0減少到0.5
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