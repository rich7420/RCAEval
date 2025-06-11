"""
GNN-KAN Module: Configuration
統一的配置管理模組
"""

import torch
import numpy as np


class SimplifiedGNNKANConfig:
    """簡化的GNN-KAN配置類 - 專注核心功能"""
    
    def __init__(self):
        # 🎯 核心KAN架構 - 保持用KAN取代MLP的核心價值
        self.input_dim = 64           # 簡化輸入維度
        self.hidden_dims = [128, 64]  # 簡化為2層隱藏層
        self.output_dim = 32          # 簡化輸出維度
        
        # 🔑 KAN設置 - 增強核心表達能力
        self.kan_grid_size = 8        # 從5增加到8，提升非線性表達
        self.kan_spline_order = 3
        self.adaptive_spline_order = True  # 新增：自適應樣條階數
        self.learnable_edges = True   # 新增：可學習的邊權重
        self.num_gnn_layers = 2       # 簡化為2層
        self.dropout = 0.1
        
        # 🎯 訓練參數 - 實用導向
        self.epochs = 30              # 減少訓練時間
        self.num_epochs = 30          # 別名，確保兼容性
        self.batch_size = 16
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        
        # 🛡️ 梯度穩定 - 簡化穩定性檢查
        self.gradient_clip_norm = 1.0
        self.use_gradient_stabilizer = True
        self.stability_check_frequency = 50  # 從20增加到50，減少檢查頻率
        self.base_l1_lambda = 0.001
        self.base_entropy_lambda = 0.001
        self.base_learning_rate = 1e-4
        self.warmup_epochs = 5
        
        # 🔧 特徵提取 - 簡化特徵融合機制
        self.window_size = 10         # 簡化窗口大小
        self.step_size = 1
        self.use_dla = True
        self.max_log_features = 20    # 從30減少到20，進一步減少噪音
        self.target_feature_dim = 64  # 簡化目標維度
        self.fusion_method = 'simple_concat'  # 簡化融合方法
        self.use_attention_fusion = False     # 移除複雜注意力機制
        self.use_stl_decomposition = False    # 簡化STL處理
        self.use_kll_processing = False       # 簡化KLL處理
        self.use_multimodal_fusion = False    # 移除過度複雜的多模態融合
        
        # PCA設置
        self.use_pca = True
        self.pca_components = 32      # 減少主成分數量
        
        # 圖構建 - 簡化但保持連通性
        self.similarity_threshold = 0.3
        self.max_edges_per_node = 5
        self.use_self_loops = True
        
        # 硬體設置
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        
        # 輸出
        self.top_k_results = 10       # 減少輸出數量
        
        # 高級功能開關 - 專注核心功能
        self.use_intelligent_graph = True
        self.use_advanced_training = False
        self.use_psm_processing = False
        
        # 早停設置
        self.early_stopping_patience = 15
        self.early_stopping_min_delta = 1e-6
        self.patience = 15  # 訓練早停耐心值
        
        # STL分解設置（向後兼容）
        self.stl_seasonal = 7
        
        # KLL設置（向後兼容）
        self.kll_k = 128


class AdvancedGNNKANConfig(SimplifiedGNNKANConfig):
    """高級GNN-KAN配置類 - 包含所有先進功能"""
    
    def __init__(self):
        super().__init__()
        
        # 🚀 高級架構設置
        self.input_dim = 128
        self.hidden_dims = [256, 128, 64]
        self.output_dim = 64
        self.num_gnn_layers = 3
        
        # 🎯 高級訓練設置
        self.epochs = 100
        self.batch_size = 32
        self.learning_rate = 5e-5
        
        # 🔬 高級特徵設置
        self.target_feature_dim = 128
        self.max_log_features = 100
        self.pca_components = 64
        self.fusion_method = 'attention'
        
        # 📊 高級圖設置
        self.max_edges_per_node = 10
        self.similarity_threshold = 0.2
        
        # ✨ 啟用高級功能
        self.use_intelligent_graph = True
        self.use_advanced_training = True
        self.use_psm_processing = True
        
        # 🔧 高級處理設置
        self.window_size = 20
        self.step_size = 5
        
        # 結果設置
        self.top_k_results = 15


class ConfigFactory:
    """配置工廠類 - 根據需求創建適當的配置"""
    
    @staticmethod
    def create_config(config_type='simplified', **kwargs):
        """創建配置對象"""
        config_map = {
            'simplified': SimplifiedGNNKANConfig,
            'advanced': AdvancedGNNKANConfig
        }
        
        if config_type not in config_map:
            print(f"⚠️ Unknown config type '{config_type}', using 'simplified'")
            config_type = 'simplified'
        
        config = config_map[config_type]()
        
        # 應用額外參數
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config


# 向後兼容性
GNNKANConfig = SimplifiedGNNKANConfig