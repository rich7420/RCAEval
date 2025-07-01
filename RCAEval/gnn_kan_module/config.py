"""
GNN-KAN 統一配置系統 - 支援最新的特徵處理和純粹KAN實現
專注於KAN取代MLP的核心價值：可學習激活函數、B-spline基函數
"""

import torch
import numpy as np


class SimplifiedGNNKANConfig:
    """
    簡化的GNN-KAN配置 - 專注於KAN核心特性
    新增：ICA、kPCA特徵處理支持
    強調：KAN vs MLP的本質差異
    """
    
    def __init__(self):
        # 🎯 核心KAN配置 - 確保純粹性
        self.kan_grid_size = 3              # KAN B-spline 網格大小 (簡化)
        self.kan_spline_order = 3           # 樣條階數
        self.kan_num_basis = 4              # 基函數數量 (簡化)
        self.adaptive_spline_order = False  # 關閉自適應以簡化
        self.learnable_activation = True    # 可學習激活函數（KAN vs MLP關鍵）
        self.minimize_linear_component = True  # 最小化MLP特性
        
        # 🔧 特徵處理配置 - 新增ICA/kPCA選項
        self.feature_method = 'ica'         # 'ica', 'kpca', 'pca', 'simplified'
        self.use_stl_decomposition = False  # 移除複雜STL分解
        self.use_kll_processing = False     # 移除複雜KLL處理
        self.use_ica = True                 # 啟用ICA特徵提取
        self.use_kpca = False               # 可選的kPCA
        self.ica_components = 12            # ICA成分數量
        self.kpca_kernel = 'rbf'           # kPCA核函數
        
        # 🎯 GNN-KAN架構配置 - 純粹KAN實現 (大幅簡化)
        self.input_dim = 32
        self.target_feature_dim = 32       # 目標特徵維度
        self.hidden_dims = [64, 32]        # 2層KAN結構，大幅降低維度
        self.output_dim = 16
        self.num_gnn_layers = 1
        self.dropout = 0.2
        self.learnable_graph = True        # 動態圖結構學習
        
        # 🚀 訓練配置 - 針對KAN優化
        self.learning_rate = 1e-4          # 適度增大學習率
        self.base_learning_rate = 1e-4     # 兼容性別名
        self.weight_decay = 1e-5
        self.num_epochs = 200              # 增加預設週期
        self.epochs = 200                  # 兼容性別名
        self.batch_size = 32
        self.patience = 25
        self.min_delta = 1e-5
        self.warmup_epochs = 15            # 預熱階段
        
        # 🔧 穩定性配置 - 簡化但有效
        self.gradient_clip_norm = 1.0      # 放寬梯度裁剪以適應低學習率
        self.stability_check_freq = 10     # 更頻繁的檢查
        self.use_layer_norm = True         # 使用LayerNorm
        self.use_batch_norm = False        # 不使用BatchNorm（避免MLP特性）
        
        # 📊 評估配置
        self.eval_metrics = ['precision', 'recall', 'f1', 'accuracy']
        self.eval_topk = [1, 3, 5]
        
        # 🎯 KAN特有的優化配置
        self.kan_l1_lambda = 1e-3          # B-spline正則化
        self.base_l1_lambda = 1e-3         # 兼容性別名
        self.kan_entropy_lambda = 1e-3     # 熵正則化
        self.base_entropy_lambda = 1e-3    # 兼容性別名
        self.spline_weight_decay = 1e-5    # 樣條權重衰減
        self.activation_weight_decay = 1e-5 # 激活權重衰減
        self.stability_check_frequency = 50 # 穩定性檢查頻率
        
        # 📁 路徑配置
        self.data_dir = "data"
        self.output_dir = "results"
        self.log_dir = "logs"
        self.model_save_dir = "models"
        
        # 🔍 調試配置
        self.debug = False
        self.verbose = True
        self.save_intermediate = False
        
        # 🖥️ 設備配置 - GPU支持
        self.use_cuda = True                # 啟用CUDA GPU加速
        self.device = 'auto'                # 'auto', 'cuda', 'cpu'
        self.gpu_memory_fraction = 0.8      # GPU記憶體使用比例
        self.cpu_fallback = True            # GPU失敗時自動切換CPU
        
        self.use_simplified_features = True
        self.max_features = 256
        self.l2_lambda = 1e-5  # L2 正則化
        self.smoothness_lambda = 1e-6  # 平滑性正則化
        self.learnable_edges = True  # 可學習邊權重
        self.max_log_features = 100  # 最大日誌特徵數
        
        # 🔧 模型特定配置 - 確保兼容性
        self.hidden_dim = self.hidden_dims[0] if self.hidden_dims else 64
        self.num_layers = self.num_gnn_layers  # 層數別名
        self.use_residual = True  # 殘差連接
        self.kan_config = self.get_kan_config()  # KAN配置對象
        
        # 🔧 新增數值穩定性配置
        self.nan_detection_freq = 5        # NaN檢測頻率
        self.parameter_reset_threshold = 3  # 參數重置閾值
        self.learning_rate_decay = 0.95    # 學習率衰減
        self.min_learning_rate = 1e-6      # 最小學習率
        self.use_ema = True                # 指數移動平均
        self.ema_decay = 0.999             # EMA衰減率
    
    def get_feature_processing_config(self):
        """獲取特徵處理配置"""
        return {
            'method': self.feature_method,
            'use_ica': self.use_ica,
            'use_kpca': self.use_kpca,
            'ica_components': self.ica_components,
            'kpca_kernel': self.kpca_kernel,
            'use_stl': self.use_stl_decomposition,
            'use_kll': self.use_kll_processing,
            'target_dim': self.input_dim
        }
    
    def get_kan_config(self):
        """獲取純粹KAN配置 - 強調非MLP特性"""
        return {
            'grid_size': self.kan_grid_size,
            'spline_order': self.kan_spline_order,
            'num_basis': self.kan_num_basis,
            'adaptive_spline_order': self.adaptive_spline_order,
            'learnable_activation': self.learnable_activation,
            'minimize_linear': self.minimize_linear_component,
            'l1_lambda': self.kan_l1_lambda,
            'entropy_lambda': self.kan_entropy_lambda,
            'spline_weight_decay': self.spline_weight_decay,
            'activation_weight_decay': self.activation_weight_decay
        }
    
    def get_gnn_config(self):
        """獲取GNN配置"""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'num_layers': self.num_gnn_layers,
            'dropout': self.dropout,
            'learnable_graph': self.learnable_graph,
            'kan_config': self.get_kan_config()
        }
    
    def get_training_config(self):
        """獲取訓練配置"""
        return {
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'gradient_clip_norm': self.gradient_clip_norm,
            'stability_check_freq': self.stability_check_freq
        }
    
    def get_device_config(self):
        """獲取設備配置"""
        return {
            'use_cuda': self.use_cuda,
            'device': self.device,
            'gpu_memory_fraction': self.gpu_memory_fraction,
            'cpu_fallback': self.cpu_fallback
        }
    
    def validate_config(self):
        """驗證配置的合理性"""
        issues = []
        
        # 檢查KAN相關配置
        if self.kan_grid_size < 3:
            issues.append("KAN grid_size should be >= 3 for effective B-spline")
        
        if self.kan_num_basis < 2:
            issues.append("KAN num_basis should be >= 2 for non-trivial approximation")
        
        # 檢查維度配置
        if self.input_dim <= 0 or self.output_dim <= 0:
            issues.append("Input and output dimensions must be positive")
        
        if len(self.hidden_dims) == 0:
            issues.append("Hidden dimensions should not be empty")
        
        # 檢查特徵處理配置
        valid_methods = ['ica', 'kpca', 'pca', 'simplified']
        if self.feature_method not in valid_methods:
            issues.append(f"Feature method must be one of {valid_methods}")
        
        return issues
    
    def update_for_kan_purity(self):
        """更新配置以最大化KAN純粹性，最小化MLP特性"""
        print("🎯 Updating config for maximum KAN purity...")
        
        # 🚀 大幅增強KAN表達能力 - 針對複雜數據集優化
        self.kan_grid_size = max(20, self.kan_grid_size)  # 高容量版本更高
        self.kan_num_basis = max(24, self.kan_num_basis)  # 更多基函數
        self.kan_spline_order = max(6, self.kan_spline_order)  # 更高階樣條
        self.adaptive_spline_order = True
        self.learnable_activation = True
        self.minimize_linear_component = True
        
        # 🎯 增強特徵處理能力 - 專門針對train-ticket類型數據
        if self.feature_method in ['simplified', 'stl']:
            self.feature_method = 'ica'  # 強制使用更強的特徵處理
        
        # 🔥 深化網絡架構 - 提升複雜模式學習能力
        if hasattr(self, 'hidden_dims'):
            # 擴展到更深的網絡 - 高容量版本
            self.hidden_dims = [768, 512, 384, 256, 192, 128, 96, 64]
        else:
            self.hidden_dims = [384, 256, 192, 128]
        
        # 📈 優化訓練配置 - 提升準確率
        self.learning_rate = min(0.0002, self.learning_rate)  # 更小學習率
        self.num_epochs = max(250, self.num_epochs)  # 更多訓練輪數
        self.warmup_epochs = max(30, getattr(self, 'warmup_epochs', 15))
        
        # 🎯 強化正則化 - 避免過擬合同時保持表達能力
        self.gradient_clip_norm = 0.3  # 更嚴格的梯度控制
        self.dropout = max(0.2, self.dropout)  # 更強的正則化
        
        # 🔧 ICA增強配置 - 專門處理複雜時序特徵
        current_ica = getattr(self, 'ica_components', None)
        if current_ica is None:
            self.ica_components = 48  # 默認值
        else:
            self.ica_components = max(48, current_ica)
        self.ica_max_iter = 1500  # 更多ICA迭代
        self.ica_fun = 'logcosh'  # 更穩定的ICA函數
        
        # 🚀 添加新的KAN特性
        self.kan_adaptive_activation = True
        self.kan_nonlinear_residual = True
        self.kan_feature_interaction = True
        self.kan_multi_scale_learning = True  # 高容量獨有
        
        # 移除MLP相關配置
        self.use_batch_norm = False         # BatchNorm是MLP常用技術
        self.use_layer_norm = True          # LayerNorm更通用
        
        self.use_stl_decomposition = False
        self.use_kll_processing = False
        
        print("✓ Config updated for KAN purity with enhanced accuracy features")


class HighCapacityGNNKANConfig(SimplifiedGNNKANConfig):
    """
    高容量KAN配置 - 保持純粹性的同時提升表達能力
    專門用於複雜場景和大規模數據
    """
    
    def __init__(self):
        super().__init__()
        
        # 🚀 增強KAN容量 - 保持純粹性
        self.kan_grid_size = 12             # 更高密度的B-spline
        self.kan_num_basis = 12             # 更多基函數
        self.kan_spline_order = 4           # 更高階樣條
        
        # 🎯 擴展架構 - 深度KAN網絡
        self.input_dim = 128
        self.target_feature_dim = 128       # 目標特徵維度  
        self.hidden_dims = [256, 192, 128, 96, 64]  # 5層深度
        self.output_dim = 64
        self.num_gnn_layers = 3
        
        # 🔧 高容量特徵處理
        self.feature_method = 'ica'         # 高容量場景下ICA更有效
        self.ica_components = 16            # 更多ICA成分
        
        # ⚡ 高容量穩定性
        self.gradient_clip_norm = 0.8       # 更嚴格的梯度控制
        self.stability_check_freq = 30      # 更頻繁的穩定性檢查
        self.dropout = 0.15                 # 更強的正則化
        
        # 📈 高容量訓練
        self.learning_rate = 0.0005         # 更小的學習率
        self.base_learning_rate = 0.0005    # 兼容性別名
        self.num_epochs = 150               # 更多訓練輪次
        self.epochs = 150                   # 兼容性別名
        self.patience = 25                  # 更大的耐心值
        self.warmup_epochs = 15             # 預熱階段
        
        # 🎯 高容量KAN正則化
        self.base_l1_lambda = 2e-3          # 兼容性別名
        self.base_entropy_lambda = 2e-3     # 兼容性別名
        self.stability_check_frequency = 30 # 穩定性檢查頻率
        
        self.use_kpca = True  # 優先使用kPCA
        self.l2_lambda = 5e-5
        self.smoothness_lambda = 2e-6
        self.learnable_edges = True
        self.max_log_features = 150
        
        # 🔧 模型特定配置 - 確保兼容性
        self.hidden_dim = self.hidden_dims[0] if self.hidden_dims else 128
        self.num_layers = self.num_gnn_layers  # 層數別名
        self.use_residual = True  # 殘差連接
        self.kan_config = self.get_kan_config()  # KAN配置對象
    
    def update_for_kan_purity(self):
        """更新配置以最大化KAN純粹性，最小化MLP特性"""
        print("🎯 Updating config for maximum KAN purity...")
        
        # 🚀 大幅增強KAN表達能力 - 針對複雜數據集優化
        self.kan_grid_size = max(20, self.kan_grid_size)  # 高容量版本更高
        self.kan_num_basis = max(24, self.kan_num_basis)  # 更多基函數
        self.kan_spline_order = max(6, self.kan_spline_order)  # 更高階樣條
        self.adaptive_spline_order = True
        self.learnable_activation = True
        self.minimize_linear_component = True
        
        # 🎯 增強特徵處理能力 - 專門針對train-ticket類型數據
        if self.feature_method in ['simplified', 'stl']:
            self.feature_method = 'ica'  # 強制使用更強的特徵處理
        
        # 🔥 深化網絡架構 - 提升複雜模式學習能力
        if hasattr(self, 'hidden_dims'):
            # 擴展到更深的網絡 - 高容量版本
            self.hidden_dims = [768, 512, 384, 256, 192, 128, 96, 64]
        else:
            self.hidden_dims = [384, 256, 192, 128]
        
        # 📈 優化訓練配置 - 提升準確率
        self.learning_rate = min(0.0002, self.learning_rate)  # 更小學習率
        self.num_epochs = max(250, self.num_epochs)  # 更多訓練輪數
        self.warmup_epochs = max(30, getattr(self, 'warmup_epochs', 15))
        
        # 🎯 強化正則化 - 避免過擬合同時保持表達能力
        self.gradient_clip_norm = 0.3  # 更嚴格的梯度控制
        self.dropout = max(0.2, self.dropout)  # 更強的正則化
        
        # 🔧 ICA增強配置 - 專門處理複雜時序特徵
        current_ica = getattr(self, 'ica_components', None)
        if current_ica is None:
            self.ica_components = 48  # 默認值
        else:
            self.ica_components = max(48, current_ica)
        self.ica_max_iter = 1500  # 更多ICA迭代
        self.ica_fun = 'logcosh'  # 更穩定的ICA函數
        
        # 🚀 添加新的KAN特性
        self.kan_adaptive_activation = True
        self.kan_nonlinear_residual = True
        self.kan_feature_interaction = True
        self.kan_multi_scale_learning = True  # 高容量獨有
        
        # 移除MLP相關配置
        self.use_batch_norm = False         # BatchNorm是MLP常用技術
        self.use_layer_norm = True          # LayerNorm更通用
        
        self.use_stl_decomposition = False
        self.use_kll_processing = False
        
        print("✓ Config updated for KAN purity with enhanced accuracy features")


class FastGNNKANConfig(SimplifiedGNNKANConfig):
    """
    快速KAN配置 - 在保持核心特性的同時優化速度
    專門用於實時應用和資源受限環境
    """
    
    def __init__(self):
        super().__init__()
        
        # ⚡ 快速KAN配置 - 保持核心價值
        self.kan_grid_size = 6              # 適中的網格密度
        self.kan_num_basis = 6              # 適中的基函數數
        self.kan_spline_order = 3           # 標準樣條階數
        
        # 🎯 精簡架構 - 少而精的KAN層
        self.input_dim = 32
        self.target_feature_dim = 32        # 目標特徵維度
        self.hidden_dims = [64, 48]         # 2層精簡結構
        self.output_dim = 24
        self.num_gnn_layers = 1
        
        # ⚡ 快速特徵處理
        self.feature_method = 'simplified'  # 最快的特徵處理
        self.use_ica = False                # 關閉耗時的ICA
        
        # 🚀 快速訓練
        self.learning_rate = 0.002          # 更大的學習率
        self.base_learning_rate = 0.002     # 兼容性別名
        self.num_epochs = 50                # 更少的訓練輪次
        self.epochs = 50                    # 兼容性別名
        self.batch_size = 64                # 更大的batch size
        self.patience = 10                  # 更小的耐心值
        self.warmup_epochs = 5              # 快速預熱
        
        # ⚡ 快速穩定性
        self.stability_check_freq = 100     # 最少的穩定性檢查
        self.stability_check_frequency = 100 # 兼容性別名
        self.gradient_clip_norm = 1.5       # 更寬鬆的梯度控制
        
        # 🎯 快速KAN正則化
        self.base_l1_lambda = 5e-4          # 兼容性別名
        self.base_entropy_lambda = 5e-4     # 兼容性別名
        
        self.l2_lambda = 1e-6
        self.smoothness_lambda = 5e-7
        self.learnable_edges = False
        self.max_log_features = 50
        
        # 🔧 模型特定配置 - 確保兼容性
        self.hidden_dim = self.hidden_dims[0] if self.hidden_dims else 64
        self.num_layers = self.num_gnn_layers  # 層數別名
        self.use_residual = True  # 殘差連接
        self.kan_config = self.get_kan_config()  # KAN配置對象
    
    def update_for_kan_purity(self):
        """更新配置以最大化KAN純粹性，最小化MLP特性"""
        print("🎯 Updating config for maximum KAN purity...")
        
        # 🚀 增強KAN表達能力 - 快速但有效
        self.kan_grid_size = max(12, self.kan_grid_size)  # 快速版本適中提升
        self.kan_num_basis = max(16, self.kan_num_basis)  # 適中的基函數
        self.kan_spline_order = max(4, self.kan_spline_order)  # 適中階樣條
        self.adaptive_spline_order = True
        self.learnable_activation = True
        self.minimize_linear_component = True
        
        # 🎯 增強特徵處理能力 - 專門針對train-ticket類型數據
        if self.feature_method == 'simplified':
            self.feature_method = 'kpca'  # 快速版本使用kPCA
        
        # 適度深化網絡架構 - 平衡速度與準確率
        if hasattr(self, 'hidden_dims'):
            # 適度擴展網絡 - 快速版本
            self.hidden_dims = [256, 192, 128, 96, 64]
        else:
            self.hidden_dims = [192, 128, 64]
        
        # 📈 優化訓練配置 - 提升準確率但保持速度
        self.learning_rate = min(0.0008, self.learning_rate)  # 適中學習率
        self.num_epochs = max(100, self.num_epochs)  # 適中訓練輪數
        self.warmup_epochs = max(15, getattr(self, 'warmup_epochs', 5))
        
        # 🎯 適度正則化 - 平衡過擬合和表達能力
        self.gradient_clip_norm = 0.8  # 適中的梯度控制
        self.dropout = max(0.1, self.dropout)  # 適中的dropout
        
        # 🔧 kPCA配置 - 快速且有效的特徵處理
        current_ica = getattr(self, 'ica_components', None)
        if current_ica is None:
            self.ica_components = 24  # 默認值
        else:
            self.ica_components = max(24, current_ica)
        self.ica_max_iter = 800  # 適中ICA迭代
        self.ica_fun = 'logcosh'  # RBF核函數
        
        # 🚀 添加新的KAN特性
        self.kan_adaptive_activation = True
        self.kan_nonlinear_residual = True
        self.kan_feature_interaction = True
        
        # 移除MLP相關配置
        self.use_batch_norm = False         # BatchNorm是MLP常用技術
        self.use_layer_norm = True          # LayerNorm更通用
        
        self.use_stl_decomposition = False
        self.use_kll_processing = False
        
        print("✓ Config updated for KAN purity with enhanced accuracy features")


class ConfigFactory:
    """配置工廠類 - 根據需求創建適當的配置"""
    
    @staticmethod
    def create_config(config_type='simplified', **kwargs):
        """創建配置對象"""
        config_map = {
            'simplified': SimplifiedGNNKANConfig,
            'high_capacity': HighCapacityGNNKANConfig,
            'fast': FastGNNKANConfig
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