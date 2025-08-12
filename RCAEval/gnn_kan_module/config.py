"""
GNN-KAN 統一配置系統 - 支援最新的特徵處理和純粹KAN實現
專注於KAN取代MLP的核心價值：可學習激活函數、B-spline基函數
"""

import torch
import numpy as np


class SimplifiedGNNKANConfig:
    """
    🚨 ISSUE 5: 泛化能力限制分析
    
    當前配置的泛化限制：
    1. **固定特徵維度問題**：
       - target_feature_dim=64 是硬編碼的，不適應不同規模系統
       - 小系統（<10個服務）：64維可能過度參數化
       - 大系統（>100個服務）：64維可能表達不足
    
    2. **KAN參數固定化問題**：
       - kan_grid_size和kan_num_basis對所有場景使用相同值
       - 不同故障類型可能需要不同的非線性複雜度
       - 靜態配置無法適應數據分佈變化
    
    3. **領域特化問題**：
       - 配置基於微服務架構優化，對其他架構可能不適用
       - 缺乏跨領域的自適應機制
       - 沒有考慮不同行業的故障模式差異
    
    4. **時間序列長度依賴**：
       - 配置假設了特定的時間窗口長度
       - 對於短時爆發故障vs長期漸變故障表現可能差異很大
    
    🔧 改進建議：
    1. 自適應維度配置：
       # target_feature_dim = max(32, min(128, num_services * 2))
       # kan_complexity = auto_tune_kan_params(data_complexity)
    
    2. 領域適應機制：
       # if domain == 'financial': config.conservative_mode = True
       # elif domain == 'gaming': config.latency_sensitive = True
    
    3. 動態配置調整：
       # config.auto_tune_from_validation(historical_performance)
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
        """
        動態更新配置以增強KAN純度 - 基於特徵維度自適應調整
        🎯 目標：保持KAN特性純粹，最小化MLP相關性
        """
        if self.target_feature_dim > 32:
            # 🔧 修正5：基於特徵維度的自適應調整
            # 中高維特徵需要更精細的基函數網格
            old_grid_size = self.kan_grid_size
            old_num_basis = self.kan_num_basis
            
            # 自適應調整KAN複雜度
            self.kan_grid_size = min(old_grid_size * 2, 20)
            self.kan_num_basis = min(old_num_basis + 4, 24)
            
            print(f"🔧 自適應KAN調整: grid_size {old_grid_size}→{self.kan_grid_size}, num_basis {old_num_basis}→{self.kan_num_basis}")
        
        # 強化KAN純度設置
        self.minimize_linear_component = True
        self.learnable_activation = True
        self.adaptive_spline_order = False  # 簡化配置中保持固定
        
        # 🔧 修正5：添加自適應調整記錄
        self._adaptation_applied = True
        self._original_params = {
            'grid_size': getattr(self, '_original_grid_size', self.kan_grid_size),
            'num_basis': getattr(self, '_original_num_basis', self.kan_num_basis)
        }
    
    def auto_adapt_to_data(self, data_characteristics):
        """
        🔧 修正5：根據數據特徵自動調整配置
        
        Args:
            data_characteristics: Dict包含數據特徵信息
                - num_services: 服務數量
                - num_metrics: 指標數量  
                - time_series_length: 時間序列長度
                - failure_type: 故障類型
                - domain: 應用領域
        """
        print("🔧 執行數據自適應配置調整...")
        
        # 1. 基於系統規模調整特徵維度
        num_services = data_characteristics.get('num_services', 10)
        num_metrics = data_characteristics.get('num_metrics', 20)
        
        # 自適應特徵維度：基於服務數量和指標複雜度
        if num_services <= 5:  # 小型系統
            self.target_feature_dim = 32
            self.hidden_dim = 48
            self.kan_grid_size = 6
            print(f"📊 檢測到小型系統({num_services}服務)，降低模型複雜度")
        elif num_services <= 15:  # 中型系統
            self.target_feature_dim = 64
            self.hidden_dim = 96
            self.kan_grid_size = 10
            print(f"📊 檢測到中型系統({num_services}服務)，使用標準配置")
        else:  # 大型系統
            self.target_feature_dim = min(128, num_services * 4)
            self.hidden_dim = min(192, num_services * 8)
            self.kan_grid_size = min(15, num_services)
            print(f"📊 檢測到大型系統({num_services}服務)，增加模型容量")
        
        # 2. 基於故障類型調整訓練策略
        failure_type = data_characteristics.get('failure_type', 'unknown')
        if failure_type == 'cascading':
            # 級聯故障需要更多訓練輪數
            self.num_epochs = max(150, self.num_epochs)
            self.learning_rate *= 0.8  # 更保守的學習率
            print("🔄 級聯故障模式：增加訓練輪數，降低學習率")
        elif failure_type == 'burst':
            # 突發故障可以快速收斂
            self.num_epochs = min(100, self.num_epochs)
            self.learning_rate *= 1.2  # 更激進的學習率
            print("💥 突發故障模式：減少訓練輪數，提高學習率")
        
        # 3. 基於領域調整正則化策略
        domain = data_characteristics.get('domain', 'general')
        if domain == 'financial':
            # 金融領域需要更保守的策略
            self.dropout = min(0.2, self.dropout * 1.5)
            self.weight_decay *= 2.0
            print("💰 金融領域：增強正則化，提高穩定性")
        elif domain == 'gaming':
            # 遊戲領域對延遲敏感，簡化模型
            self.num_gnn_layers = min(2, self.num_gnn_layers)
            self.batch_size = min(16, self.batch_size)
            print("🎮 遊戲領域：簡化模型，減少延遲")
        
        # 4. 基於時間序列長度調整
        ts_length = data_characteristics.get('time_series_length', 100)
        if ts_length < 50:
            # 短時間序列，降低模型複雜度防止過擬合
            self.dropout = max(0.2, self.dropout)
            self.num_epochs = min(80, self.num_epochs)
            print("⏱️ 短時間序列：增加dropout，減少訓練輪數")
        elif ts_length > 500:
            # 長時間序列，可以使用更複雜的模型
            self.num_epochs = max(150, self.num_epochs)
            self.batch_size = min(64, self.batch_size * 2)
            print("📈 長時間序列：增加訓練輪數和批次大小")
        
        # 記錄調整信息
        self._data_adaptation_info = {
            'num_services': num_services,
            'adapted_feature_dim': self.target_feature_dim,
            'adapted_hidden_dim': self.hidden_dim,
            'adapted_kan_grid_size': self.kan_grid_size,
            'adapted_epochs': self.num_epochs,
            'failure_type': failure_type,
            'domain': domain
        }
        
        print(f"✅ 自適應配置完成：feature_dim={self.target_feature_dim}, hidden_dim={self.hidden_dim}")
        
    def get_adaptation_summary(self):
        """獲取自適應調整的摘要信息"""
        if hasattr(self, '_data_adaptation_info'):
            return self._data_adaptation_info
        return {'adaptation_applied': False}


class HighCapacityGNNKANConfig(SimplifiedGNNKANConfig):
    """
    高容量KAN配置 - 保持純粹性的同時提升表達能力
    🔧 修正版：解決維度不匹配和穩定性問題
    """
    
    def __init__(self):
        super().__init__()
        
        # 🚀 增強KAN容量 - 保持與輸入處理器兼容
        self.kan_grid_size = 8              # 適度增加 B-spline 密度
        self.kan_num_basis = 8              # 適度增加基函數
        self.kan_spline_order = 4           # 稍高階樣條
        
        # 🎯 擴展架構 - 與實際特徵維度匹配
        self.input_dim = 64                 # 🔧 修正：匹配實際輸入特徵維度
        self.target_feature_dim = 64        # 🔧 修正：與 optimized_config 一致
        self.hidden_dims = [96, 64, 48]     # 🔧 修正：適度深化，避免過度複雜
        self.output_dim = 32                # 🔧 修正：適中的輸出維度
        self.num_gnn_layers = 2             # 🔧 修正：從3層減少到2層，提高穩定性
        
        # 🔧 高容量特徵處理 - 與當前系統兼容
        self.feature_method = 'kpca'        # 🔧 修正：與 optimized_config 一致
        self.kpca_kernel = 'rbf'            # 🔧 修正：確保一致性
        self.ica_components = 12            # 保持適中的 ICA 成分
        
        # ⚡ 高容量穩定性 - 加強數值穩定性
        self.gradient_clip_norm = 0.5       # 🔧 修正：更嚴格的梯度控制
        self.stability_check_freq = 20      # 🔧 修正：更頻繁檢查，但不過度
        self.dropout = 0.25                 # 🔧 修正：適度增強正則化
        
        # 📈 高容量訓練 - 更保守的訓練策略
        self.learning_rate = 5e-5           # 🔧 修正：更小的學習率提高穩定性
        self.base_learning_rate = 5e-5      # 兼容性別名
        self.num_epochs = 150               # 🔧 修正：適中的訓練輪次
        self.epochs = 150                   # 兼容性別名
        self.patience = 20                  # 🔧 修正：適中的耐心值
        self.warmup_epochs = 10             # 🔧 修正：適度預熱
        
        # 🎯 高容量KAN正則化 - 平衡表達能力與穩定性
        self.base_l1_lambda = 1e-3          # 保持適度正則化
        self.base_entropy_lambda = 1e-3     # 保持適度熵正則化
        self.stability_check_frequency = 25 # 🔧 修正：平衡檢查頻率
        
        # 🔧 兼容性配置 - 確保與現有系統無縫集成
        self.use_kpca = True                # 與 optimized_config 一致
        self.l2_lambda = 1e-5               # 適度 L2 正則化
        self.smoothness_lambda = 1e-6       # 平滑性正則化
        self.learnable_edges = True         # 保持動態圖學習
        self.max_log_features = 100         # 適中的日誌特徵數
        
        # 🔧 模型特定配置 - 確保維度兼容性
        self.hidden_dim = self.hidden_dims[0] if self.hidden_dims else 96
        self.num_layers = self.num_gnn_layers  # 層數別名
        self.use_residual = True            # 殘差連接提高穩定性
        self.kan_config = self.get_kan_config()  # KAN配置對象
    
    def update_for_kan_purity(self):
        """更新配置以最大化KAN純粹性，同時保持穩定性"""
        print("🎯 Updating HighCapacity config for balanced KAN purity and stability...")
        
        # 🚀 平衡的KAN增強 - 避免過度複雜化
        self.kan_grid_size = min(12, self.kan_grid_size + 2)    # 🔧 修正：適度增加
        self.kan_num_basis = min(12, self.kan_num_basis + 2)    # 🔧 修正：適度增加
        self.kan_spline_order = min(5, self.kan_spline_order + 1) # 🔧 修正：限制最大階數
        self.adaptive_spline_order = True
        self.learnable_activation = True
        self.minimize_linear_component = True
        
        # 🎯 保守的特徵處理增強
        if self.feature_method in ['simplified']:
            self.feature_method = 'kpca'    # 🔧 修正：使用穩定的 KPCA
        
        # 🔥 保守的網絡架構調整 - 避免梯度問題
        if hasattr(self, 'hidden_dims'):
            # 🔧 修正：適度擴展，保持穩定性
            self.hidden_dims = [128, 96, 64]  # 比原始方案更保守
        
        # 📈 穩定的訓練配置
        self.learning_rate = min(3e-5, self.learning_rate)  # 🔧 修正：更保守的學習率
        self.num_epochs = min(200, self.num_epochs + 20)    # 🔧 修正：適度增加訓練
        self.warmup_epochs = min(15, getattr(self, 'warmup_epochs', 10) + 5)
        
        # 🎯 增強穩定性控制
        self.gradient_clip_norm = 0.3       # 🔧 修正：更嚴格控制
        self.dropout = min(0.3, self.dropout + 0.05)  # 🔧 修正：適度增加 dropout
        
        print(f"  ✓ KAN Grid Size: {self.kan_grid_size}")
        print(f"  ✓ Hidden Dims: {self.hidden_dims}")
        print(f"  ✓ Learning Rate: {self.learning_rate}")
        print(f"  ✓ Gradient Clip: {self.gradient_clip_norm}")
        print(f"  ✓ Dropout: {self.dropout}")
    
    def get_kan_config(self):
        """🔧 新增：返回 KAN 層配置對象"""
        return {
            'grid_size': self.kan_grid_size,
            'spline_order': self.kan_spline_order,
            'num_basis': self.kan_num_basis,
            'use_residual': self.use_residual,
            'adaptive_spline_order': self.adaptive_spline_order,
            'learnable_activation': self.learnable_activation,
            'l1_lambda': self.base_l1_lambda,
            'entropy_lambda': self.base_entropy_lambda
        }
    
    def get_stability_config(self):
        """🔧 新增：返回穩定性配置"""
        return {
            'gradient_clip_norm': self.gradient_clip_norm,
            'stability_check_freq': self.stability_check_freq,
            'l2_lambda': self.l2_lambda,
            'smoothness_lambda': self.smoothness_lambda
        }


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
    
    @staticmethod
    def get_supported_types():
        """獲取支持的配置類型列表"""
        return ['simplified', 'high_capacity', 'fast']
    
    @staticmethod
    def get_config_description(config_type: str):
        """獲取配置類型的描述"""
        descriptions = {
            'simplified': '簡化配置 - 專注KAN核心特性，證明KAN取代MLP的有效性',
            'high_capacity': '高容量配置 - 最大化KAN表達能力，追求極致準確率',
            'fast': '快速配置 - 優化KAN執行速度，保持核心特性'
        }
        return descriptions.get(config_type, '未知配置類型')


# 向後兼容性
GNNKANConfig = SimplifiedGNNKANConfig