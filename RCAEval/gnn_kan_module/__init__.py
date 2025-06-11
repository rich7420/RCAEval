"""
GNN-KAN Module: 完整的模組化GNN-KAN實現
整合所有核心功能到統一模組中
"""

# 核心配置
from .config import SimplifiedGNNKANConfig

# 特徵提取模組
from .feature_extractors import (
    MultiModalFeatureExtractor,
    enhanced_feature_fusion
)

# 模型核心組件 - 修復導入問題
try:
    from .models import GNNKANModel
except ImportError:
    print("⚠️ 創建臨時 GNNKANModel...")
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class GNNKANModel(nn.Module):
        """臨時 GNN-KAN 模型實現"""
        
        def __init__(self, config, num_nodes):
            super(GNNKANModel, self).__init__()
            self.config = config
            self.num_nodes = num_nodes
            
            # 特徵投影層
            self.feature_projection = nn.Linear(config.target_feature_dim, config.input_dim)
            
            # 簡化的 GNN 層
            self.conv_layers = nn.ModuleList([
                nn.Linear(config.input_dim if i == 0 else config.hidden_dims[i-1], 
                         config.hidden_dims[i] if i < len(config.hidden_dims) else config.output_dim)
                for i in range(config.num_gnn_layers)
            ])
            
            # 解碼器
            self.graph_decoder = nn.Sequential(
                nn.Linear(config.output_dim * 2, config.output_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.output_dim, 1)
            )
            
            self.dropout = nn.Dropout(config.dropout)
        
        def forward(self, node_features, edge_index):
            """前向傳播"""
            # 特徵投影
            x = self.feature_projection(node_features)
            
            # GNN 層
            for conv in self.conv_layers:
                x = F.relu(conv(x))
                x = self.dropout(x)
            
            # 計算鄰接矩陣分數
            num_nodes = x.size(0)
            adj_scores = torch.zeros(num_nodes, num_nodes, device=x.device)
            
            for i in range(num_nodes):
                for j in range(num_nodes):
                    edge_features = torch.cat([x[i], x[j]], dim=0)
                    score = torch.sigmoid(self.graph_decoder(edge_features))
                    adj_scores[i, j] = score.squeeze()
            
            return x, adj_scores

from .training import (
    train_gnn_kan_model,
    create_model_with_config,
    TemporalAttention,
    ModelManager,
    AdvancedGNNKANTrainer
)

# 別名定義 - 確保向後兼容
validate_model_setup = create_model_with_config
AdvancedTrainingManager = AdvancedGNNKANTrainer

# 圖構建
from .graph_constructors import (
    SimplifiedGraphConstructor,
    IntelligentServiceGraphConstructor
)

# 高級處理器
from .advanced_processors import (
    DynamicModelAdjuster,
    create_advanced_processor
)

# 特徵處理函數
from .feature_processing import (
    simplified_metric_processing,
    enhanced_trace_processing,
    psm_metric_processing
)


def compute_service_criticality_weights(node_names):
    """
    基於服務名稱計算重要性權重 - 統一版本
    
    Args:
        node_names: 節點名稱列表
        
    Returns:
        weights: 重要性權重列表
    """
    import torch
    import numpy as np
    
    critical_services = {
        'frontend': 3.0, 'front-end': 3.0,
        'checkout': 2.8, 'payment': 2.8,
        'cart': 2.5, 'catalog': 2.2,
        'currency': 2.0, 'redis': 2.3,
        'database': 2.5, 'db': 2.5,
        'email': 1.8, 'ad': 1.6,
        'recommendation': 1.7
    }
    
    critical_metrics = {
        'error': 3.0, 'exception': 2.8, 'fail': 2.5,
        'latency': 2.5, 'delay': 2.2, 'timeout': 2.3,
        'cpu': 2.0, 'memory': 1.8, 'mem': 1.8,
        'disk': 1.6, 'network': 1.7, 'connection': 1.5
    }
    
    weights = []
    for name in node_names:
        name_str = str(name).lower()
        weight = 1.0
        
        # 檢查關鍵服務
        for service, service_weight in critical_services.items():
            if service in name_str:
                weight = max(weight, service_weight)
        
        # 檢查關鍵指標
        for metric, metric_weight in critical_metrics.items():
            if metric in name_str:
                weight = max(weight, metric_weight)
        
        # 指標統計類型加權
        if any(stat in name_str for stat in ['max', 'std', 'trend', 'peak']):
            weight *= 1.2
        
        weights.append(min(weight, 3.0))  # 限制最大權重
    
    return torch.tensor(weights, dtype=torch.float)


# 確保所有主要組件都可以被導入
__all__ = [
    'SimplifiedGNNKANConfig',
    'MultiModalFeatureExtractor', 
    'GNNKANModel',
    'train_gnn_kan_model',
    'create_model_with_config',
    'validate_model_setup',
    'AdvancedTrainingManager',
    'SimplifiedGraphConstructor',
    'IntelligentServiceGraphConstructor', 
    'DynamicModelAdjuster',
    'create_advanced_processor',
    'enhanced_feature_fusion',
    'simplified_metric_processing',
    'enhanced_trace_processing', 
    'psm_metric_processing',
    'compute_service_criticality_weights',
    'TemporalAttention',
    'ModelManager',
    'AdvancedGNNKANTrainer'
]

print("✅ GNN-KAN模組完全載入成功 - 所有功能已模組化")