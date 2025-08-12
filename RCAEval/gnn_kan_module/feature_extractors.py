"""
This module is deprecated.
All feature extraction logic has been moved to the `RCAEval.gnn_kan_module.processors` sub-package.
This file is kept for backward compatibility during transition and will be removed in a future version.
"""

# To prevent accidental usage, you can raise an error on import.
# raise DeprecationWarning("This module is deprecated. Use processors sub-package instead.")

"""
GNN-KAN RCA: Feature extraction modules
Contains all feature extraction related classes and functions
"""

import time
import warnings
import numpy as np
import pandas as pd
import torch
import traceback
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

# 從 utils 導入統一的權重計算函數
from .utils import compute_service_criticality_weights

# Import other required functions from the appropriate modules
try:
    from ..io.time_series import preprocess, drop_constant
except ImportError:
    print("警告：io.time_series 模組不可用，使用簡化預處理")
    
    def preprocess(data, dataset=None, **kwargs):
        """簡化的數據預處理"""
        if isinstance(data, pd.DataFrame):
            return data.fillna(method='ffill').fillna(0)
        return data
    
    def drop_constant(data):
        """簡化的常數列移除"""
        if isinstance(data, pd.DataFrame):
            return data.loc[:, data.std() > 1e-8]
        return data

# Define missing functions that were previously imported from kan module
def sliding_window_alignment(data: pd.DataFrame, window_size: int, step: int) -> List[pd.DataFrame]:
    # This function seems generic, can be kept in a common place or moved to a more specific processor.
    # For now, keeping it here but noting it's a candidate for relocation.
    if data.empty or len(data) < window_size:
        return [data] if not data.empty else []
    
    windows = []
    for i in range(0, len(data) - window_size + 1, step):
        windows.append(data.iloc[i:i + window_size])
    return windows

from .processors.log_processors import UnifiedLogProcessor
from .processors.trace_processors import UnifiedTraceProcessor
from .processors.metric_processors import UnifiedMetricProcessor

def extract_log_features(log_data, **kwargs):
    """重定向到 UnifiedLogProcessor - 避免重複定義"""
    from .processors.log_processors import UnifiedLogProcessor
    processor = UnifiedLogProcessor(**kwargs)
    return processor.process(log_data)

def extract_trace_features(trace_data, **kwargs):
    """重定向到 UnifiedTraceProcessor - 避免重複定義"""
    from .processors.trace_processors import UnifiedTraceProcessor
    processor = UnifiedTraceProcessor(**kwargs)
    return processor.process(trace_data, **kwargs)

def extract_metric_features(metric_data, **kwargs):
    """重定向到 UnifiedMetricProcessor - 避免重複定義"""
    from .processors.metric_processors import UnifiedMetricProcessor
    processor = UnifiedMetricProcessor(**kwargs)
    return processor.process(metric_data)

def build_service_dependency_graph(trace_data):
    """簡化的服務依賴圖構建"""
    try:
        import networkx as nx
        G = nx.DiGraph()
        if isinstance(trace_data, pd.DataFrame) and 'serviceName' in trace_data.columns:
            services = trace_data['serviceName'].unique()
            for service in services:
                G.add_node(service)
        return G
    except:
        import networkx as nx
        return nx.DiGraph()

def extract_service_topology_features(service_graph):
    """簡化的服務拓撲特徵提取"""
    try:
        if service_graph is None or len(service_graph.nodes()) == 0:
            return np.array([[0]]), ['empty_graph']
        
        features = []
        names = []
        for node in service_graph.nodes():
            in_degree = service_graph.in_degree(node)
            out_degree = service_graph.out_degree(node)
            features.extend([in_degree, out_degree])
            names.extend([f'{node}_in_degree', f'{node}_out_degree'])
        
        if features:
            return np.array([features]), names
        else:
            return np.array([[0]]), ['no_topology_features']
    except:
        return np.array([[0]]), ['topology_error']

def compute_topology_features(adj_matrix, node_names):
    """計算拓撲特徵"""
    try:
        features = []
        names = []
        
        # 基本拓撲統計
        degrees = np.sum(adj_matrix, axis=1)
        features.extend([
            np.mean(degrees),
            np.std(degrees),
            np.max(degrees),
            np.sum(adj_matrix) / (adj_matrix.shape[0] * adj_matrix.shape[1])  # 密度
        ])
        names.extend(['avg_degree', 'std_degree', 'max_degree', 'graph_density'])
        
        return np.array(features), names
    except:
        return np.array([0]), ['topology_default']

def extract_error_features(data):
    """提取錯誤特徵"""
    try:
        if isinstance(data, pd.DataFrame):
            error_count = len(data[data.get('level', '').str.contains('ERROR', na=False)])
            warning_count = len(data[data.get('level', '').str.contains('WARN', na=False)])
        else:
            error_count = warning_count = 0
        
        return np.array([[error_count, warning_count]]), ['error_count', 'warning_count']
    except:
        return np.array([[0, 0]]), ['error_count', 'warning_count']

warnings.filterwarnings("ignore")


class MultiModalFeatureExtractor:
    """
    多模態特徵提取器 - 現在作為統一處理器的協調器。
    所有實際的處理邏輯都已移至 `processors` 子模組中。
    """
    def __init__(self, config):
        self.config = config
        self.log_processor = UnifiedLogProcessor.from_config(config)
        self.trace_processor = UnifiedTraceProcessor.from_config(config)
        self.metric_processor = UnifiedMetricProcessor.from_config(config)

    def extract_features(self, data_dict: dict, inject_time=None, dataset=None):
        """
        從多模態數據中提取特徵。
        
        Args:
            data_dict (dict): 包含 'metrics', 'logs', 'traces' 的字典。
            inject_time: 故障注入時間。

        Returns:
            fused_features (np.ndarray): 融合後的特徵矩陣。
            node_names (list): 節點名稱列表。
        """
        all_features = {}
        
        if 'metrics' in data_dict and data_dict['metrics'] is not None:
            metric_features, metric_names = self.metric_processor.process(
                data_dict['metrics'], inject_time=inject_time
            )
            all_features['metrics'] = (metric_features, metric_names)
            
        if 'traces' in data_dict and data_dict['traces'] is not None:
            trace_features, trace_names = self.trace_processor.process(
                data_dict['traces'], inject_time=inject_time
            )
            all_features['traces'] = (trace_features, trace_names)
            
        if 'logs' in data_dict and data_dict['logs'] is not None:
            log_features, log_names = self.log_processor.process(data_dict['logs'])
            all_features['logs'] = (log_features, log_names)
            
        # At this point, you would typically fuse the features.
        # This part of the logic needs to be robustly defined.
        # For now, we'll prioritize metric features as a placeholder for fusion.
        if 'metrics' in all_features:
            return all_features['metrics']
        elif 'traces' in all_features:
            return all_features['traces']
        elif 'logs' in all_features:
            return all_features['logs']
        
            return np.array([]), []


# simplified_metric_processing 函數已移至 feature_processing.py
# 避免重複代碼，統一使用 feature_processing 中的版本


def enhanced_trace_processing(trace_data, inject_time=None):
    """重定向到統一的trace處理器 - 避免重複定義"""
    from .processors.trace_processors import UnifiedTraceProcessor
    processor = UnifiedTraceProcessor()
    return processor.enhanced_process(trace_data, inject_time=inject_time)


# _build_enhanced_service_graph 函數已移至 feature_processing.py
# 避免重複代碼，統一使用 feature_processing 中的版本


def simplified_feature_fusion(log_feats, metric_feats, topo_feats, error_feats, trace_feats, service_topo_feats, 
                           fusion_method='simple_concat', target_dim=128):
    """
    簡化的特徵融合 - 移除複雜注意力機制，專注核心功能
    
    Args:
        log_feats: 日誌特徵
        metric_feats: 指標特徵  
        topo_feats: 拓撲特徵
        error_feats: 錯誤特徵
        trace_feats: trace 特徵
        service_topo_feats: 服務拓撲特徵
        fusion_method: 融合方法（簡化為 'simple_concat' 和 'weighted'）
        target_dim: 目標維度
    
    Returns:
        融合後的特徵
    """
    features_list = []
    feature_names = []
    
    # 收集所有可用的特徵 - 簡化檢查
    if log_feats is not None and log_feats.size > 0:
        features_list.append(log_feats)
        feature_names.append('log')
    
    if metric_feats is not None and metric_feats.size > 0:
        features_list.append(metric_feats)
        feature_names.append('metric')
        
    if trace_feats is not None and trace_feats.size > 0:
        features_list.append(trace_feats)
        feature_names.append('trace')
        
    # 簡化：只使用最重要的三種特徵，移除噪音來源
    if not features_list:
        print("⚠️ 沒有可用的特徵進行融合")
        return np.array([])
    
    # 簡化的特徵對齊
    min_length = min(f.shape[0] for f in features_list)
    aligned_features = []
    
    for features in features_list:
        if features.shape[0] > min_length:
            aligned_features.append(features[:min_length])
        else:
            aligned_features.append(features)
    
    # 簡化融合方法
    if fusion_method == 'weighted':
        # 簡化的加權融合：trace=0.5, metric=0.3, log=0.2
        weights = [0.2, 0.3, 0.5] if len(aligned_features) == 3 else [1.0/len(aligned_features)] * len(aligned_features)
        
        # 標準化到相同維度
        min_cols = min(f.shape[1] for f in aligned_features)
        normalized_features = [f[:, :min_cols] for f in aligned_features]
        
        # 加權組合
        fused_features = np.zeros_like(normalized_features[0])
        for features, weight in zip(normalized_features, weights):
            fused_features += weight * features
    else:
        # 簡單拼接
        fused_features = np.hstack(aligned_features)
        print(f"✓ 簡單拼接融合 {len(feature_names)} 種特徵")
    
    # 🎯 安全的PCA降維 - 使用統一安全函數
    from .utils import safe_pca_transform
    fused_features = safe_pca_transform(fused_features, target_dim)
    print(f"✓ 安全特徵融合完成: {fused_features.shape}")
    
    return fused_features


# 簡化的特徵融合函數 - 移除複雜的注意力機制
# 使用 simplified_feature_fusion 替代原有的 enhanced_feature_fusion

# 為向後兼容性保留別名
enhanced_feature_fusion = simplified_feature_fusion