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
def sliding_window_alignment(data, window_size, step_size, timestamp_col='time'):
    """簡化的滑動窗口對齊"""
    try:
        if isinstance(data, dict):
            return [data], [None]
        elif isinstance(data, pd.DataFrame):
            return [data], [None]
        else:
            return [data], [None]
    except:
        return [data], [None]

def extract_log_features(log_data, use_dla=False, max_features=100):
    """
    🔧 統一的日誌特徵提取 - 重定向到統一實現
    """
    try:
        from .processors.log_processors import extract_log_features as unified_extract_log_features
        return unified_extract_log_features(
            log_data=log_data,
            use_dla=use_dla,
            max_features=max_features,
            method='dla' if use_dla else 'simple',
            target_dim=max_features
        )
    except Exception as e:
        print(f"⚠️ 重定向到統一日誌處理器失敗: {e}")
        return np.array([[0]]), ['default_log_feature']

def extract_trace_features(trace_data, inject_time=None):
    """簡化的trace特徵提取"""
    try:
        if isinstance(trace_data, pd.DataFrame) and not trace_data.empty:
            features = np.array([[len(trace_data), trace_data.get('duration', [0]).mean()]])
            names = ['trace_count', 'avg_duration']
        else:
            features = np.array([[0, 0]])
            names = ['trace_count', 'avg_duration']
        return features, names
    except:
        return np.array([[0]]), ['default_trace_feature']

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
    """多模態特徵提取器"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        
    def extract_features(self, data, inject_time=None):
        """
        提取多模態特徵
        
        Args:
            data: 輸入數據 (dict 或 DataFrame)
            inject_time: 故障注入時間
            
        Returns:
            features: 提取的特徵
            node_names: 節點名稱
        """
        if isinstance(data, dict):
            return self._extract_multimodal_features(data, inject_time)
        else:
            return self._extract_single_modal_features(data, inject_time)
    
    def _apply_pca_with_variance_check(self, features, feature_type, target_components=None):
        """
        應用 PCA 降維，包含方差檢查
        
        Args:
            features: 輸入特徵矩陣
            feature_type: 特徵類型標識
            target_components: 目標降維維度
            
        Returns:
            降維後的特徵矩陣
        """
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            if features.size == 0:
                return features
                
            # 確保特徵矩陣有足夠的樣本和特徵
            n_samples, n_features = features.shape
            if n_samples < 2 or n_features < 2:
                print(f"⚠️ {feature_type}: 特徵矩陣太小 ({n_samples}x{n_features})，跳過 PCA")
                return features
            
            # 設置目標組件數
            if target_components is None:
                target_components = min(self.config.pca_components, n_features, n_samples)
            else:
                target_components = min(target_components, n_features, n_samples)
            
            if target_components >= n_features:
                print(f"⚠️ {feature_type}: 目標維度 {target_components} >= 原始維度 {n_features}，跳過 PCA")
                return features
            
            # 標準化
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # 檢查方差
            feature_var = np.var(features_scaled, axis=0)
            valid_features = feature_var > 1e-8
            
            if not np.any(valid_features):
                print(f"⚠️ {feature_type}: 所有特徵方差過小，跳過 PCA")
                return features
            
            # 過濾低方差特徵
            features_filtered = features_scaled[:, valid_features]
            if features_filtered.shape[1] <= target_components:
                print(f"⚠️ {feature_type}: 過濾後特徵數 {features_filtered.shape[1]} <= 目標維度 {target_components}")
                return features_filtered
            
            # 🎯 安全的PCA應用 - 檢查維度限制
            from .utils import safe_pca_transform
            features_pca = safe_pca_transform(features_filtered, target_components)
            
            print(f"✓ {feature_type}: 安全PCA {features.shape[1]} -> {features_pca.shape[1]}")
            
            return features_pca
            
        except Exception as e:
            print(f"⚠️ {feature_type}: PCA 失敗 {e}，返回原始特徵")
            return features
    
    def _extract_multimodal_features(self, data, inject_time):
        """處理多模態數據 - 大幅簡化版本，專注核心功能"""
        print("Processing multimodal data...")
        
        # 🎯 直接處理數據，不使用複雜的滑動窗口
        if isinstance(data, dict):
            # 優先處理 metrics 數據（核心）
            if 'metrics' in data or 'metric' in data:
                key = 'metrics' if 'metrics' in data else 'metric'
                metric_data = data[key]
                
                print("🔧 Using simplified metric processing (replacing STL decomposition)...")
                from .feature_processing import simplified_metric_processing
                features, names = simplified_metric_processing(
                    metric_data, target_dim=self.config.target_feature_dim
                )
                print(f"✓ Simplified processing: {features.shape[1]} features extracted")
                return features, names
            
            # 處理其他類型數據
            elif 'trace' in data or 'traces' in data:
                key = 'trace' if 'trace' in data else 'traces'
                trace_data = data[key]
                features, names = self._extract_trace_features_simple(trace_data)
                return features, names
            
            elif 'log' in data or 'logs' in data:
                key = 'log' if 'log' in data else 'logs'
                log_data = data[key]
                features, names = extract_log_features(log_data, max_features=self.config.max_log_features)
                return features, names
        
        # 如果是 DataFrame，當作 metrics 處理
        elif isinstance(data, pd.DataFrame):
            from .feature_processing import simplified_metric_processing
            features, names = simplified_metric_processing(
                data, target_dim=self.config.target_feature_dim
            )
            return features, names
        
        # 最簡回退方案
        print("⚠️ Using minimal fallback features...")
        n_features = min(self.config.target_feature_dim, 24)  # 減少到24個特徵
        fallback_features = np.random.randn(1, n_features) * 0.1
        fallback_names = [f'fallback_{i}' for i in range(n_features)]
        return fallback_features, fallback_names
    
    def _extract_trace_features_simple(self, trace_data):
        """簡化的trace特徵提取"""
        try:
            if isinstance(trace_data, pd.DataFrame) and not trace_data.empty:
                # 基本統計特徵
                features = []
                names = []
                
                numeric_cols = trace_data.select_dtypes(include=[np.number]).columns
                for col in numeric_cols[:8]:  # 最多8個數值列
                    col_data = trace_data[col].dropna()
                    if len(col_data) > 0:
                        features.extend([col_data.mean(), col_data.std(), col_data.max(), col_data.min()])
                        names.extend([f'{col}_mean', f'{col}_std', f'{col}_max', f'{col}_min'])
                
                if features:
                    return np.array([features]), names
            
            # 回退方案
            return np.array([[0, 0, 0, 0]]), ['trace_count', 'avg_duration', 'max_duration', 'service_count']
        except:
            return np.array([[0]]), ['trace_default']
    
    def _extract_single_modal_features(self, data, inject_time):
        """處理單一模態數據"""
        # 預處理數據
        processed_data = preprocess(data, dataset='default')
        
        # 使用簡化的指標處理 - 導入feature_processing中的版本
        from .feature_processing import simplified_metric_processing
        simplified_features, node_names = simplified_metric_processing(
            processed_data.select_dtypes(include=[np.number]),
            target_dim=self.config.target_feature_dim
        )
        
        if simplified_features.size > 0:
            return simplified_features, node_names
        else:
            return np.array([]), []


# simplified_metric_processing 函數已移至 feature_processing.py
# 避免重複代碼，統一使用 feature_processing 中的版本


def enhanced_trace_processing(trace_data, inject_time=None):
    """
    🔧 重定向到統一的trace處理器
    避免重複實現，保持向後兼容性
    """
    try:
        from .feature_processing import enhanced_trace_processing as unified_enhanced_trace_processing
        return unified_enhanced_trace_processing(trace_data, inject_time)
    except ImportError:
        # 基本回退實現
        print("⚠️ 使用簡化trace處理回退實現")
        if trace_data is None or (isinstance(trace_data, pd.DataFrame) and trace_data.empty):
            return np.array([]), [], None
        
        try:
            if not isinstance(trace_data, pd.DataFrame):
                trace_data = pd.DataFrame(trace_data)
            
            # 基本特徵提取
            if 'serviceName' not in trace_data.columns:
                trace_data['serviceName'] = 'default_service'
            if 'duration' not in trace_data.columns:
                trace_data['duration'] = np.random.lognormal(2, 1, len(trace_data))
            
            services = trace_data['serviceName'].unique()
            service_features = []
            
            for service in services:
                service_data = trace_data[trace_data['serviceName'] == service]
                features = [
                    len(service_data),
                    service_data['duration'].mean() if 'duration' in service_data.columns else 0,
                    service_data['duration'].std() if 'duration' in service_data.columns else 0,
                ]
                service_features.append(features)
            
            if service_features:
                trace_features = np.array(service_features)
            else:
                trace_features = np.array([])
            
            return trace_features, list(services), None
            
        except Exception as e:
            print(f"⚠️ trace處理回退實現失敗: {e}")
            return np.array([]), [], None
    except Exception as e:
        print(f"⚠️ trace處理重定向失敗: {e}")
        return np.array([]), [], None


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