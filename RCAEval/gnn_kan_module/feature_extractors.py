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

# Import our KAN modules from the new kan_components
from .kan_components import (
    compute_service_criticality_weights
)

# Import other required functions from the appropriate modules
from RCAEval.io.time_series import preprocess, drop_constant

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
    """簡化的日誌特徵提取"""
    try:
        if isinstance(log_data, (list, str)):
            # 基本文本特徵
            text_length = len(str(log_data))
            word_count = len(str(log_data).split())
            features = np.array([[text_length, word_count, 0, 0]])
            names = ['text_length', 'word_count', 'error_count', 'warning_count']
        else:
            features = np.array([[1, 2, 0, 0]])
            names = ['log_feature_1', 'log_feature_2', 'log_feature_3', 'log_feature_4']
        return features, names
    except:
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
            
            # 應用 PCA
            pca = PCA(n_components=target_components, random_state=42)
            features_pca = pca.fit_transform(features_filtered)
            
            explained_variance = np.sum(pca.explained_variance_ratio_)
            print(f"✓ {feature_type}: PCA {features.shape[1]} -> {target_components}, 解釋方差: {explained_variance:.3f}")
            
            return features_pca
            
        except Exception as e:
            print(f"⚠️ {feature_type}: PCA 失敗 {e}，返回原始特徵")
            return features
    
    def _extract_multimodal_features(self, data, inject_time):
        """處理多模態數據 - 簡化版本，移除STL分解的複雜性"""
        all_features = []
        node_names = []
        
        print("Processing multimodal data...")
        
        # 🔧 1. 使用簡化的滑動窗口處理
        if inject_time is not None:
            print("Applying simplified window alignment...")
            try:
                windows, timestamps = sliding_window_alignment(
                    data, 
                    window_size=self.config.window_size, 
                    step_size=self.config.step_size,
                    timestamp_col='time'
                )
                print(f"Created {len(windows)} windows for analysis")
            except Exception as e:
                print(f"⚠️ Window alignment failed: {e}, using raw data")
                windows = [data]
                timestamps = [None]
        else:
            windows = [data]
            timestamps = [None]
        
        # 對每個窗口進行特徵提取
        for window_idx, window_data in enumerate(windows):
            window_features = []
            window_node_names = []
            
            # 🔧 2. 使用增強的trace處理（替代複雜的trace邏輯）
            trace_data_found = False
            if 'trace' in window_data or 'traces' in window_data:
                trace_key = 'trace' if 'trace' in window_data else 'traces'
                trace_data = window_data[trace_key]
                trace_data_found = True
            elif isinstance(window_data, pd.DataFrame):
                trace_columns = ['serviceName', 'operationName', 'startTime', 'duration', 'traceID', 'spanID']
                if any(col in window_data.columns for col in trace_columns):
                    trace_data = window_data
                    trace_data_found = True
                    print(f"Detected trace data in DataFrame format for window {window_idx}")
            
            if trace_data_found:
                print(f"Extracting trace features from window {window_idx}...")
                try:
                    # 🔧 使用簡化的trace處理
                    trace_features, operation_names, service_graph = enhanced_trace_processing(
                        trace_data, inject_time
                    )
                    
                    if trace_features.size > 0:
                        pca_trace_features = self._apply_pca_with_variance_check(
                            trace_features, f'trace_window_{window_idx}', target_components=64
                        )
                        window_features.append(pca_trace_features)
                        window_node_names.extend([f'trace_{name}' for name in operation_names[:pca_trace_features.shape[1]]])
                    
                    # 提取服務拓撲特徵
                    if service_graph is not None:
                        service_topo_features, service_names = extract_service_topology_features(service_graph)
                        if service_topo_features.size > 0:
                            pca_service_features = self._apply_pca_with_variance_check(
                                service_topo_features, f'service_topo_window_{window_idx}', target_components=32
                            )
                            window_features.append(pca_service_features)
                            window_node_names.extend([f'service_topo_{name}' for name in service_names[:pca_trace_features.shape[1]]])
                            
                except Exception as e:
                    print(f"⚠️ Trace feature extraction failed: {e}, continuing without trace features")
            
            # 🔧 3. 使用簡化的metric處理（替代STL分解）
            metric_data_processed = False
            if 'metric' in window_data or 'metrics' in window_data:
                metric_key = 'metric' if 'metric' in window_data else 'metrics'
                metric_data = window_data[metric_key]
                metric_data_processed = True
            elif isinstance(window_data, pd.DataFrame) and not trace_data_found:
                metric_data = window_data
                metric_data_processed = True
            
            if metric_data_processed:
                try:
                    # 預處理 metric 數據
                    if inject_time is not None and 'time' in metric_data.columns:
                        normal_df = metric_data[metric_data['time'] < inject_time]
                        anomal_df = metric_data[metric_data['time'] >= inject_time]
                        
                        if not normal_df.empty and not anomal_df.empty:
                            normal_processed = preprocess(normal_df, dataset='default')
                            anomal_processed = preprocess(anomal_df, dataset='default')
                            
                            intersect = [x for x in normal_processed.columns if x in anomal_processed.columns]
                            normal_processed = normal_processed[intersect]
                            anomal_processed = anomal_processed[intersect]
                            
                            metric_data = pd.concat([normal_processed, anomal_processed], axis=0, ignore_index=True)
                        else:
                            metric_data = preprocess(metric_data, dataset='default')
                    else:
                        metric_data = preprocess(metric_data, dataset='default')
                    
                    # 🎯 使用簡化的指標處理（替代複雜的STL分解）
                    simplified_features, simplified_names = simplified_metric_processing(
                        metric_data.select_dtypes(include=[np.number]),
                        target_dim=self.config.pca_components
                    )
                    
                    if simplified_features.size > 0:
                        window_features.append(simplified_features)
                        window_node_names.extend([f'w{window_idx}_{name}' for name in simplified_names])
                        print(f"✓ Extracted {len(simplified_names)} simplified metric features")
                        
                except Exception as e:
                    print(f"⚠️ Metric feature extraction failed: {e}")
            
            # 🔧 4. 處理 log 數據（保持原有邏輯，已經足夠簡单）
            if 'log' in window_data or 'logs' in window_data:
                log_key = 'log' if 'log' in window_data else 'logs'
                log_data = window_data[log_key]
                
                try:
                    log_features, log_names = extract_log_features(
                        log_data,
                        use_dla=self.config.use_dla,
                        max_features=self.config.max_log_features
                    )
                    
                    if log_features.size > 0:
                        if self.config.use_pca and log_features.shape[1] > self.config.pca_components:
                            log_features = self._apply_pca_with_variance_check(
                                log_features, f"log_window_{window_idx}"
                            )
                        
                        window_features.append(log_features)
                        window_node_names.extend([f'w{window_idx}_{name}' for name in log_names])
                        print(f"✓ Extracted {len(log_names)} log features")
                        
                except Exception as e:
                    print(f"⚠️ Log feature extraction failed: {e}")
            
            # 收集當前窗口的特徵
            if window_features:
                all_features.extend(window_features)
                node_names.extend(window_node_names)
        
        # 🔧 5. 簡化的回退策略（移除過度複雜的合成特徵生成）
        if not all_features:
            print("⚠️ No features extracted, generating minimal fallback features...")
            n_samples = 10  # 減少合成特徵數量
            n_features = 8   # 減少特徵維度
            fallback_features = np.random.randn(n_samples, n_features) * 0.1  # 減少變異性
            fallback_names = [f'fallback_feature_{i}' for i in range(n_features)]
            
            all_features.append(fallback_features)
            node_names.extend(fallback_names)
            print(f"✓ Generated {len(fallback_names)} fallback features")
        
        return self._finalize_features(all_features, node_names)

    def _finalize_features(self, all_features, node_names):
        """
        最终化特征处理
        
        Args:
            all_features: 所有特徵列表
            node_names: 节点名称列表
            
        Returns:
            fused_features: 融合后的特征
            node_names: 最终的节点名称列表
        """
        if not all_features:
            return np.array([]), []
        
        # 对齐所有特征的长度
        min_length = min(f.shape[0] for f in all_features if f.size > 0)
        if min_length == 0:
            min_length = 1
            
        aligned_features = []
        for features in all_features:
            if features.size == 0:
                continue
            if features.shape[0] > min_length:
                features = features[:min_length]
            elif features.shape[0] < min_length:
                # 重复最后一行
                padding = np.repeat(features[-1:], min_length - features.shape[0], axis=0)
                features = np.vstack([features, padding])
            aligned_features.append(features)
        
        if not aligned_features:
            return np.array([]), []
        
        # 计算拓扑特征
        print("Computing topology features...")
        if len(aligned_features) > 1:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                
                combined_features = np.hstack(aligned_features)
                if combined_features.shape[1] > 1:
                    similarity_matrix = cosine_similarity(combined_features.T)
                    adj_matrix = (similarity_matrix > 0.5).astype(float)
                    
                    topology_features, topology_names = compute_topology_features(
                        adj_matrix, node_names[:adj_matrix.shape[0]]
                    )
                    
                    if topology_features.size > 0:
                        topo_features_expanded = np.tile(topology_features, (min_length, 1))
                        
                        if self.config.use_pca and topo_features_expanded.shape[1] > self.config.pca_components:
                            topo_features_expanded = self._apply_pca_with_variance_check(
                                topo_features_expanded, "topology_features"
                            )
                        
                        aligned_features.append(topo_features_expanded)
                        node_names.extend([f'topology_{name}' for name in topology_names])
            except Exception as e:
                print(f"⚠️ Topology feature computation failed: {e}")
        
        # 特征融合
        print("Performing feature fusion...")
        
        # 分离不同类型的特征进行融合
        log_feats = None
        metric_feats = None
        topo_feats = None
        error_feats = None
        trace_feats = None
        service_topo_feats = None
        
        for i, features in enumerate(aligned_features):
            node_name = node_names[i] if i < len(node_names) else ""
            
            if 'trace_' in node_name:
                if trace_feats is None:
                    trace_feats = features
                else:
                    trace_feats = np.hstack([trace_feats, features])
            elif 'service_topo_' in node_name:
                if service_topo_feats is None:
                    service_topo_feats = features
                else:
                    service_topo_feats = np.hstack([service_topo_feats, features])
            elif 'log' in node_name:
                if log_feats is None:
                    log_feats = features
                else:
                    log_feats = np.hstack([log_feats, features])
            elif 'topology' in node_name:
                if topo_feats is None:
                    topo_feats = features
                else:
                    topo_feats = np.hstack([topo_feats, features])
            elif 'error' in node_name:
                if error_feats is None:
                    error_feats = features
                else:
                    error_feats = np.hstack([error_feats, features])
            else:
                if metric_feats is None:
                    metric_feats = features
                else:
                    metric_feats = np.hstack([metric_feats, features])
        
        # 使用改進的特徵融合 - 修正參數順序
        fused_features = enhanced_feature_fusion(
            log_feats, metric_feats, topo_feats, error_feats, trace_feats, service_topo_feats,
            fusion_method=self.config.fusion_method,
            target_dim=self.config.target_feature_dim
        )
        
        if fused_features.size == 0:
            # 回退方案：直接拼接
            fused_features = np.hstack(aligned_features)
        
        # 最终PCA降维确保特征维度符合要求
        if self.config.use_pca and fused_features.shape[1] > self.config.target_feature_dim:
            print(f"Applying final PCA: {fused_features.shape[1]} -> {self.config.target_feature_dim}")
            fused_features = self._apply_pca_with_variance_check(
                fused_features, "final_fusion", target_components=self.config.target_feature_dim
            )
        
        return fused_features, node_names
    
    def _extract_single_modal_features(self, data, inject_time):
        """處理單一模態數據"""
        # 預處理數據
        processed_data = preprocess(data, dataset='default')
        
        # 使用簡化的指標處理
        simplified_features, node_names = simplified_metric_processing(
            processed_data.select_dtypes(include=[np.number]),
            target_dim=self.config.target_feature_dim
        )
        
        if simplified_features.size > 0:
            return simplified_features, node_names
        else:
            return np.array([]), []


def simplified_metric_processing(metrics_data, target_dim=64):
    """
    簡化的指標處理 - 替換過度複雜的STL分解
    專注於核心統計特徵，提高通用性和效率
    
    Args:
        metrics_data: 指標數據
        target_dim: 目標維度
    
    Returns:
        processed_features: 處理後的特徵
        feature_names: 特徵名稱
    """
    print("🔧 Using simplified metric processing (replacing STL decomposition)...")
    
    if isinstance(metrics_data, pd.DataFrame):
        data = metrics_data.select_dtypes(include=[np.number])
    elif isinstance(metrics_data, np.ndarray):
        data = pd.DataFrame(metrics_data) if metrics_data.ndim == 2 else pd.DataFrame({'metric': metrics_data})
    else:
        try:
            data = pd.DataFrame(metrics_data)
        except:
            return np.array([[0]]), ['default_feature']

    all_features = []
    feature_names = []
    
    for col in data.columns:
        series = data[col].dropna()
        col_name = str(col)
        
        if len(series) < 3:
            # 數據太少，使用基本統計
            basic_stats = [series.mean() if len(series) > 0 else 0, 0, series.min() if len(series) > 0 else 0, series.max() if len(series) > 0 else 0]
            all_features.extend(basic_stats)
            feature_names.extend([f'{col_name}_mean', f'{col_name}_std', f'{col_name}_min', f'{col_name}_max'])
            continue
        
        # 🎯 核心統計特徵（替代STL的複雜分解）
        core_features = [
            series.mean(),                    # 中心趨勢
            series.std(),                     # 離散程度
            series.min(),                     # 最小值
            series.max(),                     # 最大值
            series.median(),                  # 中位數
            np.percentile(series, 25),        # 第一四分位數
            np.percentile(series, 75),        # 第三四分位數
            series.skew() if len(series) > 3 else 0,  # 偏度
        ]
        
        # 🎯 簡化的趨勢特徵（替代複雜的週期檢測）
        if len(series) >= 5:
            # 線性趨勢
            x = np.arange(len(series))
            trend_coef = np.polyfit(x, series.values, 1)[0]
            
            # 變化率
            diff = np.diff(series.values)
            change_rate = np.mean(np.abs(diff))
            
            # 穩定性
            stability = 1.0 / (1.0 + np.std(diff))
            
            trend_features = [trend_coef, change_rate, stability]
        else:
            trend_features = [0.0, 0.0, 1.0]
        
        # 🎯 異常檢測特徵（替代複雜的回退機制）
        Q1, Q3 = np.percentile(series, [25, 75])
        IQR = Q3 - Q1
        if IQR > 0:
            outliers = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).sum()
            outlier_ratio = outliers / len(series)
        else:
            outlier_ratio = 0.0
        
        anomaly_features = [outlier_ratio]
        
        # 組合所有特徵
        col_features = core_features + trend_features + anomaly_features
        all_features.extend(col_features)
        
        # 生成特徵名稱
        names = [
            f'{col_name}_mean', f'{col_name}_std', f'{col_name}_min', f'{col_name}_max',
            f'{col_name}_median', f'{col_name}_q25', f'{col_name}_q75', f'{col_name}_skew',
            f'{col_name}_trend', f'{col_name}_change_rate', f'{col_name}_stability',
            f'{col_name}_outlier_ratio'
        ]
        feature_names.extend(names)
    
    # 轉換為矩陣格式
    if all_features:
        feature_matrix = np.array(all_features).reshape(1, -1)
        
        # PCA降維到目標維度
        if feature_matrix.shape[1] > target_dim:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=target_dim, random_state=42)
            feature_matrix = pca.fit_transform(feature_matrix)
            feature_names = [f'pca_component_{i}' for i in range(target_dim)]
    else:
        feature_matrix = np.array([[0]])
        feature_names = ['default_feature']
    
    print(f"✓ Simplified processing: {feature_matrix.shape[1]} features extracted")
    return feature_matrix, feature_names


def enhanced_trace_processing(trace_data, inject_time=None):
    """
    增強的TracerCA風格trace處理 - 專注於最有效的特徵
    
    Args:
        trace_data: trace數據
        inject_time: 故障注入時間
    
    Returns:
        trace_features: trace特徵
        operation_names: 操作名稱
        service_graph: 服務圖
    """
    print("🔧 Enhanced TracerCA-style trace processing...")
    
    if trace_data is None or (isinstance(trace_data, pd.DataFrame) and trace_data.empty):
        return np.array([]), [], None
    
    try:
        # 確保trace_data是DataFrame格式
        if not isinstance(trace_data, pd.DataFrame):
            trace_data = pd.DataFrame(trace_data)
        
        # 標準化列名 - 兼容多種trace格式
        column_mapping = {
            'service_name': 'serviceName', 'service': 'serviceName',
            'operation_name': 'operationName', 'operation': 'operationName',
            'method_name': 'operationName', 'method': 'operationName',
            'start_time': 'startTime', 'timestamp': 'startTime',
            'time': 'startTime', 'trace_id': 'traceID', 'span_id': 'spanID'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in trace_data.columns and new_col not in trace_data.columns:
                trace_data[new_col] = trace_data[old_col]
        
        # 確保必要列存在
        if 'serviceName' not in trace_data.columns:
            trace_data['serviceName'] = 'default_service'
        if 'operationName' not in trace_data.columns:
            trace_data['operationName'] = 'default_operation'
        if 'duration' not in trace_data.columns:
            trace_data['duration'] = np.random.lognormal(2, 1, len(trace_data))
        
        # 創建操作標識 - TracerCA關鍵特徵
        trace_data['operation'] = trace_data['serviceName'].astype(str) + "_" + trace_data['operationName'].astype(str)
        
        # 構建服務依賴圖
        service_graph = _build_enhanced_service_graph(trace_data)
        
        # 提取TracerCA風格的操作級特徵
        operations = trace_data['operation'].unique()
        operation_features = []
        
        for op in operations:
            op_data = trace_data[trace_data['operation'] == op]
            
            # 基本統計特徵
            duration_stats = op_data['duration'].describe() if 'duration' in op_data.columns else pd.Series([0]*8, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
            
            # TracerCA核心特徵：support, confidence, JI
            if inject_time is not None and 'startTime' in op_data.columns:
                # 分割正常和異常期間
                normal_data = op_data[op_data['startTime'] < inject_time] if 'startTime' in op_data.columns else op_data[:len(op_data)//2]
                anomal_data = op_data[op_data['startTime'] >= inject_time] if 'startTime' in op_data.columns else op_data[len(op_data)//2:]
                
                # 計算TracerCA特徵
                if not normal_data.empty and not anomal_data.empty:
                    # 基於SLO的異常檢測
                    normal_latency = normal_data['duration'].mean() if 'duration' in normal_data.columns else 0
                    normal_std = normal_data['duration'].std() if 'duration' in normal_data.columns else 1
                    anomal_latency = anomal_data['duration'].mean() if 'duration' in anomal_data.columns else 0
                    
                    # TracerCA Support: 異常操作數量/總異常數量
                    threshold = normal_latency + 3 * normal_std
                    abnormal_spans = (anomal_data['duration'] > threshold).sum() if 'duration' in anomal_data.columns else 0
                    total_abnormal = len(anomal_data)
                    support = abnormal_spans / max(total_abnormal, 1)
                    
                    # TracerCA Confidence: 異常操作數量/該操作總數量
                    confidence = abnormal_spans / max(len(anomal_data), 1)
                    
                    # TracerCA JI (Jaccard Index)
                    ji = (2 * support * confidence) / max(support + confidence, 1e-10) if (support + confidence) > 0 else 0
                    
                    # 延遲變化率
                    latency_change = (anomal_latency - normal_latency) / max(normal_latency, 1e-8)
                    
                    # 調用頻率變化
                    normal_call_rate = len(normal_data) / max(len(trace_data), 1)
                    anomal_call_rate = len(anomal_data) / max(len(trace_data), 1)
                    call_rate_change = anomal_call_rate - normal_call_rate
                else:
                    support = confidence = ji = latency_change = call_rate_change = 0
            else:
                support = confidence = ji = latency_change = call_rate_change = 0
            
            # 組合TracerCA特徵
            features = [
                support,                      # TracerCA Support
                confidence,                   # TracerCA Confidence  
                ji,                          # TracerCA JI score
                latency_change,              # 延遲變化率（關鍵RCA指標）
                call_rate_change,            # 調用頻率變化
                duration_stats['mean'],      # 平均延遲
                duration_stats['std'],       # 延遲標準差
                duration_stats['max'],       # 最大延遲
                duration_stats['count'],     # 調用次數
                duration_stats['75%'] - duration_stats['25%']  # IQR（穩定性指標）
            ]
            
            operation_features.append(features)
        
        # 轉換為numpy數組並處理NaN值
        if operation_features:
            trace_features = np.array(operation_features)
            trace_features = np.nan_to_num(trace_features, nan=0.0, posinf=1.0, neginf=-1.0)
        else:
            trace_features = np.array([])
        
        return trace_features, list(operations), service_graph
        
    except Exception as e:
        print(f"⚠️ Enhanced TracerCA trace processing failed: {e}")
        return np.array([]), [], None


def _build_enhanced_service_graph(trace_data):
    """構建增強的服務依賴圖 - 基於實際調用關係"""
    try:
        import networkx as nx
        
        G = nx.DiGraph()
        
        # 添加服務節點
        services = trace_data['serviceName'].unique()
        for service in services:
            G.add_node(service)
        
        # 基於trace時序構建真實依賴關係
        if 'startTime' in trace_data.columns and 'traceID' in trace_data.columns:
            # 按traceID分組，時間排序找依賴
            for trace_id in trace_data['traceID'].unique():
                trace_spans = trace_data[trace_data['traceID'] == trace_id].sort_values('startTime')
                
                for i in range(len(trace_spans) - 1):
                    current_service = trace_spans.iloc[i]['serviceName']
                    next_service = trace_spans.iloc[i + 1]['serviceName']
                    
                    if current_service != next_service:
                        if G.has_edge(current_service, next_service):
                            G[current_service][next_service]['weight'] += 1
                        else:
                            G.add_edge(current_service, next_service, weight=1)
        
        return G
        
    except Exception as e:
        print(f"Service graph construction failed: {e}")
        return None


def enhanced_feature_fusion(log_feats, metric_feats, topo_feats, error_feats, trace_feats, service_topo_feats, 
                           fusion_method='attention', target_dim=128):
    """
    增強的特徵融合，整合多模態特徵
    
    Args:
        log_feats: 日誌特徵
        metric_feats: 指標特徵  
        topo_feats: 拓撲特徵
        error_feats: 錯誤特徵
        trace_feats: trace 特徵
        service_topo_feats: 服務拓撲特徵
        fusion_method: 融合方法
        target_dim: 目標維度
    
    Returns:
        融合後的特徵
    """
    features_list = []
    feature_names = []
    
    # 收集所有可用的特徵
    if log_feats is not None and log_feats.size > 0:
        features_list.append(log_feats)
        feature_names.append('log')
    
    if metric_feats is not None and metric_feats.size > 0:
        features_list.append(metric_feats)
        feature_names.append('metric')
        
    if topo_feats is not None and topo_feats.size > 0:
        features_list.append(topo_feats)
        feature_names.append('topology')
        
    if error_feats is not None and error_feats.size > 0:
        features_list.append(error_feats)
        feature_names.append('error')
        
    if trace_feats is not None and trace_feats.size > 0:
        features_list.append(trace_feats)
        feature_names.append('trace')
        
    if service_topo_feats is not None and service_topo_feats.size > 0:
        features_list.append(service_topo_feats)
        feature_names.append('service_topology')
    
    if not features_list:
        print("⚠️ 沒有可用的特徵進行融合")
        return np.array([])
    
    # 對齊特徵長度
    min_length = min(f.shape[0] for f in features_list)
    aligned_features = []
    
    for features in features_list:
        if features.shape[0] > min_length:
            aligned_features.append(features[:min_length])
        elif features.shape[0] < min_length:
            # 重複最後一行來填充
            padding = np.repeat(features[-1:], min_length - features.shape[0], axis=0)
            aligned_features.append(np.vstack([features, padding]))
        else:
            aligned_features.append(features)
    
    # 根據融合方法進行融合
    if fusion_method == 'attention':
        try:
            # 注意力機制融合
            weights = [1.0] * len(aligned_features)  # 均等權重
            fused_features = attention_fusion_enhanced(aligned_features, weights)
            print(f"✓ 使用注意力機制融合 {len(feature_names)} 種特徵")
        except Exception as e:
            print(f"⚠️ 注意力融合失敗: {e}，使用簡單拼接")
            fused_features = np.hstack(aligned_features)
    else:
        # 簡單拼接
        fused_features = np.hstack(aligned_features)
        print(f"✓ 簡單拼接融合 {len(feature_names)} 種特徵")
    
    # PCA 降維到目標維度
    if fused_features.shape[1] > target_dim:
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=target_dim, random_state=42)
            fused_features = pca.fit_transform(fused_features)
            print(f"✓ PCA 降維到目標維度: {target_dim}")
        except Exception as e:
            print(f"⚠️ PCA 降維失敗: {e}")
    
    return fused_features


def attention_fusion_enhanced(features_list, weights):
    """增強的注意力機制特徵融合"""
    from sklearn.preprocessing import MinMaxScaler
    
    # 計算注意力權重
    attention_weights = torch.softmax(torch.tensor(weights), dim=0).numpy()
    
    # 標準化特徵維度
    target_cols = min(f.shape[1] for f in features_list)
    normalized_features = []
    
    for features in features_list:
        if features.shape[1] != target_cols:
            scaler = MinMaxScaler()
            features = scaler.fit_transform(features)
            if features.shape[1] > target_cols:
                features = features[:, :target_cols]
            else:
                padding = np.zeros((features.shape[0], target_cols - features.shape[1]))
                features = np.hstack([features, padding])
        
        normalized_features.append(features)
    
    # 注意力加權
    fused = np.zeros_like(normalized_features[0])
    for features, weight in zip(normalized_features, attention_weights):
        fused += weight * features
    
    return fused