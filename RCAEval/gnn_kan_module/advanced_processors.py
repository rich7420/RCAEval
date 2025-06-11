"""
GNN-KAN Advanced Processors Module
包含高級特徵處理和多模態融合相關的類和函數
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time


class EnhancedMultiModalAttention(nn.Module):
    """增強的多模態注意力機制 - 改進特徵融合"""
    
    def __init__(self, feature_dims, hidden_dim=128, num_heads=8):
        """
        初始化多模態注意力機制
        
        Args:
            feature_dims: 各模態特徵維度列表
            hidden_dim: 隱藏層維度
            num_heads: 注意力頭數
        """
        super(EnhancedMultiModalAttention, self).__init__()
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 為每種模態創建投影層
        self.modal_projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in feature_dims
        ])
        
        # 多頭注意力機制
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 模態權重學習
        self.modal_weight_net = nn.Sequential(
            nn.Linear(hidden_dim * len(feature_dims), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(feature_dims)),
            nn.Softmax(dim=-1)
        )
        
        # 最終融合層
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 層歸一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, modal_features):
        """
        前向傳播
        
        Args:
            modal_features: 各模態特徵列表
            
        Returns:
            fused_features: 融合後的特徵
        """
        batch_size = modal_features[0].size(0)
        
        # 投影各模態特徵到統一空間
        projected_features = []
        for i, features in enumerate(modal_features):
            projected = self.modal_projections[i](features)
            projected_features.append(projected)
        
        # 堆疊為序列格式 [batch_size, num_modals, hidden_dim]
        stacked_features = torch.stack(projected_features, dim=1)
        
        # 應用多頭注意力
        attended_features, attention_weights = self.multihead_attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # 計算自適應模態權重
        flattened_features = attended_features.view(batch_size, -1)
        modal_weights = self.modal_weight_net(flattened_features)
        
        # 加權融合
        weighted_features = torch.sum(
            attended_features * modal_weights.unsqueeze(-1), 
            dim=1
        )
        
        # 最終融合和歸一化
        fused_features = self.fusion_layer(weighted_features)
        fused_features = self.layer_norm(fused_features + weighted_features)  # 殘差連接
        
        return fused_features


class AdvancedFeatureProcessor:
    """高級特徵處理器 - 結合多種先進技術"""
    
    def __init__(self, config):
        self.config = config
        self.feature_cache = {}
        
    def process_multimodal_features(self, data_dict, inject_time=None):
        """
        處理多模態特徵
        
        Args:
            data_dict: 多模態數據字典
            inject_time: 故障注入時間
            
        Returns:
            processed_features: 處理後的特徵
            feature_metadata: 特徵元數據
        """
        print("🔬 Advanced multimodal feature processing...")
        
        processed_modals = {}
        feature_metadata = {
            'modal_types': [],
            'feature_counts': {},
            'processing_methods': {}
        }
        
        # 處理trace數據
        if 'trace' in data_dict or 'traces' in data_dict:
            trace_key = 'trace' if 'trace' in data_dict else 'traces'
            trace_features, trace_meta = self._process_trace_modal(
                data_dict[trace_key], inject_time
            )
            if trace_features is not None:
                processed_modals['trace'] = trace_features
                feature_metadata['modal_types'].append('trace')
                feature_metadata['feature_counts']['trace'] = trace_features.shape[1]
                feature_metadata['processing_methods']['trace'] = trace_meta
        
        # 處理metric數據
        if 'metric' in data_dict or 'metrics' in data_dict:
            metric_key = 'metric' if 'metric' in data_dict else 'metrics'
            metric_features, metric_meta = self._process_metric_modal(
                data_dict[metric_key], inject_time
            )
            if metric_features is not None:
                processed_modals['metric'] = metric_features
                feature_metadata['modal_types'].append('metric')
                feature_metadata['feature_counts']['metric'] = metric_features.shape[1]
                feature_metadata['processing_methods']['metric'] = metric_meta
        
        # 處理log數據
        if 'log' in data_dict or 'logs' in data_dict:
            log_key = 'log' if 'log' in data_dict else 'logs'
            log_features, log_meta = self._process_log_modal(
                data_dict[log_key], inject_time
            )
            if log_features is not None:
                processed_modals['log'] = log_features
                feature_metadata['modal_types'].append('log')
                feature_metadata['feature_counts']['log'] = log_features.shape[1]
                feature_metadata['processing_methods']['log'] = log_meta
        
        # 特徵對齊和融合
        if processed_modals:
            aligned_features = self._align_modal_features(processed_modals)
            fused_features = self._advanced_feature_fusion(aligned_features)
            
            feature_metadata['final_feature_count'] = fused_features.shape[1]
            feature_metadata['fusion_method'] = 'enhanced_multimodal_attention'
            
            return fused_features, feature_metadata
        else:
            print("⚠️ No valid modal features processed")
            return None, feature_metadata
    
    def _process_trace_modal(self, trace_data, inject_time):
        """處理trace模態特徵"""
        try:
            # 使用增強的trace處理
            from .feature_extractors import enhanced_trace_processing
            
            trace_features, operation_names, service_graph = enhanced_trace_processing(
                trace_data, inject_time
            )
            
            if trace_features.size > 0:
                # 增加高級trace特徵
                advanced_features = self._extract_advanced_trace_features(
                    trace_data, inject_time, service_graph
                )
                
                if advanced_features.size > 0:
                    trace_features = np.hstack([trace_features, advanced_features])
                
                metadata = {
                    'method': 'enhanced_tracerca_style',
                    'operations_count': len(operation_names),
                    'service_graph_available': service_graph is not None,
                    'advanced_features': True
                }
                
                return torch.tensor(trace_features, dtype=torch.float), metadata
            else:
                return None, {}
                
        except Exception as e:
            print(f"⚠️ Trace modal processing failed: {e}")
            return None, {}
    
    def _process_metric_modal(self, metric_data, inject_time):
        """處理metric模態特徵"""
        try:
            # 使用PSM方法處理指標
            psm_features, psm_names = psm_metric_processing(
                metric_data, target_dim=self.config.target_feature_dim // 2
            )
            
            if psm_features.size > 0:
                # 增加時間序列分析特徵
                ts_features = self._extract_time_series_features(metric_data, inject_time)
                
                if ts_features.size > 0:
                    psm_features = np.hstack([psm_features, ts_features])
                
                metadata = {
                    'method': 'phase_space_method',
                    'feature_names': psm_names,
                    'time_series_enhanced': ts_features.size > 0
                }
                
                return torch.tensor(psm_features, dtype=torch.float), metadata
            else:
                return None, {}
                
        except Exception as e:
            print(f"⚠️ Metric modal processing failed: {e}")
            return None, {}
    
    def _process_log_modal(self, log_data, inject_time):
        """處理log模態特徵"""
        try:
            # 使用現有的log特徵提取
            from .feature_extractors import extract_log_features
            
            log_features, log_names = extract_log_features(
                log_data,
                use_dla=self.config.use_dla,
                max_features=self.config.max_log_features
            )
            
            if log_features.size > 0:
                # 增加語義特徵
                semantic_features = self._extract_log_semantic_features(log_data)
                
                if semantic_features.size > 0:
                    log_features = np.hstack([log_features, semantic_features])
                
                metadata = {
                    'method': 'enhanced_log_extraction',
                    'feature_names': log_names,
                    'semantic_enhanced': semantic_features.size > 0,
                    'use_dla': self.config.use_dla
                }
                
                return torch.tensor(log_features, dtype=torch.float), metadata
            else:
                return None, {}
                
        except Exception as e:
            print(f"⚠️ Log modal processing failed: {e}")
            return None, {}
    
    def _extract_advanced_trace_features(self, trace_data, inject_time, service_graph):
        """提取高級trace特徵"""
        try:
            if not isinstance(trace_data, pd.DataFrame):
                trace_data = pd.DataFrame(trace_data)
            
            advanced_features = []
            
            # 服務依賴深度特徵
            if service_graph is not None:
                try:
                    import networkx as nx
                    # 計算圖的拓撲特徵
                    if isinstance(service_graph, nx.Graph):
                        centrality_features = [
                            np.mean(list(nx.degree_centrality(service_graph).values())),
                            np.mean(list(nx.closeness_centrality(service_graph).values())),
                            np.mean(list(nx.betweenness_centrality(service_graph).values()))
                        ]
                        advanced_features.extend(centrality_features)
                except:
                    advanced_features.extend([0.0, 0.0, 0.0])
            else:
                advanced_features.extend([0.0, 0.0, 0.0])
            
            # 高級調用模式特徵
            call_patterns = self._extract_call_patterns(trace_data)
            advanced_features.extend(call_patterns)
            
            # 高級延遲特徵
            latency_features = self._extract_latency_features(trace_data)
            advanced_features.extend(latency_features)
            
            # 高級依賴關係特徵
            dependency_features = self._extract_dependency_features(trace_data)
            advanced_features.extend(dependency_features)
            
            # 時間模式特徵
            if 'startTime' in trace_data.columns and inject_time is not None:
                try:
                    timestamps = pd.to_datetime(trace_data['startTime'])
                    
                    # 調用頻率變化
                    pre_inject = timestamps[timestamps < inject_time]
                    post_inject = timestamps[timestamps >= inject_time]
                    
                    if len(pre_inject) > 0 and len(post_inject) > 0:
                        pre_rate = len(pre_inject) / max((inject_time - timestamps.min()).total_seconds(), 1)
                        post_rate = len(post_inject) / max((timestamps.max() - inject_time).total_seconds(), 1)
                        rate_change = (post_rate - pre_rate) / max(pre_rate, 1e-8)
                        advanced_features.append(rate_change)
                    else:
                        advanced_features.append(0.0)
                except:
                    advanced_features.append(0.0)
            else:
                advanced_features.append(0.0)
            
            # 錯誤傳播特徵
            if 'statusCode' in trace_data.columns or 'status' in trace_data.columns:
                status_col = 'statusCode' if 'statusCode' in trace_data.columns else 'status'
                try:
                    error_count = len(trace_data[trace_data[status_col].astype(str).str.contains('error|fail|4[0-9][0-9]|5[0-9][0-9]', case=False, na=False)])
                    error_rate = error_count / max(len(trace_data), 1)
                    advanced_features.append(error_rate)
                except:
                    advanced_features.append(0.0)
            else:
                advanced_features.append(0.0)
            
            return np.array(advanced_features).reshape(1, -1) if advanced_features else np.array([])
            
        except Exception as e:
            print(f"⚠️ Advanced trace feature extraction failed: {e}")
            return np.array([])
    
    def _extract_call_patterns(self, trace_features):
        """提取調用模式特徵"""
        try:
            if isinstance(trace_features, pd.DataFrame):
                if trace_features.empty:
                    return [0.0, 0.0, 0.0]
                # 轉換為數值矩陣
                numeric_cols = trace_features.select_dtypes(include=[np.number])
                if numeric_cols.empty:
                    return [0.0, 0.0, 0.0]
                trace_array = numeric_cols.values
            else:
                trace_array = np.array(trace_features)
            
            if trace_array.ndim == 1:
                trace_array = trace_array.reshape(-1, 1)
            
            if trace_array.shape[0] < 2:
                return [0.0, 0.0, 0.0]
            
            # 調用頻率模式
            call_counts = np.sum(trace_array > 0, axis=0)
            call_frequency = np.mean(call_counts)
            call_variance = np.var(call_counts)
            
            # 調用時序模式
            temporal_pattern = np.mean(np.diff(trace_array, axis=0))
            
            return [call_frequency, call_variance, temporal_pattern]
        except Exception as e:
            print(f"⚠️ Call pattern extraction failed: {e}")
            return [0.0, 0.0, 0.0]
    
    def _extract_latency_features(self, trace_features):
        """提取延遲特徵"""
        try:
            if isinstance(trace_features, pd.DataFrame):
                # 尋找延遲相關的列
                latency_cols = [col for col in trace_features.columns 
                              if any(keyword in col.lower() for keyword in ['latency', 'duration', 'time', 'delay'])]
                
                if latency_cols:
                    latency_col = trace_features[latency_cols[0]].dropna()
                elif trace_features.shape[1] > 0:
                    # 使用最後一列作為延遲信息
                    latency_col = trace_features.iloc[:, -1].dropna()
                else:
                    return [0.0, 0.0, 0.0, 0.0]
            else:
                trace_array = np.array(trace_features)
                if trace_array.ndim == 2 and trace_array.shape[1] > 0:
                    latency_col = trace_array[:, -1]
                else:
                    latency_col = trace_array.flatten()
            
            if len(latency_col) == 0:
                return [0.0, 0.0, 0.0, 0.0]
            
            # 延遲統計特徵
            latency_stats = [
                np.mean(latency_col),
                np.std(latency_col),
                np.percentile(latency_col, 95),
                np.percentile(latency_col, 99)
            ]
            
            return latency_stats
        except Exception as e:
            print(f"⚠️ Latency feature extraction failed: {e}")
            return [0.0, 0.0, 0.0, 0.0]
    
    def _extract_dependency_features(self, trace_features):
        """提取依賴關係特徵"""
        try:
            if isinstance(trace_features, pd.DataFrame):
                numeric_cols = trace_features.select_dtypes(include=[np.number])
                if numeric_cols.shape[1] < 2:
                    return [0.0, 0.0, 0.0, 0.0]
                trace_array = numeric_cols.values
            else:
                trace_array = np.array(trace_features)
                if trace_array.ndim == 1:
                    return [0.0, 0.0, 0.0, 0.0]
            
            if trace_array.shape[1] < 2:
                return [0.0, 0.0, 0.0, 0.0]
            
            # 計算列間相關性作為依賴關係指標
            corr_matrix = np.corrcoef(trace_array.T)
            
            # 提取相關性統計特徵
            upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            
            # 處理NaN值
            upper_triangle = upper_triangle[~np.isnan(upper_triangle)]
            if len(upper_triangle) == 0:
                return [0.0, 0.0, 0.0, 0.0]
            
            dependency_stats = [
                np.mean(upper_triangle),
                np.std(upper_triangle),
                np.max(upper_triangle),
                len(upper_triangle[upper_triangle > 0.5])  # 強相關數量
            ]
            
            return dependency_stats
        except Exception as e:
            print(f"⚠️ Dependency feature extraction failed: {e}")
            return [0.0, 0.0, 0.0, 0.0]
    
    def _semantic_log_processing(self, log_features):
        """語義日誌處理"""
        try:
            if isinstance(log_features, np.ndarray):
                # 錯誤密度特徵
                error_density = self._compute_error_density(log_features)
                
                # 日誌模式特徵
                pattern_features = self._extract_log_patterns(log_features)
                
                # 嚴重性特徵
                severity_features = self._extract_severity_features(log_features)
                
                # 組合日誌特徵
                combined_features = np.concatenate([
                    error_density, pattern_features, severity_features
                ])
                
                feature_names = (
                    [f'error_density_{i}' for i in range(len(error_density))] +
                    [f'log_pattern_{i}' for i in range(len(pattern_features))] +
                    [f'severity_{i}' for i in range(len(severity_features))]
                )
                
                return combined_features.reshape(1, -1), feature_names
            else:
                return np.array([[0]]), ['log_default']
                
        except Exception as e:
            print(f"⚠️ Semantic log processing failed: {e}")
            return np.array([[0]]), ['log_error']
    
    def _compute_error_density(self, log_features):
        """計算錯誤密度"""
        if log_features.size == 0:
            return np.array([0.0, 0.0])
        
        # 假設高值表示錯誤
        error_threshold = np.percentile(log_features.flatten(), 90)
        error_count = np.sum(log_features > error_threshold)
        error_density = error_count / log_features.size
        
        # 錯誤聚集度
        if log_features.ndim == 2:
            error_clustering = np.std(np.sum(log_features > error_threshold, axis=1))
        else:
            error_clustering = 0.0
        
        return np.array([error_density, error_clustering])
    
    def _extract_log_patterns(self, log_features):
        """提取日誌模式"""
        if log_features.size == 0:
            return np.array([0.0, 0.0, 0.0])
        
        # 重複模式檢測
        if log_features.ndim == 2 and log_features.shape[0] > 1:
            # 計算行間相似性
            similarities = []
            for i in range(log_features.shape[0] - 1):
                sim = np.corrcoef(log_features[i], log_features[i + 1])[0, 1]
                if not np.isnan(sim):
                    similarities.append(sim)
            
            if similarities:
                pattern_consistency = np.mean(similarities)
                pattern_variance = np.var(similarities)
            else:
                pattern_consistency = 0.0
                pattern_variance = 0.0
        else:
            pattern_consistency = 0.0
            pattern_variance = 0.0
        
        # 模式複雜度
        pattern_complexity = np.std(log_features.flatten())
        
        return np.array([pattern_consistency, pattern_variance, pattern_complexity])
    
    def _extract_severity_features(self, log_features):
        """提取嚴重性特徵"""
        if log_features.size == 0:
            return np.array([0.0, 0.0, 0.0])
        
        flattened = log_features.flatten()
        
        # 嚴重性分佈
        severity_levels = [
            np.sum(flattened > np.percentile(flattened, 99)),  # 極高嚴重性
            np.sum(flattened > np.percentile(flattened, 95)),  # 高嚴重性
            np.sum(flattened > np.percentile(flattened, 75))   # 中等嚴重性
        ]
        
        return np.array(severity_levels) / len(flattened)
    
    def _generic_feature_processing(self, features, modality):
        """通用特徵處理"""
        try:
            if isinstance(features, np.ndarray) and features.size > 0:
                # 基本統計特徵
                stats_features = [
                    np.mean(features),
                    np.std(features),
                    np.min(features),
                    np.max(features),
                    np.median(features)
                ]
                
                # 分佈特徵
                if features.size > 10:
                    hist, _ = np.histogram(features.flatten(), bins=10)
                    hist_features = hist / np.sum(hist)
                else:
                    hist_features = np.zeros(10)
                
                # 組合特徵
                combined_features = np.concatenate([stats_features, hist_features])
                
                feature_names = (
                    ['mean', 'std', 'min', 'max', 'median'] +
                    [f'hist_{i}' for i in range(10)]
                )
                
                return combined_features.reshape(1, -1), feature_names
            else:
                return np.array([[0]]), [f'{modality}_default']
                
        except Exception as e:
            print(f"⚠️ Generic processing failed for {modality}: {e}")
            return np.array([[0]]), [f'{modality}_error']


class RobustFeatureNormalizer:
    """魯棒特徵正規化器 - 處理異常值和缺失值"""
    
    def __init__(self, method='robust', clip_percentiles=(1, 99)):
        self.method = method
        self.clip_percentiles = clip_percentiles
        self.fitted_params = {}
        
    def fit_transform(self, features):
        """擬合並轉換特徵"""
        if features.size == 0:
            return features
        
        try:
            if self.method == 'robust':
                return self._robust_normalize(features)
            elif self.method == 'quantile':
                return self._quantile_normalize(features)
            else:
                return self._standard_normalize(features)
                
        except Exception as e:
            print(f"⚠️ Feature normalization failed: {e}")
            return features
    
    def _robust_normalize(self, features):
        """魯棒正規化 - 使用中位數和MAD"""
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        normalized = features.copy()
        
        for i in range(features.shape[1]):
            col = features[:, i]
            
            # 計算中位數和中位絕對偏差
            median = np.median(col)
            mad = np.median(np.abs(col - median))
            
            if mad > 0:
                normalized[:, i] = (col - median) / (1.4826 * mad)  # 1.4826 是正規化常數
            else:
                normalized[:, i] = col - median
        
        return normalized
    
    def _quantile_normalize(self, features):
        """分位數正規化"""
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        normalized = features.copy()
        
        for i in range(features.shape[1]):
            col = features[:, i]
            
            # 計算分位數
            q1, q99 = np.percentile(col, self.clip_percentiles)
            
            # 裁剪異常值
            clipped = np.clip(col, q1, q99)
            
            # 正規化到 [0, 1]
            if q99 > q1:
                normalized[:, i] = (clipped - q1) / (q99 - q1)
            else:
                normalized[:, i] = np.zeros_like(clipped)
        
        return normalized
    
    def _standard_normalize(self, features):
        """標準正規化"""
        scaler = StandardScaler()
        return scaler.fit_transform(features)


def psm_metric_processing(metrics_data, target_dim=64):
    """
    PSM（相空間方法）指標處理 - 替代簡化的指標處理
    專注於捕捉複雜系統動態特徵
    
    Args:
        metrics_data: 指標數據
        target_dim: 目標維度
    
    Returns:
        processed_features: 處理後的特徵
        feature_names: 特徵名稱
    """
    print("🔧 Using PSM (Phase Space Method) for metric processing...")
    
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
        
        if len(series) < 5:
            # 數據太少，使用基本統計
            basic_stats = [series.mean() if len(series) > 0 else 0, 
                          series.std() if len(series) > 1 else 0, 
                          series.min() if len(series) > 0 else 0, 
                          series.max() if len(series) > 0 else 0]
            all_features.extend(basic_stats)
            feature_names.extend([f'{col_name}_mean', f'{col_name}_std', f'{col_name}_min', f'{col_name}_max'])
            continue
        
        # 🎯 PSM核心特徵：相空間重構
        # 1. 時間延遲嵌入（相空間重構的關鍵）
        embedded_features = _phase_space_embedding(series.values)
        
        # 2. 動力學不變量
        dynamics_features = _compute_dynamics_invariants(series.values)
        
        # 3. 穩定性指標
        stability_features = _compute_stability_metrics(series.values)
        
        # 4. 頻域特徵
        frequency_features = _compute_frequency_features(series.values)
        
        # 組合PSM特徵
        col_features = embedded_features + dynamics_features + stability_features + frequency_features
        all_features.extend(col_features)
        
        # 生成特徵名稱
        names = [
            f'{col_name}_embed_dim1', f'{col_name}_embed_dim2', f'{col_name}_embed_corr',
            f'{col_name}_lyapunov', f'{col_name}_entropy', f'{col_name}_complexity',
            f'{col_name}_variance_ratio', f'{col_name}_stability_index', f'{col_name}_prediction_error',
            f'{col_name}_dominant_freq', f'{col_name}_spectral_entropy', f'{col_name}_freq_stability'
        ]
        feature_names.extend(names)
    
    # 轉換為矩陣格式
    if all_features:
        feature_matrix = np.array(all_features).reshape(1, -1)
        
        # PCA降維到目標維度 - 保留重要信息
        if feature_matrix.shape[1] > target_dim:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=target_dim, random_state=42)
            feature_matrix = pca.fit_transform(feature_matrix)
            # 基於解釋方差重新命名特徵
            explained_var = pca.explained_variance_ratio_
            feature_names = [f'psm_pc{i}_var{explained_var[i]:.3f}' for i in range(target_dim)]
    else:
        feature_matrix = np.array([[0]])
        feature_names = ['default_feature']
    
    print(f"✓ PSM processing: {feature_matrix.shape[1]} features extracted")
    return feature_matrix, feature_names


def _phase_space_embedding(series, embedding_dim=3, delay=1):
    """時間延遲嵌入進行相空間重構"""
    if len(series) < embedding_dim * delay:
        return [0.0, 0.0, 0.0]
    
    # 構建延遲向量
    embedded_matrix = []
    for i in range(len(series) - (embedding_dim - 1) * delay):
        vector = [series[i + j * delay] for j in range(embedding_dim)]
        embedded_matrix.append(vector)
    
    embedded_matrix = np.array(embedded_matrix)
    
    # 相空間特徵
    features = []
    if embedded_matrix.size > 0:
        # 主要方向的方差
        features.append(np.var(embedded_matrix[:, 0]))
        features.append(np.var(embedded_matrix[:, 1]) if embedded_matrix.shape[1] > 1 else 0)
        # 維度間相關性
        if embedded_matrix.shape[1] > 1:
            corr = np.corrcoef(embedded_matrix[:, 0], embedded_matrix[:, 1])[0, 1]
            features.append(corr if not np.isnan(corr) else 0)
        else:
            features.append(0)
    else:
        features = [0.0, 0.0, 0.0]
    
    return features


def _compute_dynamics_invariants(series):
    """計算動力學不變量"""
    if len(series) < 10:
        return [0.0, 0.0, 0.0]
    
    # 近似Lyapunov指數
    diffs = np.diff(series)
    lyapunov_approx = np.mean(np.log(np.abs(diffs) + 1e-10))
    
    # 樣本熵
    sample_entropy = _compute_sample_entropy(series)
    
    # 複雜度度量
    complexity = np.std(diffs) / (np.mean(np.abs(series)) + 1e-10)
    
    return [lyapunov_approx, sample_entropy, complexity]


def _compute_stability_metrics(series):
    """計算穩定性指標"""
    if len(series) < 5:
        return [0.0, 0.0, 0.0]
    
    # 方差比率（短期vs長期）
    mid = len(series) // 2
    var_ratio = np.var(series[:mid]) / (np.var(series[mid:]) + 1e-10)
    
    # 穩定性指數
    mean_abs_dev = np.mean(np.abs(series - np.mean(series)))
    stability_index = 1.0 / (1.0 + mean_abs_dev)
    
    # 預測誤差（簡單線性預測）
    if len(series) > 10:
        x = np.arange(len(series))
        trend = np.polyfit(x, series, 1)
        predicted = np.polyval(trend, x)
        prediction_error = np.mean((series - predicted) ** 2)
    else:
        prediction_error = 0.0
    
    return [var_ratio, stability_index, prediction_error]


def _compute_frequency_features(series):
    """計算頻域特徵"""
    if len(series) < 8:
        return [0.0, 0.0, 0.0]
    
    # FFT
    fft = np.fft.fft(series)
    power_spectrum = np.abs(fft) ** 2
    
    # 主導頻率
    dominant_freq = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
    
    # 譜熵
    power_norm = power_spectrum / (np.sum(power_spectrum) + 1e-10)
    spectral_entropy = -np.sum(power_norm * np.log(power_norm + 1e-10))
    
    # 頻率穩定性
    freq_stability = 1.0 / (1.0 + np.std(power_spectrum))
    
    return [float(dominant_freq), spectral_entropy, freq_stability]


def _compute_sample_entropy(series, m=2, r=None):
    """計算樣本熵"""
    if r is None:
        r = 0.2 * np.std(series)
    
    def _maxdist(xi, xj, m):
        return max([abs(ua - va) for ua, va in zip(xi, xj)])
    
    def _phi(m):
        N = len(series)
        patterns = np.array([series[i:i + m] for i in range(N - m + 1)])
        C = np.zeros(N - m + 1)
        
        for i in range(N - m + 1):
            template_i = patterns[i]
            for j in range(N - m + 1):
                if _maxdist(template_i, patterns[j], m) <= r:
                    C[i] += 1.0
        
        phi = (N - m + 1.0) ** (-1) * np.sum(np.log(C / (N - m + 1.0)))
        return phi
    
    try:
        return _phi(m) - _phi(m + 1)
    except:
        return 0.0


def create_advanced_processor(config):
    """創建高級處理器實例"""
    return AdvancedFeatureProcessor(config)


class DynamicModelAdjuster:
    """動態模型調整器 - 根據訓練過程動態調整模型參數"""
    
    def __init__(self, adjustment_frequency=50):
        self.adjustment_frequency = adjustment_frequency
        self.loss_history = []
        self.adjustment_count = 0
        
    def should_adjust(self, epoch, current_loss):
        """判斷是否需要調整模型"""
        self.loss_history.append(current_loss)
        
        if epoch % self.adjustment_frequency == 0 and epoch > 0:
            if len(self.loss_history) >= self.adjustment_frequency:
                recent_losses = self.loss_history[-self.adjustment_frequency:]
                loss_variance = np.var(recent_losses)
                
                # 如果損失變化很小，可能需要調整
                if loss_variance < 1e-6:
                    return True
                    
                # 如果損失持續增加，也需要調整
                if len(recent_losses) >= 10:
                    trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
                    if trend > 0:  # 上升趨勢
                        return True
        
        return False
    
    def adjust_model_parameters(self, model, optimizer):
        """調整模型參數"""
        print(f"🔧 Performing model adjustment #{self.adjustment_count + 1}")
        
        # 1. 調整學習率
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            param_group['lr'] *= 0.8  # 降低學習率
            print(f"   📉 Learning rate: {old_lr:.6f} -> {param_group['lr']:.6f}")
        
        # 2. 添加權重噪聲（幫助跳出局部最優）
        with torch.no_grad():
            for param in model.parameters():
                if param.dim() > 1:  # 只對權重矩陣添加噪聲
                    noise = torch.randn_like(param) * 0.01
                    param.add_(noise)
        
        self.adjustment_count += 1
        print(f"   ✅ Model adjustment completed")
    
    def get_adjustment_stats(self):
        """獲取調整統計信息"""
        return {
            'total_adjustments': self.adjustment_count,
            'loss_history_length': len(self.loss_history),
            'recent_loss_variance': np.var(self.loss_history[-50:]) if len(self.loss_history) >= 50 else 0
        }

# 添加缺失的函數
def _extract_time_series_features(self, metric_data, inject_time):
    """提取時間序列特徵"""
    try:
        if isinstance(metric_data, pd.DataFrame):
            if 'timestamp' in metric_data.columns:
                # 按時間排序
                metric_data = metric_data.sort_values('timestamp')
                
                # 提取數值列
                numeric_cols = metric_data.select_dtypes(include=[np.number])
                
                if numeric_cols.empty:
                    return np.array([])
                
                # 時間序列統計特徵
                ts_features = []
                
                for col in numeric_cols.columns:
                    series = numeric_cols[col].dropna()
                    
                    if len(series) > 5:
                        # 趨勢特徵
                        trend = np.polyfit(range(len(series)), series, 1)[0]
                        ts_features.append(trend)
                        
                        # 季節性特徵（簡化）
                        if len(series) > 12:
                            seasonal_strength = np.std(series[:12]) / (np.mean(series[:12]) + 1e-8)
                        else:
                            seasonal_strength = 0.0
                        ts_features.append(seasonal_strength)
                        
                        # 自相關特徵
                        if len(series) > 2:
                            autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
                            ts_features.append(autocorr if not np.isnan(autocorr) else 0)
                        else:
                            ts_features.append(0.0)
                
                return np.array(ts_features).reshape(1, -1) if ts_features else np.array([])
            else:
                return np.array([])
        else:
            return np.array([])
            
    except Exception as e:
        print(f"⚠️ Time series feature extraction failed: {e}")
        return np.array([])

def _extract_log_semantic_features(self, log_data):
    """提取日誌語義特徵"""
    try:
        if isinstance(log_data, pd.DataFrame):
            # 關鍵詞檢測
            error_keywords = ['error', 'exception', 'fail', 'timeout', 'crash']
            warning_keywords = ['warning', 'warn', 'deprecated', 'slow']
            
            semantic_features = []
            
            # 錯誤關鍵詞密度
            if 'message' in log_data.columns or 'content' in log_data.columns:
                text_col = 'message' if 'message' in log_data.columns else 'content'
                text_data = log_data[text_col].astype(str).str.lower()
                
                error_density = sum(text_data.str.contains('|'.join(error_keywords), na=False)) / len(text_data)
                warning_density = sum(text_data.str.contains('|'.join(warning_keywords), na=False)) / len(text_data)
                
                semantic_features.extend([error_density, warning_density])
            else:
                semantic_features.extend([0.0, 0.0])
            
            # 日誌級別分佈
            if 'level' in log_data.columns:
                level_counts = log_data['level'].value_counts(normalize=True)
                error_ratio = level_counts.get('ERROR', 0) + level_counts.get('error', 0)
                warning_ratio = level_counts.get('WARNING', 0) + level_counts.get('warning', 0)
                semantic_features.extend([error_ratio, warning_ratio])
            else:
                semantic_features.extend([0.0, 0.0])
            
            return np.array(semantic_features).reshape(1, -1) if semantic_features else np.array([])
        else:
            return np.array([])
            
    except Exception as e:
        print(f"⚠️ Log semantic feature extraction failed: {e}")
        return np.array([])

def _align_modal_features(self, processed_modals):
    """對齊不同模態的特徵"""
    try:
        # 找到最大的節點數量
        max_nodes = max(features.size(0) for features in processed_modals.values())
        
        aligned_modals = {}
        for modal_name, features in processed_modals.items():
            if features.size(0) < max_nodes:
                # 填充到相同大小
                padding_size = max_nodes - features.size(0)
                padding = torch.zeros(padding_size, features.size(1), dtype=features.dtype, device=features.device)
                aligned_features = torch.cat([features, padding], dim=0)
            else:
                aligned_features = features[:max_nodes]
            
            aligned_modals[modal_name] = aligned_features
        
        return aligned_modals
        
    except Exception as e:
        print(f"⚠️ Modal feature alignment failed: {e}")
        return processed_modals

def _advanced_feature_fusion(self, aligned_features):
    """高級特徵融合"""
    try:
        if not aligned_features:
            return torch.tensor([[0.0]])
        
        # 簡單拼接融合
        feature_list = list(aligned_features.values())
        
        if len(feature_list) == 1:
            return feature_list[0]
        
        # 確保所有特徵有相同的第一維
        min_dim0 = min(f.size(0) for f in feature_list)
        aligned_list = [f[:min_dim0] for f in feature_list]
        
        # 拼接特徵
        fused_features = torch.cat(aligned_list, dim=1)
        
        return fused_features
        
    except Exception as e:
        print(f"⚠️ Advanced feature fusion failed: {e}")
        # 返回第一個可用特徵
        if aligned_features:
            return list(aligned_features.values())[0]
        else:
            return torch.tensor([[0.0]])

# 將方法添加到 AdvancedFeatureProcessor 類中
AdvancedFeatureProcessor._extract_time_series_features = _extract_time_series_features
AdvancedFeatureProcessor._extract_log_semantic_features = _extract_log_semantic_features
AdvancedFeatureProcessor._align_modal_features = _align_modal_features
AdvancedFeatureProcessor._advanced_feature_fusion = _advanced_feature_fusion

# 確保所有需要的類都被導出
__all__ = [
    'EnhancedMultiModalAttention',
    'AdvancedFeatureProcessor', 
    'RobustFeatureNormalizer',
    'DynamicModelAdjuster',
    'psm_metric_processing',
    'create_advanced_processor'
]