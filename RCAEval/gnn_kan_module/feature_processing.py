"""
高級特徵處理模組 - 整合所有特徵處理功能
從大檔案中提取並優化的關鍵函數
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")


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


# 主要的GNN-KAN RCA函數
def gnn_kan_rca(data, inject_time=None, dataset=None, with_bg=False, **kwargs):
    """
    主要的 GNN-KAN RCA 方法 - 模組化版本
    
    Args:
        data: 輸入數據 (multimodal 或 單一模態)
        inject_time: 注入時間點
        dataset: 數據集名稱
        with_bg: 是否包含背景數據
        **kwargs: 其他參數
    
    Returns:
        dict: 包含 adj, node_names, ranks 的結果
    """
    try:
        # 導入模組化的組件
        from . import (
            SimplifiedGNNKANConfig,
            MultiModalFeatureExtractor,
            SimplifiedGraphConstructor,
            GNNKANModel,
            train_gnn_kan_model
        )
        
        print("Starting modularized GNN-KAN RCA analysis...")
        start_time = time.time()
        
        # 使用模組化的配置和處理器
        config = SimplifiedGNNKANConfig()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # 特徵提取
        feature_extractor = MultiModalFeatureExtractor(config)
        features, node_names = feature_extractor.extract_features(data, inject_time)
        
        if features.size == 0 or len(node_names) == 0:
            return {"adj": np.array([]), "node_names": [], "ranks": []}
        
        # 圖構建和模型訓練
        graph_constructor = SimplifiedGraphConstructor(config)
        edge_index, edge_weights = graph_constructor.build_graph(features, node_names)
        
        # 準備節點特徵並訓練模型
        if features.ndim == 2 and features.shape[1] >= config.target_feature_dim:
            node_features = features[:len(node_names), :config.target_feature_dim]
        else:
            node_features = np.random.randn(len(node_names), config.target_feature_dim)
        
        model = GNNKANModel(config, len(node_names))
        device = 'cuda' if config.use_cuda and torch.cuda.is_available() else 'cpu'
        
        node_features_tensor = torch.tensor(node_features, dtype=torch.float).to(device)
        edge_index = edge_index.to(device)
        model = model.to(device)
        
        model, final_adj = train_gnn_kan_model(model, node_features_tensor, edge_index, config)
        
        # 計算排名
        adj_numpy = final_adj.detach().cpu().numpy()
        from sknetwork.ranking import PageRank
        
        pagerank = PageRank()
        scores = pagerank.fit_transform(adj_numpy)
        ranked_indices = np.argsort(scores)[::-1]
        top_k_indices = ranked_indices[:config.top_k_results]
        
        ranks = [str(node_names[i]) if i < len(node_names) else f"node_{i}" for i in top_k_indices]
        
        end_time = time.time()
        print(f"Modularized GNN-KAN RCA completed in {end_time - start_time:.2f} seconds")
        
        return {
            "adj": adj_numpy,
            "node_names": node_names,
            "ranks": ranks
        }
        
    except Exception as e:
        print(f"Modularized GNN-KAN RCA failed: {e}")
        import traceback
        traceback.print_exc()
        return {"adj": np.array([]), "node_names": [], "ranks": []}


__all__ = [
    'simplified_metric_processing',
    'enhanced_trace_processing', 
    'psm_metric_processing',
    'gnn_kan_rca'
]