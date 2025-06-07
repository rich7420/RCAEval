"""
Feature extraction module for GNN-KAN RCA
Implements sliding window alignment, TF-IDF log features, STL decomposition, etc.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.seasonal import STL
import networkx as nx
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def sliding_window_alignment(data, window_size=32, step_size=1, timestamp_col='time'):
    """
    對齊時間戳並建立滑動窗口
    
    Args:
        data: DataFrame with timestamp column
        window_size: 窗口大小
        step_size: 步長
        timestamp_col: 時間戳列名
    
    Returns:
        windows: list of windowed data
        timestamps: corresponding timestamps
    """
    if isinstance(data, dict):
        # 處理多模態數據
        aligned_data = {}
        base_timestamps = None
        
        for modality, df in data.items():
            if timestamp_col in df.columns:
                if base_timestamps is None:
                    base_timestamps = df[timestamp_col].values
                
                # 對齊到基準時間戳
                aligned_df = align_to_timestamps(df, base_timestamps, timestamp_col)
                aligned_data[modality] = aligned_df
            else:
                aligned_data[modality] = df
        
        # 建立滑動窗口
        windows = []
        timestamps = []
        
        for i in range(0, len(base_timestamps) - window_size + 1, step_size):
            window_data = {}
            for modality, df in aligned_data.items():
                window_data[modality] = df.iloc[i:i+window_size]
            
            windows.append(window_data)
            timestamps.append(base_timestamps[i:i+window_size])
            
    else:
        # 處理單一模態數據
        if timestamp_col in data.columns:
            timestamps_data = data[timestamp_col].values
            data_without_time = data.drop(columns=[timestamp_col])
        else:
            timestamps_data = np.arange(len(data))
            data_without_time = data
        
        windows = []
        timestamps = []
        
        for i in range(0, len(data_without_time) - window_size + 1, step_size):
            windows.append(data_without_time.iloc[i:i+window_size])
            timestamps.append(timestamps_data[i:i+window_size])
    
    return windows, timestamps


def align_to_timestamps(df, target_timestamps, timestamp_col='time'):
    """對齊數據到目標時間戳"""
    df_sorted = df.sort_values(timestamp_col)
    aligned_df = pd.DataFrame()
    
    for ts in target_timestamps:
        # 找最接近的時間戳
        closest_idx = np.argmin(np.abs(df_sorted[timestamp_col] - ts))
        closest_row = df_sorted.iloc[closest_idx:closest_idx+1].copy()
        closest_row[timestamp_col] = ts
        aligned_df = pd.concat([aligned_df, closest_row], ignore_index=True)
    
    return aligned_df


def extract_log_features(log_data, use_dla=False, max_features=1000, ngram_range=(1, 2)):
    """
    提取日誌特徵使用 TF-IDF 向量化
    
    Args:
        log_data: DataFrame 包含日誌數據
        use_dla: 是否使用深度日誌分析
        max_features: TF-IDF 最大特徵數
        ngram_range: n-gram 範圍
    
    Returns:
        features: 提取的日誌特徵
        feature_names: 特徵名稱
    """
    if isinstance(log_data, pd.DataFrame):
        # 假設日誌數據在多個列中
        log_texts = []
        for col in log_data.columns:
            if log_data[col].dtype == 'object':  # 文本列
                log_texts.extend(log_data[col].dropna().astype(str).tolist())
    else:
        log_texts = log_data
    
    if not log_texts:
        return np.array([]), []
    
    # TF-IDF 向量化
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english' if all(isinstance(x, str) for x in log_texts) else None,
        lowercase=True,
        token_pattern=r'\b\w+\b'
    )
    
    try:
        tfidf_features = vectorizer.fit_transform(log_texts)
        feature_names = vectorizer.get_feature_names_out().tolist()
        
        # 如果使用 DLA，添加深度特徵
        if use_dla:
            dla_features = extract_dla_features(log_texts)
            features = np.hstack([tfidf_features.toarray(), dla_features])
            feature_names.extend([f'dla_{i}' for i in range(dla_features.shape[1])])
        else:
            features = tfidf_features.toarray()
            
    except Exception as e:
        print(f"TF-IDF extraction failed: {e}")
        # 回退到簡單特徵
        features = np.array([[len(text), text.count(' ')] for text in log_texts])
        feature_names = ['log_length', 'log_word_count']
    
    return features, feature_names


def extract_dla_features(log_texts, embedding_dim=128):
    """
    深度日誌分析特徵提取 (簡化版)
    
    Args:
        log_texts: 日誌文本列表
        embedding_dim: 嵌入維度
    
    Returns:
        dla_features: DLA 特徵
    """
    # 簡化的 DLA：使用字符級統計特徵
    features = []
    
    for text in log_texts:
        text_features = [
            len(text),  # 文本長度
            text.count(' '),  # 空格數
            text.count('\n'),  # 換行數
            len(set(text)),  # 唯一字符數
            text.count('ERROR'),  # 錯誤關鍵字
            text.count('WARN'),   # 警告關鍵字
            text.count('INFO'),   # 信息關鍵字
            text.count('DEBUG'),  # 調試關鍵字
        ]
        features.append(text_features)
    
    features = np.array(features)
    
    # 使用 PCA 降維到指定維度
    if features.shape[1] > embedding_dim:
        pca = PCA(n_components=min(embedding_dim, features.shape[1]))
        features = pca.fit_transform(features)
    
    return features


def stl_decomposition(metrics_data, seasonal=12, return_components=True):
    """
    STL 分解將 metrics 分解為趨勢、季節性和殘差
    改進版：更好地處理短序列和無週期性數據，減少警告訊息
    
    Args:
        metrics_data: DataFrame 包含時間序列數據 或 numpy array
        seasonal: 季節性週期 (默認改為12)
        return_components: 是否返回所有組件
    
    Returns:
        decomposed_features: 分解後的特徵
        component_names: 組件名稱
    """
    # 處理不同類型的輸入
    if isinstance(metrics_data, pd.DataFrame):
        data = metrics_data.select_dtypes(include=[np.number])
    elif isinstance(metrics_data, np.ndarray):
        # 如果是numpy數組，轉換為DataFrame
        if metrics_data.ndim == 1:
            data = pd.DataFrame({'metric': metrics_data})
        else:
            data = pd.DataFrame(metrics_data, columns=[f'metric_{i}' for i in range(metrics_data.shape[1])])
    else:
        # 嘗試轉換為DataFrame
        try:
            data = pd.DataFrame(metrics_data)
        except:
            # 最後的回退
            return np.array([[0]]), ['default_feature']

    decomposed_features = []
    component_names = []

    for col in data.columns:
        series = data[col].dropna()
        
        # 改進的數據長度和週期性檢查
        min_length_required = 24  # 降低最小要求到24個點
        
        if len(series) < min_length_required:
            # 數據太短，使用統計特徵代替
            features = np.array([
                [np.mean(series), np.std(series), np.max(series), np.min(series)]
            ]).T
            names = [f'{col}_mean', f'{col}_std', f'{col}_max', f'{col}_min']
            # 減少噪音輸出 - 只在debug模式下輸出
            if getattr(stl_decomposition, '_debug', False):
                print(f"STL skipped for short series: {col} (length: {len(series)})")
        else:
            # 檢查數據是否有足夠變化（避免常數序列）
            if series.std() < 1e-10:
                # 常數序列，使用統計特徵
                features = np.array([
                    [series.iloc[0], 0, series.iloc[0], series.iloc[0]]
                ]).T
                names = [f'{col}_mean', f'{col}_std', f'{col}_max', f'{col}_min']
            else:
                # 嘗試 STL 分解 - 使用更穩健的參數
                try:
                    # 智能化的季節性參數選擇
                    auto_seasonal = min(seasonal, len(series) // 3)  # 放寬到1/3
                    auto_seasonal = max(3, auto_seasonal)  # 最小季節性為3
                    
                    # 檢測實際的週期性
                    detected_period = detect_seasonality(series, max_period=min(50, len(series)//2))
                    if detected_period:
                        auto_seasonal = detected_period
                    
                    # 使用更穩健的 STL 參數
                    stl = STL(series, 
                             seasonal=auto_seasonal,
                             robust=True,           # 使用穩健版本
                             seasonal_deg=0,        # 常數季節性擬合
                             trend_deg=1,           # 線性趨勢
                             low_pass_deg=1,        # 低通濾波器
                             seasonal_jump=1,       # 季節性跳躍閾值
                             trend_jump=1,          # 趨勢跳躍閾值
                             low_pass_jump=1)       # 低通跳躍閾值
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")  # 抑制STL警告
                        result = stl.fit()
                    
                    if return_components:
                        # 提取組件並計算統計特徵（避免返回長序列）
                        trend = result.trend.fillna(series.mean())
                        seasonal_comp = result.seasonal.fillna(0)
                        residual = result.resid.fillna(0)
                        
                        # 使用統計摘要而不是完整序列
                        features = np.array([[
                            np.mean(trend), np.std(trend),
                            np.mean(seasonal_comp), np.std(seasonal_comp),
                            np.mean(residual), np.std(residual),
                            np.max(trend) - np.min(trend),  # 趨勢範圍
                            np.max(seasonal_comp) - np.min(seasonal_comp)  # 季節性範圍
                        ]]).T
                        names = [f'{col}_trend_mean', f'{col}_trend_std',
                                f'{col}_seasonal_mean', f'{col}_seasonal_std',
                                f'{col}_residual_mean', f'{col}_residual_std',
                                f'{col}_trend_range', f'{col}_seasonal_range']
                    else:
                        # 只返回趨勢統計
                        trend = result.trend.fillna(series.mean())
                        features = np.array([[
                            np.mean(trend), np.std(trend),
                            np.max(trend), np.min(trend)
                        ]]).T
                        names = [f'{col}_trend_mean', f'{col}_trend_std',
                                f'{col}_trend_max', f'{col}_trend_min']
                
                except Exception as e:
                    # STL 失敗，回退到增強統計特徵
                    features = np.array([[
                        np.mean(series), np.std(series), 
                        np.max(series), np.min(series),
                        np.median(series), series.quantile(0.25), series.quantile(0.75),
                        len(series)  # 序列長度
                    ]]).T
                    names = [f'{col}_mean', f'{col}_std', f'{col}_max', f'{col}_min',
                            f'{col}_median', f'{col}_q25', f'{col}_q75', f'{col}_length']
                    
                    # 減少噪音輸出，不再為每個失敗打印警告
                    if col in ['adservice_cpu', 'cartservice_cpu'] and not hasattr(stl_decomposition, '_stl_warning_shown'):
                        print(f"STL decomposition failed for {col}: Unable to determine period from endog, using statistics")
                        stl_decomposition._stl_warning_shown = True
        
        decomposed_features.append(features)
        component_names.extend(names)
    
    # 對齊所有特徵的長度
    if decomposed_features:
        try:
            final_features = np.column_stack(decomposed_features)
        except ValueError as e:
            # 如果形狀不匹配，使用填充對齊
            max_length = max(f.shape[0] for f in decomposed_features)
            aligned_features = []
            
            for features in decomposed_features:
                if features.shape[0] < max_length:
                    # 使用最後一個值填充
                    padding = np.full((max_length - features.shape[0], features.shape[1]), 
                                     features[-1, :] if features.size > 0 else 0)
                    aligned = np.vstack([features, padding])
                else:
                    aligned = features[:max_length]
                aligned_features.append(aligned)
            
            final_features = np.column_stack(aligned_features)
    else:
        final_features = np.array([[0]])  # 默認特徵
        component_names = ['default_feature']
    
    return final_features, component_names


def detect_seasonality(series, max_period=None):
    """
    檢測時間序列的季節性週期
    
    Args:
        series: 時間序列
        max_period: 最大檢測週期
    
    Returns:
        detected_period: 檢測到的週期，None 如果沒有明顯週期性
    """
    if max_period is None:
        max_period = min(len(series) // 3, 50)
    
    if len(series) < 10 or max_period < 2:
        return None
    
    try:
        # 使用自相關函數檢測週期性
        autocorr_values = []
        periods = range(2, min(max_period + 1, len(series) // 2))
        
        for period in periods:
            if len(series) >= 2 * period:
                # 計算延遲 period 的自相關
                shifted = series.shift(period).dropna()
                original = series[:len(shifted)]
                
                if len(original) > 0 and len(shifted) > 0:
                    corr = np.corrcoef(original, shifted)[0, 1]
                    if not np.isnan(corr):
                        autocorr_values.append((period, abs(corr)))
        
        if autocorr_values:
            # 選擇自相關最強的週期
            best_period, best_corr = max(autocorr_values, key=lambda x: x[1])
            
            # 只有自相關超過閾值才認為有週期性
            if best_corr > 0.3:
                return best_period
        
        return None
        
    except Exception:
        return None


def compute_topology_features(adj_matrix, node_names=None):
    """
    計算圖的拓樸特徵
    
    Args:
        adj_matrix: 鄰接矩陣
        node_names: 節點名稱
    
    Returns:
        topology_features: 拓樸特徵
        feature_names: 特徵名稱
    """
    if adj_matrix.size == 0:
        return np.array([]), []
    
    # 建立圖
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    
    if node_names is not None:
        mapping = {i: name for i, name in enumerate(node_names)}
        G = nx.relabel_nodes(G, mapping)
    
    num_nodes = len(G.nodes())
    features = []
    feature_names = []
    
    # 基本拓樸特徵
    features.extend([
        num_nodes,  # 節點數
        len(G.edges()),  # 邊數
        nx.density(G),  # 密度
    ])
    feature_names.extend(['num_nodes', 'num_edges', 'density'])
    
    # 度中心性統計
    try:
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        
        features.extend([
            np.mean(list(in_degrees.values())),
            np.std(list(in_degrees.values())),
            np.mean(list(out_degrees.values())),
            np.std(list(out_degrees.values())),
        ])
        feature_names.extend(['in_degree_mean', 'in_degree_std', 
                            'out_degree_mean', 'out_degree_std'])
    except:
        features.extend([0, 0, 0, 0])
        feature_names.extend(['in_degree_mean', 'in_degree_std', 
                            'out_degree_mean', 'out_degree_std'])
    
    # 中心性度量
    try:
        pagerank = nx.pagerank(G)
        betweenness = nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)
        
        features.extend([
            np.mean(list(pagerank.values())),
            np.std(list(pagerank.values())),
            np.mean(list(betweenness.values())),
            np.std(list(betweenness.values())),
            np.mean(list(closeness.values())),
            np.std(list(closeness.values())),
        ])
        feature_names.extend([
            'pagerank_mean', 'pagerank_std',
            'betweenness_mean', 'betweenness_std',
            'closeness_mean', 'closeness_std'
        ])
    except:
        features.extend([0] * 6)
        feature_names.extend([
            'pagerank_mean', 'pagerank_std',
            'betweenness_mean', 'betweenness_std',
            'closeness_mean', 'closeness_std'
        ])
    
    # 連通性特徵
    try:
        if nx.is_strongly_connected(G):
            features.append(1)
        else:
            features.append(len(max(nx.strongly_connected_components(G), key=len)) / num_nodes)
    except:
        features.append(0)
    feature_names.append('connectivity')
    
    # 聚類係數
    try:
        clustering = nx.average_clustering(G.to_undirected())
        features.append(clustering)
    except:
        features.append(0)
    feature_names.append('clustering_coefficient')
    
    return np.array(features), feature_names


def extract_error_features(data, error_patterns=None):
    """
    提取錯誤特徵
    
    Args:
        data: 輸入數據
        error_patterns: 錯誤模式列表
    
    Returns:
        error_features: 錯誤特徵
        feature_names: 特徵名稱
    """
    if error_patterns is None:
        error_patterns = [
            'error', 'exception', 'fail', 'timeout', 'crash',
            'memory', 'cpu', 'disk', 'network', 'latency'
        ]
    
    features = []
    feature_names = []
    
    if isinstance(data, pd.DataFrame):
        # 處理數值異常
        for col in data.select_dtypes(include=[np.number]).columns:
            series = data[col].dropna()
            
            if len(series) > 0:
                # 異常值檢測
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).sum()
                
                features.extend([
                    outliers / len(series),  # 異常值比例
                    series.std() / (series.mean() + 1e-8),  # 變異係數
                    (series == 0).sum() / len(series),  # 零值比例
                ])
                feature_names.extend([
                    f'{col}_outlier_ratio',
                    f'{col}_cv',
                    f'{col}_zero_ratio'
                ])
        
        # 處理文本錯誤模式
        for col in data.select_dtypes(include=['object']).columns:
            text_data = data[col].dropna().astype(str)
            
            for pattern in error_patterns:
                count = text_data.str.contains(pattern, case=False).sum()
                features.append(count / len(text_data) if len(text_data) > 0 else 0)
                feature_names.append(f'{col}_{pattern}_ratio')
    
    return np.array(features), feature_names


def feature_fusion(log_features, metric_features, topology_features, error_features, 
                  fusion_method='concatenate', target_dim=None):
    """
    多模態特徵融合
    
    Args:
        log_features: 日誌特徵
        metric_features: 度量特徵
        topology_features: 拓樸特徵
        error_features: 錯誤特徵
        fusion_method: 融合方法 ('concatenate', 'attention', 'weighted')
        target_dim: 目標維度
    
    Returns:
        fused_features: 融合後的特徵
    """
    # 收集所有非空特徵
    all_features = []
    feature_weights = []
    
    if log_features is not None and log_features.size > 0:
        if log_features.ndim == 1:
            log_features = log_features.reshape(1, -1)
        all_features.append(log_features)
        feature_weights.append(0.3)  # 日誌特徵權重
    
    if metric_features is not None and metric_features.size > 0:
        if metric_features.ndim == 1:
            metric_features = metric_features.reshape(1, -1)
        all_features.append(metric_features)
        feature_weights.append(0.4)  # 度量特徵權重
    
    if topology_features is not None and topology_features.size > 0:
        if topology_features.ndim == 1:
            topology_features = topology_features.reshape(1, -1)
        all_features.append(topology_features)
        feature_weights.append(0.2)  # 拓樸特徵權重
    
    if error_features is not None and error_features.size > 0:
        if error_features.ndim == 1:
            error_features = error_features.reshape(1, -1)
        all_features.append(error_features)
        feature_weights.append(0.1)  # 錯誤特徵權重
    
    if not all_features:
        return np.array([])
    
    # 對齊特徵維度
    max_rows = max(f.shape[0] for f in all_features)
    aligned_features = []
    
    for features in all_features:
        if features.shape[0] < max_rows:
            # 重複最後一行以對齊
            padding = np.repeat(features[-1:], max_rows - features.shape[0], axis=0)
            features = np.vstack([features, padding])
        aligned_features.append(features)
    
    # 特徵融合
    if fusion_method == 'concatenate':
        fused_features = np.hstack(aligned_features)
    
    elif fusion_method == 'weighted':
        # 加權平均（需要特徵維度相同）
        # 先標準化到相同維度
        normalized_features = []
        target_cols = min(f.shape[1] for f in aligned_features)
        
        for features in aligned_features:
            if features.shape[1] > target_cols:
                # PCA 降維
                pca = PCA(n_components=target_cols)
                features = pca.fit_transform(features)
            elif features.shape[1] < target_cols:
                # 填充零
                padding = np.zeros((features.shape[0], target_cols - features.shape[1]))
                features = np.hstack([features, padding])
            
            normalized_features.append(features)
        
        # 加權融合
        fused_features = np.zeros_like(normalized_features[0])
        for features, weight in zip(normalized_features, feature_weights):
            fused_features += weight * features
    
    elif fusion_method == 'attention':
        # 注意力機制融合（簡化版）
        fused_features = attention_fusion(aligned_features, feature_weights)
    
    else:
        fused_features = np.hstack(aligned_features)
    
    # 降維到目標維度
    if target_dim is not None and fused_features.shape[1] > target_dim:
        pca = PCA(n_components=target_dim)
        fused_features = pca.fit_transform(fused_features)
    
    return fused_features


def attention_fusion(features_list, weights):
    """注意力機制特徵融合"""
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


def extract_trace_features(trace_data, inject_time=None):
    """
    從 trace/span 數據中提取特徵，參考 TracerCA 方法
    
    Args:
        trace_data: DataFrame 包含 trace/span 數據
                   需要的列: serviceName, methodName, operationName, startTime, duration
        inject_time: 故障注入時間點
    
    Returns:
        features: trace 特徵矩陣 [num_operations, num_features]
        node_names: 操作名稱列表
        service_graph: 服務依賴圖 (NetworkX Graph)
    """
    print("Extracting trace features...")
    
    if trace_data is None or trace_data.empty:
        return np.array([]), [], None
    
    try:
        # 1. 數據預處理 (參考 TracerCA)
        span_df = trace_data.copy()
        span_df["methodName"] = span_df["methodName"].fillna(span_df.get("operationName", ""))
        span_df["operation"] = span_df["serviceName"] + "_" + span_df["methodName"]
        
        # 2. 構建服務依賴圖
        service_graph = build_service_dependency_graph(span_df)
        
        # 3. 如果有 inject_time，執行 TracerCA 風格的分析
        if inject_time is not None:
            trace_features, operation_names = _extract_trace_anomaly_features(
                span_df, inject_time
            )
        else:
            # 沒有 inject_time 時，提取基本統計特徵
            trace_features, operation_names = _extract_trace_statistical_features(span_df)
        
        print(f"✓ Extracted {trace_features.shape[0]} trace operations with {trace_features.shape[1]} features")
        
        return trace_features, operation_names, service_graph
        
    except Exception as e:
        print(f"Trace feature extraction failed: {e}")
        return np.array([]), [], None


def _extract_trace_anomaly_features(span_df, inject_time):
    """基於 TracerCA 的異常檢測特徵提取"""
    
    # 分割正常和異常時期 (參考 TracerCA)
    normal_df = span_df[span_df["startTime"] + span_df["duration"] < inject_time]
    anomal_df = span_df[span_df["startTime"] + span_df["duration"] >= inject_time]
    
    if normal_df.empty or anomal_df.empty:
        return _extract_trace_statistical_features(span_df)
    
    # 計算正常時期的 SLO (參考 TracerCA 的 get_operation_slo)
    normal_slo = {}
    for op in normal_df["operation"].dropna().unique():
        op_data = normal_df[normal_df["operation"] == op]["duration"]
        mean_duration = op_data.mean() / 1000  # 轉換為毫秒
        std_duration = op_data.std() / 1000
        normal_slo[op] = {"mean": mean_duration, "std": std_duration}
    
    # 檢測異常 span
    anomal_df = anomal_df.copy()
    anomal_df["mean"] = anomal_df["operation"].apply(
        lambda op: normal_slo.get(op, {}).get("mean", 0)
    )
    anomal_df["std"] = anomal_df["operation"].apply(
        lambda op: normal_slo.get(op, {}).get("std", 1)
    )
    anomal_df["abnormal"] = (
        anomal_df["duration"] / 1000 >= 
        anomal_df["mean"] + 3 * anomal_df["std"]
    )
    
    # 計算 TracerCA 特徵
    operations = list(span_df["operation"].dropna().unique())
    features = []
    
    for op in operations:
        # Support: |abnormal_traces of operation A| / |total abnormal traces|
        op_abnormal_count = anomal_df[anomal_df["operation"] == op]["abnormal"].sum()
        total_abnormal_count = anomal_df["abnormal"].sum()
        support = op_abnormal_count / max(total_abnormal_count, 1)
        
        # Confidence: |abnormal traces of operation A| / |total traces of operation A|
        op_total_count = anomal_df[anomal_df["operation"] == op].shape[0]
        confidence = op_abnormal_count / max(op_total_count, 1)
        
        # JI (Jaccard Index): 2 * support * confidence / (support + confidence)
        ji = (2 * support * confidence / max(support + confidence, 1e-10) 
              if support + confidence > 0 else 0)
        
        # 額外的統計特徵
        op_normal_data = normal_df[normal_df["operation"] == op]
        op_anomal_data = anomal_df[anomal_df["operation"] == op]
        
        # 延遲統計
        normal_latency_mean = op_normal_data["duration"].mean() if not op_normal_data.empty else 0
        anomal_latency_mean = op_anomal_data["duration"].mean() if not op_anomal_data.empty else 0
        latency_change = (anomal_latency_mean - normal_latency_mean) / max(normal_latency_mean, 1)
        
        # 調用頻率變化
        normal_call_rate = len(op_normal_data) / max(len(normal_df), 1)
        anomal_call_rate = len(op_anomal_data) / max(len(anomal_df), 1)
        call_rate_change = anomal_call_rate - normal_call_rate
        
        features.append([
            support,                    # TracerCA support
            confidence,                 # TracerCA confidence  
            ji,                        # TracerCA JI score
            latency_change,            # 延遲變化率
            call_rate_change,          # 調用頻率變化
            normal_latency_mean,       # 正常時期平均延遲
            anomal_latency_mean,       # 異常時期平均延遲
            op_abnormal_count,         # 異常 span 數量
        ])
    
    feature_names = [
        'support', 'confidence', 'ji_score', 'latency_change', 
        'call_rate_change', 'normal_latency', 'anomal_latency', 'abnormal_count'
    ]
    
    return np.array(features), operations


def _extract_trace_statistical_features(span_df):
    """提取基本統計特徵 (當沒有 inject_time 時)"""
    
    operations = list(span_df["operation"].dropna().unique())
    features = []
    
    for op in operations:
        op_data = span_df[span_df["operation"] == op]
        
        # 延遲統計
        duration_stats = op_data["duration"].describe()
        
        # 調用頻率
        call_count = len(op_data)
        call_rate = call_count / len(span_df)
        
        # 時間分佈特徵
        time_span = (op_data["startTime"].max() - op_data["startTime"].min()) / 1e6  # 轉為秒
        
        features.append([
            duration_stats['mean'],     # 平均延遲
            duration_stats['std'],      # 延遲標準差
            duration_stats['min'],      # 最小延遲
            duration_stats['max'],      # 最大延遲
            duration_stats['50%'],      # 中位數延遲
            call_count,                 # 調用次數
            call_rate,                  # 調用頻率
            time_span,                  # 時間跨度
        ])
    
    feature_names = [
        'latency_mean', 'latency_std', 'latency_min', 'latency_max',
        'latency_median', 'call_count', 'call_rate', 'time_span'
    ]
    
    return np.array(features), operations


def build_service_dependency_graph(span_df):
    """
    從 trace 數據構建服務依賴圖
    
    Args:
        span_df: span 數據
    
    Returns:
        NetworkX DiGraph 表示服務依賴關係
    """
    print("Building service dependency graph from traces...")
    
    G = nx.DiGraph()
    
    try:
        # 添加所有服務作為節點
        services = span_df["serviceName"].dropna().unique()
        G.add_nodes_from(services)
        
        # 根據 trace 構建邊
        if 'traceId' in span_df.columns:
            for trace_id, trace_group in span_df.groupby('traceId'):
                # 按開始時間排序
                trace_group = trace_group.sort_values('startTime')
                services_in_trace = trace_group['serviceName'].tolist()
                
                # 添加相鄰服務之間的邊
                for i in range(len(services_in_trace) - 1):
                    src = services_in_trace[i]
                    dst = services_in_trace[i + 1]
                    
                    if src != dst:  # 避免自環
                        if G.has_edge(src, dst):
                            G[src][dst]['weight'] += 1
                        else:
                            G.add_edge(src, dst, weight=1)
        else:
            # 如果沒有 traceId，基於服務出現順序構建依賴
            unique_services = span_df['serviceName'].unique()
            for i in range(len(unique_services) - 1):
                G.add_edge(unique_services[i], unique_services[i + 1], weight=1)
        
        print(f"✓ Built service dependency graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
    except Exception as e:
        print(f"Failed to build service dependency graph: {e}")
    
    return G


def extract_service_topology_features(service_graph, service_names=None):
    """
    從服務依賴圖中提取拓樸特徵
    
    Args:
        service_graph: NetworkX 服務依賴圖
        service_names: 服務名稱列表
    
    Returns:
        topology_features: 拓樸特徵矩陣
        feature_names: 特徵名稱
    """
    if service_graph is None or service_graph.number_of_nodes() == 0:
        return np.array([]), []
    
    features = []
    feature_names = []
    
    try:
        # 計算圖的全局特徵
        num_nodes = service_graph.number_of_nodes()
        num_edges = service_graph.number_of_edges()
        
        # 連通性
        is_connected = nx.is_weakly_connected(service_graph)
        
        # 平均度
        degrees = dict(service_graph.degree())
        avg_degree = np.mean(list(degrees.values())) if degrees else 0
        
        # 聚類係數
        try:
            clustering = nx.average_clustering(service_graph.to_undirected())
        except:
            clustering = 0
        
        # 最短路徑長度
        try:
            if is_connected:
                avg_shortest_path = nx.average_shortest_path_length(service_graph)
            else:
                avg_shortest_path = 0
        except:
            avg_shortest_path = 0
        
        # 中心性指標
        try:
            pagerank_centrality = nx.pagerank(service_graph)
            betweenness_centrality = nx.betweenness_centrality(service_graph)
            
            # 取平均值和標準差
            pagerank_mean = np.mean(list(pagerank_centrality.values()))
            pagerank_std = np.std(list(pagerank_centrality.values()))
            betweenness_mean = np.mean(list(betweenness_centrality.values()))
            betweenness_std = np.std(list(betweenness_centrality.values()))
            
        except:
            pagerank_mean = pagerank_std = betweenness_mean = betweenness_std = 0
        
        # 組裝特徵
        global_features = [
            num_nodes, num_edges, avg_degree, clustering, avg_shortest_path,
            int(is_connected), pagerank_mean, pagerank_std, 
            betweenness_mean, betweenness_std
        ]
        
        global_feature_names = [
            'service_count', 'dependency_count', 'avg_degree', 'clustering',
            'avg_path_length', 'is_connected', 'pagerank_mean', 'pagerank_std',
            'betweenness_mean', 'betweenness_std'
        ]
        
        features.extend(global_features)
        feature_names.extend(global_feature_names)
        
        print(f"✓ Extracted {len(features)} service topology features")
        
    except Exception as e:
        print(f"Service topology feature extraction failed: {e}")
        return np.array([]), []
    
    return np.array(features).reshape(1, -1), feature_names


def kll_feature_processing(data, sketch_size=1024):
    """
    KLL (K-ary Lossy List) 特徵處理
    用於大規模數據的分位數估計和特徵提取
    
    Args:
        data: 輸入數據 (DataFrame 或 numpy array)
        sketch_size: KLL sketch 大小
    
    Returns:
        kll_features: KLL 處理後的特徵
        feature_names: 特徵名稱
    """
    print("Processing KLL features...")
    
    if isinstance(data, pd.DataFrame):
        numeric_data = data.select_dtypes(include=[np.number])
    else:
        numeric_data = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    
    features = []
    feature_names = []
    
    for col in numeric_data.columns:
        series = numeric_data[col].dropna()
        
        if len(series) == 0:
            continue
            
        try:
            # 簡化版 KLL：計算關鍵分位數
            quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            kll_quantiles = [series.quantile(q) for q in quantiles]
            
            # 額外的統計特徵
            additional_stats = [
                series.mean(),                    # 均值
                series.std(),                     # 標準差
                series.skew(),                    # 偏度
                series.kurtosis(),                # 峰度
                (series.max() - series.min()),    # 範圍
                len(series.unique()) / len(series) if len(series) > 0 else 0,  # 唯一值比例
            ]
            
            # 合併特徵
            col_features = kll_quantiles + additional_stats
            features.extend(col_features)
            
            # 特徵名稱
            quantile_names = [f'{col}_q{int(q*100):02d}' for q in quantiles]
            stat_names = [f'{col}_mean', f'{col}_std', f'{col}_skew', 
                         f'{col}_kurtosis', f'{col}_range', f'{col}_unique_ratio']
            feature_names.extend(quantile_names + stat_names)
            
        except Exception as e:
            print(f"KLL processing failed for {col}: {e}")
            # 回退到基本統計
            basic_stats = [series.mean(), series.std(), series.min(), series.max()]
            features.extend(basic_stats)
            feature_names.extend([f'{col}_mean', f'{col}_std', f'{col}_min', f'{col}_max'])
    
    if not features:
        print("No KLL features extracted, returning empty array")
        return np.array([]), []
    
    kll_features = np.array(features).reshape(1, -1)
    print(f"✓ Extracted {kll_features.shape[1]} KLL features")
    
    return kll_features, feature_names


# 測試函數
def test_feature_extraction():
    """測試特徵提取功能"""
    print("Testing feature extraction modules...")
    
    # 測試數據
    np.random.seed(42)
    test_data = pd.DataFrame({
        'time': range(100),
        'metric1': np.random.randn(100),
        'metric2': np.random.randn(100) + np.sin(np.arange(100) * 0.1),
        'log_text': ['INFO: normal operation'] * 50 + ['ERROR: system failure'] * 50
    })
    
    # 測試滑動窗口
    windows, timestamps = sliding_window_alignment(test_data, window_size=10)
    print(f"Created {len(windows)} windows")
    
    # 測試 STL 分解
    stl_features, stl_names = stl_decomposition(test_data[['metric1', 'metric2']])
    print(f"STL features shape: {stl_features.shape}")
    
    # 測試日誌特徵
    log_features, log_names = extract_log_features(test_data[['log_text']])
    print(f"Log features shape: {log_features.shape}")
    
    # 測試拓樸特徵
    adj_matrix = np.random.rand(5, 5)
    adj_matrix = (adj_matrix > 0.7).astype(float)
    topo_features, topo_names = compute_topology_features(adj_matrix)
    print(f"Topology features: {len(topo_features)}")
    
    print("Feature extraction tests completed!")


if __name__ == "__main__":
    test_feature_extraction()

def enhanced_feature_fusion(metric_features, log_features, trace_features, 
                          topology_features, error_features, 
                          fusion_method='adaptive', target_dim=64):
    """
    增强的多模态特征融合，专为 GNN-KAN 设计
    
    Args:
        metric_features: 度量特徵 (時間序列)
        log_features: 日誌特徵 
        trace_features: 追蹤特徵
        topology_features: 拓樸特徵
        error_features: 錯誤特徵
        fusion_method: 融合方法 ('adaptive', 'attention', 'weighted', 'concatenate')
        target_dim: 目標特徵維度
    
    Returns:
        fused_features: 融合後的特徵矩陣
        feature_names: 特徵名稱列表
    """
    print(f"Enhanced feature fusion with method: {fusion_method}")
    
    # 收集有效特徵
    valid_features = []
    feature_sources = []
    all_feature_names = []
    
    # 處理度量特徵
    if metric_features is not None and metric_features.size > 0:
        if metric_features.ndim == 1:
            metric_features = metric_features.reshape(1, -1)
        valid_features.append(metric_features)
        feature_sources.append('metric')
        all_feature_names.extend([f'metric_{i}' for i in range(metric_features.shape[1])])
    
    # 處理日誌特徵
    if log_features is not None and log_features.size > 0:
        if log_features.ndim == 1:
            log_features = log_features.reshape(1, -1)
        valid_features.append(log_features)
        feature_sources.append('log')
        all_feature_names.extend([f'log_{i}' for i in range(log_features.shape[1])])
    
    # 處理追蹤特徵
    if trace_features is not None and trace_features.size > 0:
        if trace_features.ndim == 1:
            trace_features = trace_features.reshape(1, -1)
        valid_features.append(trace_features)
        feature_sources.append('trace')
        all_feature_names.extend([f'trace_{i}' for i in range(trace_features.shape[1])])
    
    # 處理拓樸特徵
    if topology_features is not None and topology_features.size > 0:
        if topology_features.ndim == 1:
            topology_features = topology_features.reshape(1, -1)
        valid_features.append(topology_features)
        feature_sources.append('topology')
        all_feature_names.extend([f'topo_{i}' for i in range(topology_features.shape[1])])
    
    # 處理錯誤特徵
    if error_features is not None and error_features.size > 0:
        if error_features.ndim == 1:
            error_features = error_features.reshape(1, -1)
        valid_features.append(error_features)
        feature_sources.append('error')
        all_feature_names.extend([f'error_{i}' for i in range(error_features.shape[1])])
    
    if not valid_features:
        print("No valid features for fusion, returning empty array")
        return np.array([]), []
    
    # 對齊特徵維度
    max_rows = max(f.shape[0] for f in valid_features)
    aligned_features = []
    
    for features in valid_features:
        if features.shape[0] < max_rows:
            # 重複行以對齊
            repeat_times = max_rows // features.shape[0]
            remainder = max_rows % features.shape[0]
            
            repeated = np.tile(features, (repeat_times, 1))
            if remainder > 0:
                extra = features[:remainder]
                features_aligned = np.vstack([repeated, extra])
            else:
                features_aligned = repeated
        else:
            features_aligned = features[:max_rows]
        
        aligned_features.append(features_aligned)
    
    # 特徵融合
    if fusion_method == 'adaptive':
        fused_features = _adaptive_fusion(aligned_features, feature_sources)
    elif fusion_method == 'attention':
        fused_features = _attention_fusion_enhanced(aligned_features, feature_sources)
    elif fusion_method == 'weighted':
        weights = _compute_feature_weights(feature_sources)
        fused_features = _weighted_fusion(aligned_features, weights)
    else:  # concatenate
        fused_features = np.hstack(aligned_features)
    
    # 降維到目標維度
    if target_dim is not None and fused_features.shape[1] > target_dim:
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(target_dim, fused_features.shape[1]))
            fused_features = pca.fit_transform(fused_features)
            
            # 更新特徵名稱
            final_feature_names = [f'pca_{i}' for i in range(fused_features.shape[1])]
        except:
            # 如果 PCA 失敗，截斷特徵
            fused_features = fused_features[:, :target_dim]
            final_feature_names = all_feature_names[:target_dim]
    else:
        final_feature_names = all_feature_names[:fused_features.shape[1]] if len(all_feature_names) >= fused_features.shape[1] else all_feature_names
    
    print(f"✓ Enhanced fusion completed: {fused_features.shape}")
    return fused_features, final_feature_names


def _adaptive_fusion(features_list, feature_sources):
    """自適應特徵融合"""
    # 計算每個特徵源的重要性權重
    importance_weights = {
        'metric': 0.35,    # 度量特徵最重要
        'log': 0.25,       # 日誌特徵
        'trace': 0.20,     # 追蹤特徵  
        'topology': 0.15,  # 拓樸特徵
        'error': 0.05      # 錯誤特徵
    }
    
    # 標準化所有特徵到相同尺度
    from sklearn.preprocessing import StandardScaler
    
    normalized_features = []
    for features in features_list:
        scaler = StandardScaler()
        try:
            normalized = scaler.fit_transform(features)
        except:
            normalized = features
        normalized_features.append(normalized)
    
    # 找到最小的特徵維度
    min_cols = min(f.shape[1] for f in normalized_features)
    
    # 截斷或填充特徵到相同維度
    aligned_features = []
    for features in normalized_features:
        if features.shape[1] > min_cols:
            # 取前 min_cols 個特徵
            aligned = features[:, :min_cols]
        elif features.shape[1] < min_cols:
            # 用零填充
            padding = np.zeros((features.shape[0], min_cols - features.shape[1]))
            aligned = np.hstack([features, padding])
        else:
            aligned = features
        aligned_features.append(aligned)
    
    # 加權融合
    fused = np.zeros_like(aligned_features[0])
    for features, source in zip(aligned_features, feature_sources):
        weight = importance_weights.get(source, 0.1)
        fused += weight * features
    
    return fused


def _attention_fusion_enhanced(features_list, feature_sources):
    """增強的注意力機制融合"""
    # 計算注意力權重 (基於特徵方差)
    attention_scores = []
    
    for features in features_list:
        # 使用特徵方差作為注意力分數
        variance_score = np.mean(np.var(features, axis=0))
        attention_scores.append(variance_score)
    
    # 軟件最大化
    attention_scores = np.array(attention_scores)
    attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores))
    
    # 標準化特徵維度
    target_cols = min(f.shape[1] for f in features_list)
    normalized_features = []
    
    for features in features_list:
        if features.shape[1] != target_cols:
            # 使用平均池化調整維度
            if features.shape[1] > target_cols:
                # 降維：平均池化
                pool_size = features.shape[1] // target_cols
                pooled = []
                for i in range(target_cols):
                    start_idx = i * pool_size
                    end_idx = min((i + 1) * pool_size, features.shape[1])
                    pooled_col = np.mean(features[:, start_idx:end_idx], axis=1, keepdims=True)
                    pooled.append(pooled_col)
                features = np.hstack(pooled)
            else:
                # 升維：重複最後一列
                padding = np.repeat(features[:, -1:], target_cols - features.shape[1], axis=1)
                features = np.hstack([features, padding])
        
        normalized_features.append(features)
    
    # 注意力加權融合
    fused = np.zeros_like(normalized_features[0])
    for features, weight in zip(normalized_features, attention_weights):
        fused += weight * features
    
    return fused


def _weighted_fusion(features_list, weights):
    """加權特徵融合"""
    # 對齊特徵維度
    target_cols = min(f.shape[1] for f in features_list)
    aligned_features = []
    
    for features in features_list:
        if features.shape[1] > target_cols:
            aligned = features[:, :target_cols]
        elif features.shape[1] < target_cols:
            padding = np.zeros((features.shape[0], target_cols - features.shape[1]))
            aligned = np.hstack([features, padding])
        else:
            aligned = features
        aligned_features.append(aligned)
    
    # 加權融合
    fused = np.zeros_like(aligned_features[0])
    for features, weight in zip(aligned_features, weights):
        fused += weight * features
    
    return fused


def _compute_feature_weights(feature_sources):
    """計算特徵權重"""
    weight_map = {
        'metric': 0.4,
        'log': 0.25,
        'trace': 0.2,
        'topology': 0.1,
        'error': 0.05
    }
    
    weights = [weight_map.get(source, 0.1) for source in feature_sources]
    
    # 歸一化權重
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    
    return weights