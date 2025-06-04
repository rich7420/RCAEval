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


def stl_decomposition(metrics_data, seasonal=7, return_components=True):
    """
    STL 分解將 metrics 分解為趨勢、季節性和殘差
    
    Args:
        metrics_data: DataFrame 包含時間序列數據
        seasonal: 季節性週期
        return_components: 是否返回所有組件
    
    Returns:
        decomposed_features: 分解後的特徵
        component_names: 組件名稱
    """
    if isinstance(metrics_data, pd.DataFrame):
        data = metrics_data.select_dtypes(include=[np.number])
    else:
        data = pd.DataFrame(metrics_data)
    
    decomposed_features = []
    component_names = []
    
    for col in data.columns:
        series = data[col].dropna()
        
        if len(series) < 2 * seasonal:
            # 數據太少，無法分解，使用原始數據
            features = series.values.reshape(-1, 1)
            names = [f'{col}_original']
        else:
            try:
                # STL 分解
                stl = STL(series, seasonal=seasonal)
                result = stl.fit()
                
                if return_components:
                    # 返回所有組件
                    trend = result.trend.fillna(0).values
                    seasonal_comp = result.seasonal.fillna(0).values
                    residual = result.resid.fillna(0).values
                    
                    features = np.column_stack([trend, seasonal_comp, residual])
                    names = [f'{col}_trend', f'{col}_seasonal', f'{col}_residual']
                else:
                    # 只返回趨勢
                    features = result.trend.fillna(0).values.reshape(-1, 1)
                    names = [f'{col}_trend']
                    
            except Exception as e:
                print(f"STL decomposition failed for {col}: {e}")
                # 回退到原始數據
                features = series.values.reshape(-1, 1)
                names = [f'{col}_original']
        
        decomposed_features.append(features)
        component_names.extend(names)
    
    # 對齊所有特徵的長度
    min_length = min(f.shape[0] for f in decomposed_features)
    decomposed_features = [f[:min_length] for f in decomposed_features]
    
    # 合併所有特徵
    if decomposed_features:
        final_features = np.column_stack(decomposed_features)
    else:
        final_features = np.array([])
    
    return final_features, component_names


def kll_feature_processing(features, k=1024, epsilon=0.01):
    """
    KLL (K-Minimum/Low-Latency) 算法特徵處理
    用於特徵選擇和維度降低
    
    Args:
        features: 輸入特徵矩陣
        k: KLL 參數
        epsilon: 精度參數
    
    Returns:
        processed_features: 處理後的特徵
    """
    if features.size == 0:
        return features
    
    features = np.array(features)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    
    # KLL 草圖算法的簡化實現
    # 1. 特徵重要性評估
    feature_importance = np.var(features, axis=0)
    
    # 2. 選擇前 k 個最重要的特徵
    if features.shape[1] > k:
        top_k_indices = np.argsort(feature_importance)[-k:]
        features = features[:, top_k_indices]
    
    # 3. 量化處理
    processed_features = quantize_features(features, epsilon)
    
    # 4. 標準化
    scaler = StandardScaler()
    processed_features = scaler.fit_transform(processed_features)
    
    return processed_features


def quantize_features(features, epsilon=0.01):
    """特徵量化處理"""
    quantized = np.zeros_like(features)
    
    for i in range(features.shape[1]):
        col = features[:, i]
        min_val, max_val = np.min(col), np.max(col)
        
        if max_val - min_val > 0:
            # 計算量化等級
            num_levels = int(1 / epsilon)
            quantized[:, i] = np.round(
                (col - min_val) / (max_val - min_val) * (num_levels - 1)
            ) / (num_levels - 1) * (max_val - min_val) + min_val
        else:
            quantized[:, i] = col
    
    return quantized


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