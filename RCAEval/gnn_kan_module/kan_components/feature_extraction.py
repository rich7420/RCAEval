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
    🔧 重定向到統一日誌處理器
    """
    try:
        from ..processors.log_processors import extract_log_features as unified_extract_log_features
        return unified_extract_log_features(
            log_data=log_data,
            use_dla=use_dla,
            max_features=max_features,
            method='dla' if use_dla else 'tfidf',
            target_dim=min(max_features, 1000)
        )
    except Exception as e:
        print(f"⚠️ 重定向到統一日誌處理器失敗: {e}")
        return np.array([[0]]), ['default_log_feature']


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
    
    # 🎯 安全的PCA降維 - 檢查維度限制
    if features.shape[1] > embedding_dim:
        # 確保n_components不超過min(n_samples, n_features)
        max_components = min(features.shape[0], features.shape[1])
        actual_components = min(embedding_dim, max_components)
        
        if actual_components > 0:
            pca = PCA(n_components=actual_components)
            features = pca.fit_transform(features)
            
            # 如果降維後維度仍不足embedding_dim，用零填充
            if features.shape[1] < embedding_dim:
                padding = np.zeros((features.shape[0], embedding_dim - features.shape[1]))
                features = np.hstack([features, padding])
        else:
            # 無法進行PCA，直接填充
            if features.shape[1] < embedding_dim:
                padding = np.zeros((features.shape[0], embedding_dim - features.shape[1]))
                features = np.hstack([features, padding])
            else:
                features = features[:, :embedding_dim]
    
    return features


def stl_decomposition(metrics_data, seasonal=7, return_components=True):
    """
    🎯 簡化的STL分解 - 移除過度複雜性，專注核心功能
    
    Args:
        metrics_data: DataFrame 包含時間序列數據 或 numpy array
        seasonal: 季節性週期 (默認7)
        return_components: 是否返回所有組件
    
    Returns:
        decomposed_features: 分解後的特徵矩陣
        component_names: 組件名稱列表
    """
    print("🔧 Using simplified STL decomposition...")
    
    # 處理不同類型的輸入
    if isinstance(metrics_data, pd.DataFrame):
        data = metrics_data.select_dtypes(include=[np.number])
    elif isinstance(metrics_data, np.ndarray):
        if metrics_data.ndim == 1:
            data = pd.DataFrame({'metric': metrics_data})
        else:
            data = pd.DataFrame(metrics_data, columns=[f'metric_{i}' for i in range(metrics_data.shape[1])])
    else:
        try:
            data = pd.DataFrame(metrics_data)
        except:
            return np.array([[0]]), ['default_feature']

    decomposed_features = []
    component_names = []
    
    for col in data.columns:
        series = data[col].dropna()
        col_name = str(col)
        
        print(f"Processing {col} (length: {len(series)})...")
        
        # 🔧 數據質量檢查
        if len(series) < 6:
            print(f"⚠️ {col}: 數據太少，使用基本統計")
            basic_features = _compute_basic_statistics(series, col_name)
            decomposed_features.extend(basic_features['features'])
            component_names.extend(basic_features['names'])
            continue
        
        # 檢查數據變異性
        if series.std() < 1e-10:
            print(f"⚠️ {col}: 常數序列，使用常數特徵")
            const_features = _compute_constant_features(series, col_name)
            decomposed_features.extend(const_features['features'])
            component_names.extend(const_features['names'])
            continue
        
        # 🎯 簡化的STL分解嘗試
        optimal_seasonal = min(seasonal, len(series) // 3, 12)
        optimal_seasonal = max(optimal_seasonal, 3)
        
        try:
            if len(series) >= 2 * optimal_seasonal + 1:
                # 簡單的STL分解
                stl = STL(series, seasonal=optimal_seasonal, robust=True)
                result = stl.fit()
                
                # 驗證結果質量
                if _validate_stl_result_simple(result, series):
                    stl_features = _extract_stl_components_simple(result, col_name)
                    decomposed_features.extend(stl_features['features'])
                    component_names.extend(stl_features['names'])
                    print(f"✓ {col}: STL分解成功")
                    continue
            
            # 如果STL失敗，使用統計分解
            stat_features = _statistical_decomposition_simple(series, col_name)
            decomposed_features.extend(stat_features['features'])
            component_names.extend(stat_features['names'])
            print(f"✓ {col}: 使用統計分解")
                
        except Exception as e:
            print(f"⚠️ {col}: 分解失敗 - {e}，使用基本統計")
            fallback_features = _compute_basic_statistics(series, col_name)
            decomposed_features.extend(fallback_features['features'])
            component_names.extend(fallback_features['names'])
    
    # 🔧 特徵對齊和質量控制
    if decomposed_features:
        try:
            # 確保所有特徵都是數值
            cleaned_features = []
            for feat_list in decomposed_features:
                if isinstance(feat_list, (list, np.ndarray)):
                    feat_array = np.array(feat_list, dtype=float)
                    feat_array = np.nan_to_num(feat_array, nan=0.0, posinf=1e6, neginf=-1e6)
                    cleaned_features.append(feat_array)
                else:
                    cleaned_features.append(np.array([float(feat_list)]))
            
            # 對齊特徵長度
            if len(cleaned_features) == 1:
                final_features = cleaned_features[0].reshape(-1, 1)
            else:
                final_features = np.column_stack(cleaned_features)
            
        except Exception as e:
            print(f"⚠️ 特徵對齊失敗: {e}")
            final_features = np.array([[0]])
            component_names = ['default_feature']
    else:
        final_features = np.array([[0]])
        component_names = ['default_feature']
    
    print(f"✓ 簡化STL分解完成: {final_features.shape[1]} 特徵")
    return final_features, component_names


def _compute_basic_statistics(series, col_name):
    """計算基本統計特徵"""
    try:
        if len(series) == 0:
            return {'features': [0] * 6, 'names': [f'{col_name}_feat_{i}' for i in range(6)]}
        
        features = [
            np.mean(series),
            np.std(series) + 1e-8,
            np.min(series),
            np.max(series),
            np.median(series),
            len(series)
        ]
        
        names = [
            f'{col_name}_mean', f'{col_name}_std', f'{col_name}_min', 
            f'{col_name}_max', f'{col_name}_median', f'{col_name}_length'
        ]
        
        return {'features': features, 'names': names}
        
    except Exception:
        return {'features': [0] * 6, 'names': [f'{col_name}_feat_{i}' for i in range(6)]}


def _compute_constant_features(series, col_name):
    """處理常數序列"""
    const_val = series.iloc[0] if len(series) > 0 else 0
    
    features = [const_val, 0, const_val, const_val, const_val, len(series)]
    names = [
        f'{col_name}_const_mean', f'{col_name}_const_std', f'{col_name}_const_min', 
        f'{col_name}_const_max', f'{col_name}_const_median', f'{col_name}_const_length'
    ]
    
    return {'features': features, 'names': names}


def _validate_stl_result_simple(result, original_series):
    """簡化的STL分解結果驗證"""
    try:
        if hasattr(result, 'trend') and hasattr(result, 'seasonal') and hasattr(result, 'resid'):
            trend = result.trend
            seasonal = result.seasonal  
            residual = result.resid
            
            # 檢查是否有太多NaN值
            if (trend.isna().sum() > len(trend) * 0.5 or 
                seasonal.isna().sum() > len(seasonal) * 0.5):
                return False
            
            return True
        
        return False
        
    except Exception:
        return False


def _extract_stl_components_simple(result, col_name):
    """簡化的STL組件特徵提取"""
    try:
        trend = result.trend.fillna(method='ffill').fillna(method='bfill').fillna(0)
        seasonal = result.seasonal.fillna(0)
        residual = result.resid.fillna(0)
        
        # 基本組件統計
        features = [
            np.mean(trend),
            np.std(trend),
            np.mean(seasonal),
            np.std(seasonal),
            np.mean(residual),
            np.std(residual)
        ]
        
        names = [
            f'{col_name}_trend_mean', f'{col_name}_trend_std',
            f'{col_name}_seasonal_mean', f'{col_name}_seasonal_std',
            f'{col_name}_residual_mean', f'{col_name}_residual_std'
        ]
        
        return {'features': features, 'names': names}
        
    except Exception as e:
        print(f"特徵提取失敗: {e}")
        return {'features': [0] * 6, 'names': [f'{col_name}_feat_{i}' for i in range(6)]}


def _statistical_decomposition_simple(series, col_name):
    """簡化的統計分解"""
    try:
        # 移動平均趨勢
        window = min(7, len(series) // 3)
        if window < 3:
            window = 3
            
        trend = series.rolling(window=window, center=True).mean()
        trend = trend.fillna(method='ffill').fillna(method='bfill').fillna(series.mean())
        
        # 去趨勢
        detrended = series - trend
        
        # 簡單季節性
        seasonal_mean = detrended.mean()
        seasonal = pd.Series([seasonal_mean] * len(series), index=series.index)
        
        # 殘差
        residual = series - trend - seasonal
        
        # 計算特徵
        features = [
            np.mean(trend),
            np.std(trend),
            np.mean(seasonal),
            np.std(seasonal),
            np.mean(residual),
            np.std(residual)
        ]
        
        names = [
            f'{col_name}_stat_trend_mean', f'{col_name}_stat_trend_std',
            f'{col_name}_stat_seasonal_mean', f'{col_name}_stat_seasonal_std',
            f'{col_name}_stat_residual_mean', f'{col_name}_stat_residual_std'
        ]
        
        return {'features': features, 'names': names}
        
    except Exception as e:
        print(f"統計分解失敗: {e}")
        return _compute_basic_statistics(series, col_name)


# 移除過度複雜的輔助函數，保留核心安全函數
def _safe_skew(series):
    """安全的偏度計算"""
    try:
        if len(series) > 2:
            return series.skew()
        return 0
    except:
        return 0


def _safe_kurtosis(series):
    """安全的峰度計算"""
    try:
        if len(series) > 3:
            return series.kurtosis()
        return 0
    except:
        return 0


def _safe_autocorr(series, lag=1):
    """安全的自相關計算"""
    try:
        if len(series) > lag + 1:
            return series.autocorr(lag=lag)
        return 0
    except:
        return 0


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
                # 🎯 安全的PCA降維
                max_components = min(features.shape[0], features.shape[1])
                actual_components = min(target_cols, max_components)
                
                if actual_components > 0:
                    pca = PCA(n_components=actual_components)
                    features = pca.fit_transform(features)
                    
                    # 如果降維後仍不足，填充零
                    if features.shape[1] < target_cols:
                        padding = np.zeros((features.shape[0], target_cols - features.shape[1]))
                        features = np.hstack([features, padding])
                else:
                    # 無法PCA，直接截斷或填充
                    features = features[:, :target_cols] if features.shape[1] >= target_cols else features
                    if features.shape[1] < target_cols:
                        padding = np.zeros((features.shape[0], target_cols - features.shape[1]))
                        features = np.hstack([features, padding])
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
    
    # 🎯 安全的PCA降維到目標維度
    if target_dim is not None and fused_features.shape[1] > target_dim:
        max_components = min(fused_features.shape[0], fused_features.shape[1])
        actual_components = min(target_dim, max_components)
        
        if actual_components > 0:
            pca = PCA(n_components=actual_components)
            fused_features = pca.fit_transform(fused_features)
            
            # 如果降維後仍不足target_dim，填充零
            if fused_features.shape[1] < target_dim:
                padding = np.zeros((fused_features.shape[0], target_dim - fused_features.shape[1]))
                fused_features = np.hstack([fused_features, padding])
        else:
            # 無法PCA，直接截斷或填充
            if fused_features.shape[1] >= target_dim:
                fused_features = fused_features[:, :target_dim]
            else:
                padding = np.zeros((fused_features.shape[0], target_dim - fused_features.shape[1]))
                fused_features = np.hstack([fused_features, padding])
    elif target_dim is not None and fused_features.shape[1] < target_dim:
        # 特徵不足，填充零
        padding = np.zeros((fused_features.shape[0], target_dim - fused_features.shape[1]))
        fused_features = np.hstack([fused_features, padding])
    
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
            normal_latency_mean,       # 正常时期平均延遲
            anomal_latency_mean,       # 異常時期平均延遲
            op_abnormal_count         # 異常 span 數量
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


# 注意：compute_service_criticality_weights 函數已移動到 utils.py 中統一管理
# 避免重複定義和循環導入問題


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

# 🔧 移除重複定義 - 這些函數已在 gnn_kan.py 中定義，不需要重複

def enhanced_feature_fusion(trace_features, metric_features, log_features, 
                          fusion_method='adaptive', target_dim=64):
    """
    增強的多模態特徵融合 - 智能自適應融合
    
    Args:
        trace_features: trace 特徵 [nodes, trace_dim]
        metric_features: metric 特徵 [nodes, metric_dim] 
        log_features: log 特徵 [nodes, log_dim]
        fusion_method: 融合方法 ('adaptive', 'concatenate', 'weighted', 'attention')
        target_dim: 目標特徵維度
    
    Returns:
        fused_features: 融合後的特徵 [nodes, target_dim]
        fusion_weights: 各模態的融合權重
    """
    print(f"🔗 Enhanced feature fusion with method: {fusion_method}")
    
    # 收集有效的特徵模態
    valid_features = []
    modality_names = []
    
    if trace_features is not None and trace_features.size > 0:
        if trace_features.ndim == 1:
            trace_features = trace_features.reshape(1, -1)
        valid_features.append(trace_features)
        modality_names.append('trace')
    
    if metric_features is not None and metric_features.size > 0:
        if metric_features.ndim == 1:
            metric_features = metric_features.reshape(1, -1)
        valid_features.append(metric_features)
        modality_names.append('metric')
    
    if log_features is not None and log_features.size > 0:
        if log_features.ndim == 1:
            log_features = log_features.reshape(1, -1)
        valid_features.append(log_features)
        modality_names.append('log')
    
    if not valid_features:
        print("⚠️ No valid features for fusion, returning zero features")
        return np.zeros((1, target_dim)), {}
    
    # 對齊節點數量
    max_nodes = max(f.shape[0] for f in valid_features)
    aligned_features = []
    
    for features in valid_features:
        if features.shape[0] < max_nodes:
            # 重複最後一行以對齊節點數
            padding = np.repeat(features[-1:], max_nodes - features.shape[0], axis=0)
            features = np.vstack([features, padding])
        aligned_features.append(features)
    
    print(f"✓ Aligned {len(aligned_features)} modalities to {max_nodes} nodes")
    
    # 根據融合方法處理
    if fusion_method == 'adaptive':
        fused_features, fusion_weights = _adaptive_fusion(aligned_features, modality_names, target_dim)
    elif fusion_method == 'attention':
        fused_features, fusion_weights = _attention_fusion_enhanced(aligned_features, modality_names, target_dim)
    elif fusion_method == 'weighted':
        fused_features, fusion_weights = _weighted_fusion(aligned_features, modality_names, target_dim)
    else:  # concatenate
        fused_features, fusion_weights = _concatenate_fusion(aligned_features, modality_names, target_dim)
    
    print(f"✓ Enhanced fusion completed: {fused_features.shape} -> target_dim={target_dim}")
    return fused_features, fusion_weights


def _adaptive_fusion(features_list, modality_names, target_dim):
    """自適應融合 - 根據特徵質量動態調整權重"""
    
    # 計算每個模態的特徵質量指標
    quality_scores = []
    for features in features_list:
        # 特徵質量 = 方差 + 非零比例 + 數值穩定性
        variance = np.var(features, axis=0).mean()
        non_zero_ratio = np.mean(features != 0)
        stability = 1.0 / (1.0 + np.std(features))
        
        quality = variance * non_zero_ratio * stability
        quality_scores.append(quality)
    
    # 歸一化質量分數作為權重
    total_quality = sum(quality_scores)
    if total_quality > 0:
        adaptive_weights = [q / total_quality for q in quality_scores]
    else:
        adaptive_weights = [1.0 / len(features_list)] * len(features_list)
    
    # 加權融合
    fused_features, _ = _weighted_fusion(features_list, modality_names, target_dim, adaptive_weights)
    
    fusion_weights = dict(zip(modality_names, adaptive_weights))
    return fused_features, fusion_weights


def _attention_fusion_enhanced(features_list, modality_names, target_dim):
    """增強的注意力融合"""
    
    # 標準化所有特徵到相同維度
    normalized_features = []
    min_dim = min(f.shape[1] for f in features_list)
    
    for features in features_list:
        if features.shape[1] > min_dim:
            # PCA 降維
            pca = PCA(n_components=min_dim)
            features = pca.fit_transform(features)
        elif features.shape[1] < min_dim:
            # 零填充
            padding = np.zeros((features.shape[0], min_dim - features.shape[1]))
            features = np.hstack([features, padding])
        
        # 標準化
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        normalized_features.append(features)
    
    # 計算注意力權重 (簡化版)
    attention_weights = []
    for i, features in enumerate(normalized_features):
        # 特徵重要性 = 平均絕對值
        importance = np.mean(np.abs(features))
        attention_weights.append(importance)
    
    # 軟最大歸一化
    attention_weights = np.array(attention_weights)
    attention_weights = attention_weights / np.sum(attention_weights)
    
    # 加權組合
    fused_features = np.zeros_like(normalized_features[0])
    for features, weight in zip(normalized_features, attention_weights):
        fused_features += weight * features
    
    # 降維到目標維度
    if fused_features.shape[1] > target_dim:
        pca = PCA(n_components=target_dim)
        fused_features = pca.fit_transform(fused_features)
    elif fused_features.shape[1] < target_dim:
        padding = np.zeros((fused_features.shape[0], target_dim - fused_features.shape[1]))
        fused_features = np.hstack([fused_features, padding])
    
    fusion_weights = dict(zip(modality_names, attention_weights))
    return fused_features, fusion_weights


def _weighted_fusion(features_list, modality_names, target_dim, weights=None):
    """加權融合"""
    
    if weights is None:
        # 默認權重：trace > metric > log
        default_weights = {'trace': 0.4, 'metric': 0.4, 'log': 0.2}
        weights = [default_weights.get(name, 1.0 / len(features_list)) 
                  for name in modality_names]
    
    # 標準化權重
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    else:
        weights = [1.0 / len(features_list)] * len(features_list)
    
    # 對齊特徵維度
    min_dim = min(f.shape[1] for f in features_list)
    aligned_features = []
    
    for features in features_list:
        if features.shape[1] > min_dim:
            # PCA 降維
            pca = PCA(n_components=min_dim)
            features = pca.fit_transform(features)
        elif features.shape[1] < min_dim:
            # 零填充
            padding = np.zeros((features.shape[0], min_dim - features.shape[1]))
            features = np.hstack([features, padding])
        
        # 標準化
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        aligned_features.append(features)
    
    # 加權組合
    fused_features = np.zeros_like(aligned_features[0])
    for features, weight in zip(aligned_features, weights):
        fused_features += weight * features
    
    # 調整到目標維度
    if fused_features.shape[1] > target_dim:
        pca = PCA(n_components=target_dim)
        fused_features = pca.fit_transform(fused_features)
    elif fused_features.shape[1] < target_dim:
        padding = np.zeros((fused_features.shape[0], target_dim - fused_features.shape[1]))
        fused_features = np.hstack([fused_features, padding])
    
    fusion_weights = dict(zip(modality_names, weights))
    return fused_features, fusion_weights


def _concatenate_fusion(features_list, modality_names, target_dim):
    """連接融合"""
    
    # 標準化所有特徵
    standardized_features = []
    for features in features_list:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        standardized_features.append(features)
    
    # 連接特徵
    fused_features = np.hstack(standardized_features)
    
    # 降維到目標維度
    if fused_features.shape[1] > target_dim:
        pca = PCA(n_components=target_dim)
        fused_features = pca.fit_transform(fused_features)
    elif fused_features.shape[1] < target_dim:
        padding = np.zeros((fused_features.shape[0], target_dim - fused_features.shape[1]))
        fused_features = np.hstack([fused_features, padding])
    
    # 等權重
    equal_weights = [1.0 / len(modality_names)] * len(modality_names)
    fusion_weights = dict(zip(modality_names, equal_weights))
    
    return fused_features, fusion_weights


# 確保所有函數都可以被導入
__all__ = [
    'sliding_window_alignment',
    'extract_log_features', 
    'stl_decomposition',
    'compute_topology_features',
    'extract_error_features',
    'feature_fusion',
    'extract_trace_features',
    'build_service_dependency_graph',
    'extract_service_topology_features', 
    'kll_feature_processing',
    'enhanced_feature_fusion',  # 新增
    'test_feature_extraction',
    'compute_service_criticality_weights'  # 新增
]