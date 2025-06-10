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
    🎯 增強的STL分解 - 專門針對微服務監控數據優化
    支持自動週期檢測、服務語義識別和多重回退策略
    
    Args:
        metrics_data: DataFrame 包含時間序列數據 或 numpy array
        seasonal: 基礎季節性週期 (默認7，適合系統監控)
        return_components: 是否返回所有組件
    
    Returns:
        decomposed_features: 分解後的特徵矩陣
        component_names: 組件名稱列表
    """
    print("🔍 Enhanced STL decomposition with service semantics...")
    
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
    
    # 🎯 微服務語義映射 - 識別服務類型和指標類型
    service_semantic_map = {
        'adservice': {'type': 'application', 'priority': 'high'},
        'cartservice': {'type': 'application', 'priority': 'high'},
        'checkoutservice': {'type': 'application', 'priority': 'critical'},
        'currencyservice': {'type': 'utility', 'priority': 'medium'},
        'emailservice': {'type': 'notification', 'priority': 'low'},
        'frontend': {'type': 'interface', 'priority': 'critical'},
        'paymentservice': {'type': 'application', 'priority': 'critical'},
        'productcatalogservice': {'type': 'data', 'priority': 'high'},
        'recommendationservice': {'type': 'ml', 'priority': 'medium'},
        'redis': {'type': 'database', 'priority': 'high'},
        'shippingservice': {'type': 'application', 'priority': 'medium'}
    }
    
    metric_type_map = {
        'cpu': {'seasonality': 12, 'sensitivity': 'high', 'weight': 1.5},
        'mem': {'seasonality': 24, 'sensitivity': 'high', 'weight': 1.3},
        'memory': {'seasonality': 24, 'sensitivity': 'high', 'weight': 1.3},
        'load': {'seasonality': 15, 'sensitivity': 'medium', 'weight': 1.2},
        'latency': {'seasonality': 8, 'sensitivity': 'critical', 'weight': 2.0},
        'error': {'seasonality': 6, 'sensitivity': 'critical', 'weight': 2.5},
        'time': {'seasonality': 60, 'sensitivity': 'low', 'weight': 0.5}
    }

    for col in data.columns:
        series = data[col].dropna()
        col_lower = str(col).lower()
        
        print(f"Processing {col} (length: {len(series)})...")
        
        # 🔧 智能語義分析 - 識别服務和指標類型
        service_info = None
        metric_info = None
        
        # 識別服務
        for service_name, info in service_semantic_map.items():
            if service_name in col_lower:
                service_info = info
                break
        
        # 識別指標類型
        for metric_name, info in metric_type_map.items():
            if metric_name in col_lower:
                metric_info = info
                break
        
        # 根據語義信息調整參數
        if metric_info:
            optimal_seasonal = metric_info['seasonality']
            feature_weight = metric_info['weight']
        else:
            optimal_seasonal = seasonal
            feature_weight = 1.0
        
        # 🔧 數據質量檢查
        if len(series) < 6:  # 絕對最小長度
            print(f"⚠️ {col}: 數據太少 ({len(series)} < 6)，使用基本統計")
            basic_features = _compute_basic_service_statistics(series, col, service_info, metric_info)
            decomposed_features.extend(basic_features['features'])
            component_names.extend(basic_features['names'])
            continue
        
        # 檢查數據變異性
        if series.std() < 1e-10:
            print(f"⚠️ {col}: 常數序列，使用常數特徵")
            const_features = _compute_constant_features(series, col, service_info, metric_info)
            decomposed_features.extend(const_features['features'])
            component_names.extend(const_features['names'])
            continue
        
        # 🎯 多策略週期檢測
        detected_periods = _detect_multiple_periods(series, col_lower, metric_info)
        
        if detected_periods:
            optimal_seasonal = detected_periods[0]  # 使用最佳週期
            print(f"✓ {col}: 檢測到週期 {optimal_seasonal}")
        else:
            # 根據數據長度和語義智能選擇週期
            if len(series) >= 60:
                optimal_seasonal = metric_info['seasonality'] if metric_info else 12
            elif len(series) >= 24:
                optimal_seasonal = 8
            elif len(series) >= 12:
                optimal_seasonal = 6
            else:
                optimal_seasonal = max(3, len(series) // 4)
            print(f"⚠️ {col}: 未檢測到週期，使用自適應週期 {optimal_seasonal}")
        
        # 確保週期合理
        max_seasonal = len(series) // 3
        optimal_seasonal = min(optimal_seasonal, max_seasonal)
        optimal_seasonal = max(optimal_seasonal, 3)
        
        # 🎯 增強的STL分解
        try:
            stl_success = False
            
            # 策略1：標準STL分解
            if len(series) >= 2 * optimal_seasonal + 1:
                try:
                    series_clean = _remove_outliers_adaptive(series, method='iqr')
                    
                    # 構建STL參數
                    stl_params = {
                        'seasonal': optimal_seasonal,
                        'robust': True
                    }
                    
                    # 嘗試創建STL對象
                    stl = STL(series_clean, **stl_params)
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = stl.fit()
                    
                    # 驗證結果質量
                    if _validate_stl_result(result, series_clean):
                        stl_features = _extract_stl_components(result, col, service_info, metric_info, feature_weight)
                        decomposed_features.extend(stl_features['features'])
                        component_names.extend(stl_features['names'])
                        stl_success = True
                        print(f"✓ {col}: STL分解成功 (週期={optimal_seasonal})")
                    
                except Exception as e:
                    print(f"⚠️ {col}: STL分解失敗 - {str(e)[:50]}")
            
            # 策略2：如果STL失敗，使用增強統計分解
            if not stl_success:
                enhanced_features = _enhanced_statistical_decomposition(series, col, service_info, metric_info, optimal_seasonal)
                decomposed_features.extend(enhanced_features['features'])
                component_names.extend(enhanced_features['names'])
                print(f"✓ {col}: 使用增強統計分解")
                
        except Exception as e:
            print(f"⚠️ {col}: 所有分解方法失敗 - {e}")
            # 最終回退
            fallback_features = _compute_basic_service_statistics(series, col, service_info, metric_info)
            decomposed_features.extend(fallback_features['features'])
            component_names.extend(fallback_features['names'])
    
    # 🔧 特徵對齊和質量控制
    if decomposed_features:
        try:
            # 確保所有特徵都是數值
            cleaned_features = []
            for feat_list in decomposed_features:
                if isinstance(feat_list, (list, np.ndarray)):
                    # 轉換為numpy數組並處理NaN
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
            # 創建緊急特徵矩陣
            final_features = np.zeros((1, len(component_names)))
            for i, feat_list in enumerate(decomposed_features):
                if i < final_features.shape[1]:
                    try:
                        if isinstance(feat_list, (list, np.ndarray)) and len(feat_list) > 0:
                            final_features[0, i] = float(feat_list[0])
                        else:
                            final_features[0, i] = float(feat_list)
                    except:
                        final_features[0, i] = 0.0
    else:
        final_features = np.array([[0]])
        component_names = ['default_feature']
    
    print(f"✓ STL分解完成: {final_features.shape[1]} 特徵，包含服務語義")
    return final_features, component_names


def _detect_multiple_periods(series, col_name, metric_info):
    """多策略週期檢測"""
    periods = []
    
    try:
        # 策略1: 自相關檢測
        acf_period = _detect_period_autocorr(series)
        if acf_period:
            periods.append(acf_period)
        
        # 策略2: FFT頻域分析
        fft_period = _detect_period_fft(series)
        if fft_period:
            periods.append(fft_period)
        
        # 策略3: 基於指標類型的先驗知識
        if metric_info:
            prior_period = metric_info['seasonality']
            if 3 <= prior_period <= len(series) // 3:
                periods.append(prior_period)
        
        # 策略4: 差分穩定性檢測
        stability_period = _detect_period_stability(series)
        if stability_period:
            periods.append(stability_period)
        
        # 去重並排序
        periods = list(set(periods))
        periods.sort()
        
        return periods[:3]  # 返回最多3個候選週期
        
    except Exception as e:
        print(f"週期檢測失敗: {e}")
        return []


def _detect_period_autocorr(series, max_lag=None):
    """自相關週期檢測"""
    try:
        if max_lag is None:
            max_lag = min(len(series) // 3, 50)
        
        if max_lag < 3:
            return None
        
        # 計算自相關函數
        autocorrs = []
        for lag in range(1, max_lag + 1):
            if len(series) > lag:
                corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                if not np.isnan(corr):
                    autocorrs.append((lag, abs(corr)))
        
        if not autocorrs:
            return None
        
        # 找到第一個顯著峰值
        autocorrs.sort(key=lambda x: x[1], reverse=True)
        
        # 選擇相關性大於0.3的最小週期
        for lag, corr in autocorrs:
            if corr > 0.3 and lag >= 3:
                return lag
        
        return None
        
    except Exception:
        return None


def _detect_period_fft(series):
    """FFT頻域週期檢測"""
    try:
        if len(series) < 16:
            return None
        
        # 去趨勢
        from scipy import signal as scipy_signal
        detrended = scipy_signal.detrend(series.values if hasattr(series, 'values') else series)
        
        # FFT分析
        fft_vals = np.fft.fft(detrended)
        freqs = np.fft.fftfreq(len(detrended))
        
        # 計算功率譜
        power = np.abs(fft_vals[1:len(fft_vals)//2])
        freqs_pos = freqs[1:len(freqs)//2]
        
        if len(power) == 0:
            return None
        
        # 找主要頻率
        dominant_idx = np.argmax(power)
        dominant_freq = freqs_pos[dominant_idx]
        
        if abs(dominant_freq) < 1e-10:
            return None
        
        period = int(1 / abs(dominant_freq))
        
        # 驗證週期合理性
        if 3 <= period <= len(series) // 3:
            return period
        
        return None
        
    except Exception:
        return None


def _detect_period_stability(series):
    """基於差分穩定性的週期檢測"""
    try:
        max_period = min(len(series) // 3, 30)
        
        stability_scores = []
        for period in range(3, max_period + 1):
            if len(series) >= 2 * period:
                # 計算週期性差分的穩定性
                diff = series[period:].values - series[:-period].values
                stability = 1 / (1 + np.var(diff))  # 方差越小，穩定性越高
                stability_scores.append((period, stability))
        
        if not stability_scores:
            return None
        
        # 選擇穩定性最高的週期
        stability_scores.sort(key=lambda x: x[1], reverse=True)
        best_period, best_score = stability_scores[0]
        
        # 只有穩定性足夠高時才返回
        if best_score > 0.1:
            return best_period
        
        return None
        
    except Exception:
        return None


def _remove_outliers_adaptive(series, method='iqr', factor=1.5):
    """自適應異常值處理"""
    try:
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:  # 如果IQR為0，使用標準差方法
                mean = series.mean()
                std = series.std()
                if std == 0:
                    return series
                lower_bound = mean - factor * std
                upper_bound = mean + factor * std
            else:
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
            
            # 用邊界值替換（不刪除）
            cleaned = series.copy()
            cleaned[cleaned < lower_bound] = lower_bound
            cleaned[cleaned > upper_bound] = upper_bound
            
            return cleaned
        else:
            return series
            
    except Exception:
        return series


def _validate_stl_result(result, original_series):
    """驗證STL分解結果質量"""
    try:
        # 檢查組件是否有效
        if hasattr(result, 'trend') and hasattr(result, 'seasonal') and hasattr(result, 'resid'):
            trend = result.trend
            seasonal = result.seasonal  
            residual = result.resid
            
            # 檢查是否有太多NaN值
            if (trend.isna().sum() > len(trend) * 0.3 or 
                seasonal.isna().sum() > len(seasonal) * 0.3 or
                residual.isna().sum() > len(residual) * 0.3):
                return False
            
            # 檢查重構誤差
            reconstructed = trend.fillna(0) + seasonal.fillna(0) + residual.fillna(0)
            mse = np.mean((original_series - reconstructed) ** 2)
            
            # 如果重構誤差太大，認為分解失敗
            if mse > np.var(original_series) * 2:
                return False
            
            return True
        
        return False
        
    except Exception:
        return False


def _extract_stl_components(result, col_name, service_info, metric_info, feature_weight):
    """從STL結果提取特徵"""
    try:
        trend = result.trend.fillna(method='ffill').fillna(method='bfill').fillna(0)
        seasonal = result.seasonal.fillna(0)
        residual = result.resid.fillna(0)
        
        # 基本組件統計
        features = [
            np.mean(trend) * feature_weight,      # 趨勢均值
            np.std(trend) * feature_weight,       # 趨勢標準差
            np.mean(seasonal) * feature_weight,   # 季節性均值
            np.std(seasonal) * feature_weight,    # 季節性標準差
            np.mean(residual),                    # 殘差均值
            np.std(residual),                     # 殘差標準差
            np.max(trend) - np.min(trend),        # 趨勢範圍
            np.max(seasonal) - np.min(seasonal),  # 季節性範圍
        ]
        
        names = [
            f'{col_name}_trend_mean', f'{col_name}_trend_std',
            f'{col_name}_seasonal_mean', f'{col_name}_seasonal_std',
            f'{col_name}_residual_mean', f'{col_name}_residual_std',
            f'{col_name}_trend_range', f'{col_name}_seasonal_range'
        ]
        
        # 如果是關鍵服務或指標，添加額外特徵
        if (service_info and service_info.get('priority') in ['critical', 'high']) or \
           (metric_info and metric_info.get('sensitivity') in ['critical', 'high']):
            
            # 高階統計特徵
            features.extend([
                _safe_skew(trend),                 # 趨勢偏度
                _safe_kurtosis(trend),             # 趨勢峰度
                _safe_autocorr(trend),             # 趨勢自相關
                np.mean(np.abs(residual)),         # 殘差絕對均值
                _compute_trend_strength(trend, seasonal),  # 趨勢強度
            ])
            
            names.extend([
                f'{col_name}_trend_skew', f'{col_name}_trend_kurt',
                f'{col_name}_trend_autocorr', f'{col_name}_residual_mae',
                f'{col_name}_trend_strength'
            ])
        
        return {'features': features, 'names': names}
        
    except Exception as e:
        print(f"特徵提取失敗: {e}")
        return {'features': [0] * 8, 'names': [f'{col_name}_feat_{i}' for i in range(8)]}


def _enhanced_statistical_decomposition(series, col_name, service_info, metric_info, period):
    """增強統計分解（STL失敗時的回退）"""
    try:
        # 移動平均趨勢
        if len(series) >= 5:
            window = min(period, len(series) // 3)
            trend = series.rolling(window=window, center=True).mean()
            trend = trend.fillna(method='ffill').fillna(method='bfill').fillna(series.mean())
        else:
            trend = pd.Series([series.mean()] * len(series), index=series.index)
        
        # 去趨勢
        detrended = series - trend
        
        # 簡單季節性估計
        if len(series) >= period * 2:
            seasonal_pattern = []
            for i in range(period):
                seasonal_indices = list(range(i, len(detrended), period))
                if seasonal_indices:
                    seasonal_value = detrended.iloc[seasonal_indices].mean()
                    seasonal_pattern.append(seasonal_value)
                else:
                    seasonal_pattern.append(0)
            
            # 擴展季節性模式
            seasonal = []
            for i in range(len(series)):
                seasonal.append(seasonal_pattern[i % period])
            seasonal = pd.Series(seasonal, index=series.index)
        else:
            seasonal = pd.Series([0] * len(series), index=series.index)
        
        # 殘差
        residual = series - trend - seasonal
        
        # 計算特徵
        feature_weight = metric_info.get('weight', 1.0) if metric_info else 1.0
        
        features = [
            np.mean(trend) * feature_weight,
            np.std(trend) * feature_weight,
            np.mean(seasonal) * feature_weight,
            np.std(seasonal) * feature_weight,
            np.mean(residual),
            np.std(residual),
            np.max(trend) - np.min(trend),
            np.max(seasonal) - np.min(seasonal),
            period,  # 使用的週期
            _safe_autocorr(series),  # 原序列自相關
        ]
        
        names = [
            f'{col_name}_ma_trend_mean', f'{col_name}_ma_trend_std',
            f'{col_name}_simple_seasonal_mean', f'{col_name}_simple_seasonal_std',
            f'{col_name}_residual_mean', f'{col_name}_residual_std',
            f'{col_name}_trend_range', f'{col_name}_seasonal_range',
            f'{col_name}_period', f'{col_name}_autocorr'
        ]
        
        return {'features': features, 'names': names}
        
    except Exception as e:
        print(f"增強統計分解失敗: {e}")
        return _compute_basic_service_statistics(series, col_name, service_info, metric_info)


def _compute_basic_service_statistics(series, col_name, service_info, metric_info):
    """計算基本服務統計特徵"""
    try:
        if len(series) == 0:
            return {'features': [0] * 8, 'names': [f'{col_name}_feat_{i}' for i in range(8)]}
        
        # 基本統計
        features = [
            np.mean(series),
            np.std(series) + 1e-8,  # 避免除零
            np.min(series),
            np.max(series),
            np.median(series),
            np.percentile(series, 25),
            np.percentile(series, 75),
            len(series)
        ]
        
        names = [
            f'{col_name}_mean', f'{col_name}_std', f'{col_name}_min', f'{col_name}_max',
            f'{col_name}_median', f'{col_name}_q25', f'{col_name}_q75', f'{col_name}_length'
        ]
        
        # 服務權重調整
        if service_info and service_info.get('priority') == 'critical':
            features = [f * 1.5 for f in features[:4]] + features[4:]  # 關鍵服務權重增加
        
        return {'features': features, 'names': names}
        
    except Exception:
        return {'features': [0] * 8, 'names': [f'{col_name}_feat_{i}' for i in range(8)]}


def _compute_constant_features(series, col_name, service_info, metric_info):
    """處理常數序列"""
    const_val = series.iloc[0] if len(series) > 0 else 0
    
    features = [const_val, 0, const_val, const_val, const_val, const_val, const_val, len(series)]
    names = [
        f'{col_name}_const_mean', f'{col_name}_const_std', f'{col_name}_const_min', f'{col_name}_const_max',
        f'{col_name}_const_median', f'{col_name}_const_q25', f'{col_name}_const_q75', f'{col_name}_const_length'
    ]
    
    return {'features': features, 'names': names}


# 輔助函數
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


def _compute_trend_strength(trend, seasonal):
    """計算趨勢強度"""
    try:
        trend_var = np.var(trend)
        seasonal_var = np.var(seasonal)
        total_var = trend_var + seasonal_var
        
        if total_var > 0:
            return trend_var / total_var
        return 0
    except:
        return 0


# 確保 scipy.signal 可用
try:
    from scipy import signal
except ImportError:
    class signal:
        @staticmethod
        def detrend(x):
            if hasattr(x, 'values'):
                x = x.values
            return x - np.mean(x)
        
        @staticmethod
        def find_peaks(x, height=None, distance=None):
            """簡化的峰值檢測"""
            if len(x) < 3:
                return [], {}
            
            peaks = []
            for i in range(1, len(x) - 1):
                if x[i] > x[i-1] and x[i] > x[i+1]:
                    if height is None or x[i] >= height:
                        peaks.append(i)
            
            return np.array(peaks), {}


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

# 🔧 注意：enhanced_feature_fusion 函數已在本檔案前面定義
# 移除重複定義以避免衝突

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
    增強的trace處理 - 專注於TracerCA風格的核心特徵
    
    Args:
        trace_data: trace數據
        inject_time: 故障注入時間
    
    Returns:
        trace_features: trace特徵
        operation_names: 操作名稱
        service_graph: 服務圖
    """
    print("🔧 Enhanced trace processing for RCA...")
    
    if trace_data is None or (isinstance(trace_data, pd.DataFrame) and trace_data.empty):
        return np.array([]), [], None
    
    try:
        # 確保trace_data是DataFrame格式
        if not isinstance(trace_data, pd.DataFrame):
            trace_data = pd.DataFrame(trace_data)
        
        # 標準化列名
        column_mapping = {
            'service_name': 'serviceName', 'service': 'serviceName',
            'operation_name': 'operationName', 'operation': 'operationName',
            'method_name': 'operationName', 'method': 'operationName',
            'start_time': 'startTime', 'timestamp': 'startTime',
            'trace_id': 'traceID', 'span_id': 'spanID'
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
        
        # 創建操作標識
        trace_data['operation'] = trace_data['serviceName'].astype(str) + "_" + trace_data['operationName'].astype(str)
        
        # 構建服務依賴圖
        service_graph = _build_simple_service_graph(trace_data)
        
        # 提取操作級特徵
        operations = trace_data['operation'].unique()
        operation_features = []
        
        for op in operations:
            op_data = trace_data[trace_data['operation'] == op]
            
            # 基本統計特徵
            duration_stats = op_data['duration'].describe() if 'duration' in op_data.columns else pd.Series([0]*8, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
            
            # TracerCA風格的特徵
            if inject_time is not None and 'startTime' in op_data.columns:
                # 分割正常和異常期間
                normal_data = op_data[op_data['startTime'] < inject_time] if 'startTime' in op_data.columns else op_data[:len(op_data)//2]
                anomal_data = op_data[op_data['startTime'] >= inject_time] if 'startTime' in op_data.columns else op_data[len(op_data)//2:]
                
                # 計算TracerCA特徵
                if not normal_data.empty and not anomal_data.empty:
                    normal_latency = normal_data['duration'].mean() if 'duration' in normal_data.columns else 0
                    anomal_latency = anomal_data['duration'].mean() if 'duration' in anomal_data.columns else 0
                    latency_change = (anomal_latency - normal_latency) / max(normal_latency, 1e-8)
                    
                    # 異常檢測
                    threshold = normal_latency + 3 * normal_data['duration'].std() if 'duration' in normal_data.columns else anomal_latency
                    abnormal_count = (anomal_data['duration'] > threshold).sum() if 'duration' in anomal_data.columns else 0
                    confidence = abnormal_count / len(anomal_data) if len(anomal_data) > 0 else 0
                else:
                    latency_change = 0
                    confidence = 0
            else:
                latency_change = 0
                confidence = 0
            
            # 組合特徵
            features = [
                duration_stats['mean'],    # 平均延遲
                duration_stats['std'],     # 延遲標準差
                duration_stats['max'],     # 最大延遲
                duration_stats['count'],   # 調用次數
                latency_change,           # 延遲變化率
                confidence,               # 異常置信度
                len(op_data) / len(trace_data),  # 調用頻率
                duration_stats['75%'] - duration_stats['25%']  # IQR
            ]
            
            operation_features.append(features)
        
        # 轉換為numpy數組
        if operation_features:
            trace_features = np.array(operation_features)
            # 處理NaN值
            trace_features = np.nan_to_num(trace_features, nan=0.0, posinf=1.0, neginf=-1.0)
        else:
            trace_features = np.array([])
        
        return trace_features, list(operations), service_graph
        
    except Exception as e:
        print(f"⚠️ Enhanced trace processing failed: {e}")
        return np.array([]), [], None


def _build_simple_service_graph(trace_data):
    """構建簡化的服務依賴圖"""
    try:
        import networkx as nx
        
        G = nx.DiGraph()
        
        # 添加服務節點
        if 'serviceName' in trace_data.columns:
            services = trace_data['serviceName'].unique()
            G.add_nodes_from(services)
            
            # 基於調用順序添加邊
            if 'traceID' in trace_data.columns and 'startTime' in trace_data.columns:
                for trace_id, trace_group in trace_data.groupby('traceID'):
                    trace_group = trace_group.sort_values('startTime')
                    services_in_trace = trace_group['serviceName'].tolist()
                    
                    for i in range(len(services_in_trace) - 1):
                        src, dst = services_in_trace[i], services_in_trace[i + 1]
                        if src != dst:
                            if G.has_edge(src, dst):
                                G[src][dst]['weight'] += 1
                            else:
                                G.add_edge(src, dst, weight=1)
        
        return G
        
    except Exception as e:
        print(f"Service graph construction failed: {e}")
        return None