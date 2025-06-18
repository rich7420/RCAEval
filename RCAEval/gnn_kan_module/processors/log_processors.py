"""
統一日誌處理器 - 合併重複的日誌特徵提取函數
解決 feature_extractors.py 和 kan_components/feature_extraction.py 中的重複定義
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

# 導入統一接口
from ..core.base_classes import BaseFeatureProcessor
from ..core.data_interface import StandardizedData, DataType

warnings.filterwarnings("ignore")


class UnifiedLogProcessor(BaseFeatureProcessor):
    """
    統一日誌處理器 - 合併原有的重複實現
    結合簡化版本和完整版本的優勢
    """
    
    def __init__(self, target_dim: int = 64, method: str = 'auto', 
                 max_features: int = 1000, use_tfidf: bool = True, **kwargs):
        """
        初始化統一日誌處理器
        
        Args:
            target_dim: 目標特徵維度
            method: 處理方法 ('simple', 'tfidf', 'dla', 'auto')
            max_features: TF-IDF最大特徵數
            use_tfidf: 是否使用TF-IDF（複雜版本）
            **kwargs: 其他參數
        """
        super().__init__(target_dim, method, **kwargs)
        self.max_features = max_features
        self.use_tfidf = use_tfidf
        self.vectorizer = None
        self.scaler = StandardScaler()
        
        # 根據方法自動選擇策略
        if method == 'auto':
            self.method = 'tfidf' if use_tfidf else 'simple'
    
    def fit(self, data: Union[StandardizedData, Any], **kwargs) -> 'UnifiedLogProcessor':
        """訓練日誌處理器"""
        # 標準化數據
        if not isinstance(data, StandardizedData):
            from ..core.data_interface import UnifiedDataInterface
            data = UnifiedDataInterface.standardize_input(data, data_type='logs')
        
        # 提取文本數據
        log_texts = self._extract_log_texts(data)
        
        # 根據方法訓練
        if self.method == 'tfidf' and len(log_texts) > 0:
            self._fit_tfidf(log_texts)
        
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[StandardizedData, Any]) -> Tuple[np.ndarray, List[str]]:
        """轉換日誌數據為特徵"""
        # 標準化數據
        if not isinstance(data, StandardizedData):
            from ..core.data_interface import UnifiedDataInterface
            data = UnifiedDataInterface.standardize_input(data, data_type='logs')
        
        # 提取文本數據
        log_texts = self._extract_log_texts(data)
        
        # 根據方法提取特徵
        if self.method == 'simple':
            return self._extract_simple_features(log_texts)
        elif self.method == 'tfidf':
            return self._extract_tfidf_features(log_texts)
        elif self.method == 'dla':
            return self._extract_dla_features(log_texts)
        else:
            # 自動選擇
            if len(log_texts) > 10 and self.use_tfidf:
                return self._extract_tfidf_features(log_texts)
            else:
                return self._extract_simple_features(log_texts)
    
    def _extract_log_texts(self, data: StandardizedData) -> List[str]:
        """從標準化數據中提取日誌文本"""
        if hasattr(data, 'metadata') and 'original_data' in data.metadata:
            original_data = data.metadata['original_data']
        else:
            original_data = data.data
        
        log_texts = []
        
        if isinstance(original_data, pd.DataFrame):
            # 從DataFrame中提取文本列
            for col in original_data.columns:
                if original_data[col].dtype == 'object':
                    log_texts.extend(original_data[col].dropna().astype(str).tolist())
        elif isinstance(original_data, (list, tuple)):
            log_texts = [str(item) for item in original_data]
        elif isinstance(original_data, str):
            log_texts = [original_data]
        else:
            log_texts = [str(original_data)]
        
        # 清理空文本
        log_texts = [text for text in log_texts if text.strip()]
        
        return log_texts
    
    def _extract_simple_features(self, log_texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        簡化日誌特徵提取 - 來自 feature_extractors.py
        快速且穩定，適合小量數據
        """
        if not log_texts:
            features = np.array([[0, 0, 0, 0, 0, 0]])
            feature_names = ['text_length', 'word_count', 'error_count', 'warning_count', 'exception_count', 'unique_words']
            return features, feature_names
        
        # 合併所有文本進行整體分析
        combined_text = ' '.join(log_texts)
        
        # 基本統計特徵
        features = [
            len(combined_text),                        # 文本總長度
            len(combined_text.split()),                # 總詞數
            combined_text.lower().count('error'),      # 錯誤計數
            combined_text.lower().count('warning'),    # 警告計數
            combined_text.lower().count('exception'),  # 異常計數
            len(set(combined_text.split())),          # 唯一詞數
            combined_text.lower().count('info'),       # 信息計數
            combined_text.lower().count('debug'),      # 調試計數
            len(log_texts),                           # 日誌條目數
            np.mean([len(text) for text in log_texts]) # 平均日誌長度
        ]
        
        feature_names = [
            'text_length', 'word_count', 'error_count', 'warning_count', 
            'exception_count', 'unique_words', 'info_count', 'debug_count',
            'log_entry_count', 'avg_log_length'
        ]
        
        return np.array([features]), feature_names
    
    def _extract_tfidf_features(self, log_texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        TF-IDF日誌特徵提取 - 來自 kan_components/feature_extraction.py
        複雜但準確，適合大量數據
        """
        if not log_texts:
            return self._extract_simple_features(log_texts)
        
        try:
            # 如果沒有訓練過vectorizer，使用當前數據訓練
            if self.vectorizer is None:
                self._fit_tfidf(log_texts)
            
            # TF-IDF特徵提取
            tfidf_features = self.vectorizer.transform(log_texts).toarray()
            feature_names = self.vectorizer.get_feature_names_out().tolist()
            
            # 聚合多個日誌條目的特徵（如果有多條）
            if tfidf_features.shape[0] > 1:
                # 使用統計聚合
                aggregated_features = np.array([
                    np.mean(tfidf_features, axis=0),    # 平均值
                    np.max(tfidf_features, axis=0),     # 最大值
                    np.std(tfidf_features, axis=0),     # 標準差
                ]).flatten()
                
                # 生成聚合特徵名稱
                agg_feature_names = []
                for stat in ['mean', 'max', 'std']:
                    agg_feature_names.extend([f'{stat}_{name}' for name in feature_names])
                
                features = aggregated_features.reshape(1, -1)
                feature_names = agg_feature_names
            else:
                features = tfidf_features
            
            return features, feature_names
            
        except Exception as e:
            print(f"⚠️ TF-IDF提取失敗: {e}，回退到簡化特徵")
            return self._extract_simple_features(log_texts)
    
    def _extract_dla_features(self, log_texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        深度日誌分析特徵提取 - 簡化版
        結合了字符級和語義級特徵
        """
        if not log_texts:
            return self._extract_simple_features(log_texts)
        
        try:
            all_features = []
            
            for text in log_texts:
                # 字符級特徵
                char_features = [
                    len(text),                          # 文本長度
                    text.count(' '),                    # 空格數
                    text.count('\n'),                   # 換行數
                    len(set(text)),                     # 唯一字符數
                    text.count('.'),                    # 句號數
                    text.count(','),                    # 逗號數
                ]
                
                # 關鍵字特徵
                keyword_features = [
                    text.upper().count('ERROR'),        # 錯誤
                    text.upper().count('WARN'),         # 警告
                    text.upper().count('INFO'),         # 信息
                    text.upper().count('DEBUG'),        # 調試
                    text.upper().count('EXCEPTION'),    # 異常
                    text.upper().count('TIMEOUT'),      # 超時
                    text.upper().count('CONNECTION'),   # 連接
                    text.upper().count('FAILED'),       # 失敗
                ]
                
                # 結構特徵
                struct_features = [
                    text.count('['),                    # 方括號
                    text.count('{'),                    # 大括號
                    text.count(':'),                    # 冒號
                    text.count('='),                    # 等號
                ]
                
                combined_features = char_features + keyword_features + struct_features
                all_features.append(combined_features)
            
            # 轉換為numpy數組
            features_matrix = np.array(all_features)
            
            # 如果有多條日誌，聚合特徵
            if features_matrix.shape[0] > 1:
                aggregated = np.array([
                    np.mean(features_matrix, axis=0),
                    np.max(features_matrix, axis=0),
                    np.min(features_matrix, axis=0),
                    np.std(features_matrix, axis=0)
                ]).flatten()
                features = aggregated.reshape(1, -1)
            else:
                features = features_matrix
            
            # 生成特徵名稱
            base_names = [
                'text_length', 'space_count', 'newline_count', 'unique_chars',
                'period_count', 'comma_count', 'error_kw', 'warn_kw', 'info_kw',
                'debug_kw', 'exception_kw', 'timeout_kw', 'connection_kw', 'failed_kw',
                'bracket_count', 'brace_count', 'colon_count', 'equal_count'
            ]
            
            if features_matrix.shape[0] > 1:
                feature_names = []
                for stat in ['mean', 'max', 'min', 'std']:
                    feature_names.extend([f'{stat}_{name}' for name in base_names])
            else:
                feature_names = base_names
            
            return features, feature_names
            
        except Exception as e:
            print(f"⚠️ DLA特徵提取失敗: {e}，回退到簡化特徵")
            return self._extract_simple_features(log_texts)
    
    def _fit_tfidf(self, log_texts: List[str]):
        """訓練TF-IDF向量化器"""
        try:
            max_df = min(0.95, len(log_texts) - 1) if len(log_texts) > 1 else 1.0
            self.vectorizer = TfidfVectorizer(
                max_features=min(self.max_features, len(log_texts) * 10),
                ngram_range=(1, 2),
                stop_words='english' if all(isinstance(x, str) for x in log_texts) else None,
                lowercase=True,
                token_pattern=r'\b\w+\b',
                min_df=1,
                max_df=max_df
            )
            self.vectorizer.fit(log_texts)
        except Exception as e:
            print(f"⚠️ TF-IDF訓練失敗: {e}")
            self.vectorizer = None


# 向後兼容函數
def extract_log_features(log_data, use_dla=False, max_features=1000, method='auto', target_dim=64):
    """
    向後兼容的日誌特徵提取函數
    統一了原有的兩個不同實現
    
    Args:
        log_data: 日誌數據
        use_dla: 是否使用深度日誌分析
        max_features: 最大特徵數
        method: 處理方法
        target_dim: 目標維度
        
    Returns:
        features: 特徵矩陣
        feature_names: 特徵名稱列表
    """
    # 根據參數選擇方法
    if use_dla:
        method = 'dla'
    elif method == 'auto':
        # 根據數據量自動選擇
        if hasattr(log_data, '__len__') and len(log_data) > 10:
            method = 'tfidf'
        else:
            method = 'simple'
    
    # 創建處理器並處理
    processor = UnifiedLogProcessor(
        target_dim=target_dim,
        method=method,
        max_features=max_features,
        use_tfidf=(method == 'tfidf')
    )
    
    return processor.process(log_data) 