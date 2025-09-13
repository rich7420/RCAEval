"""
統一數據接口 - 標準化所有 input 格式
解決不同檔案對相同數據類型處理方式不一致的問題
"""

import numpy as np
import pandas as pd
import torch
from enum import Enum
from typing import Union, Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")


class DataType(Enum):
    """數據類型枚舉"""
    METRICS = "metrics"
    LOGS = "logs" 
    TRACES = "traces"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"


@dataclass
class StandardizedData:
    """標準化數據對象"""
    data: np.ndarray
    feature_names: List[str]
    node_names: List[str]
    data_type: DataType
    original_shape: Tuple[int, ...]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """數據驗證和清理"""
        # 確保數據是numpy數組
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data)
        
        # 處理無效值
        self.data = np.nan_to_num(self.data, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 確保特徵名稱列表長度正確
        if len(self.feature_names) != self.data.shape[1] if self.data.ndim > 1 else 1:
            self.feature_names = [f'feature_{i}' for i in range(self.data.shape[1] if self.data.ndim > 1 else 1)]
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """數據形狀"""
        return self.data.shape
    
    @property
    def num_samples(self) -> int:
        """樣本數量"""
        return self.data.shape[0] if self.data.ndim > 0 else 1
    
    @property 
    def num_features(self) -> int:
        """特徵數量"""
        return self.data.shape[1] if self.data.ndim > 1 else 1
    
    def to_tensor(self, device: str = 'cpu') -> torch.Tensor:
        """轉換為PyTorch張量 - 確保類型安全"""
        # 🔧 確保數據類型一致性，避免 numpy.float32 到 torch.FloatTensor 不匹配
        if isinstance(self.data, np.ndarray):
            # 先轉換為 float64，再轉為 torch.float32
            data_safe = self.data.astype(np.float64)
        else:
            data_safe = self.data
        return torch.tensor(data_safe, dtype=torch.float32, device=device)


class UnifiedDataInterface:
    """統一數據接口 - 標準化所有 input 格式"""
    
    @staticmethod
    def standardize_input(data: Any, 
                         data_type: Union[str, DataType] = 'auto',
                         node_names: Optional[List[str]] = None,
                         inject_time: Optional[float] = None) -> StandardizedData:
        """
        標準化輸入數據格式
        
        Args:
            data: 任意格式的輸入數據
            data_type: 數據類型 ('metrics', 'logs', 'traces', 'multimodal', 'auto')
            node_names: 節點名稱列表
            inject_time: 故障注入時間
            
        Returns:
            StandardizedData: 標準化的數據對象
        """
        # 轉換數據類型
        if isinstance(data_type, str):
            try:
                data_type = DataType(data_type)
            except ValueError:
                data_type = DataType.UNKNOWN
        
        # 自動檢測數據類型
        if data_type == DataType.UNKNOWN or data_type.value == 'auto':
            data_type = UnifiedDataInterface._detect_data_type(data)
        
        # 根據數據類型進行標準化
        if data_type == DataType.MULTIMODAL:
            return UnifiedDataInterface._standardize_multimodal(data, node_names, inject_time)
        elif data_type == DataType.METRICS:
            return UnifiedDataInterface._standardize_metrics(data, node_names)
        elif data_type == DataType.LOGS:
            return UnifiedDataInterface._standardize_logs(data, node_names)
        elif data_type == DataType.TRACES:
            return UnifiedDataInterface._standardize_traces(data, node_names, inject_time)
        else:
            return UnifiedDataInterface._standardize_generic(data, node_names, data_type)
    
    @staticmethod
    def _detect_data_type(data: Any) -> DataType:
        """自動檢測數據類型"""
        if isinstance(data, dict):
            # 檢查是否包含多模態數據的關鍵字
            keys = set(data.keys())
            multimodal_keys = {'metrics', 'logs', 'traces', 'node_names'}
            if multimodal_keys.intersection(keys):
                return DataType.MULTIMODAL
            elif any(key.lower().find('log') >= 0 for key in keys):
                return DataType.LOGS
            elif any(key.lower().find('trace') >= 0 for key in keys):
                return DataType.TRACES
            else:
                return DataType.METRICS
        elif isinstance(data, pd.DataFrame):
            # 根據列名檢測
            columns = [str(col).lower() for col in data.columns]
            if any('trace' in col or 'span' in col for col in columns):
                return DataType.TRACES
            elif any('log' in col or 'message' in col for col in columns):
                return DataType.LOGS
            else:
                return DataType.METRICS
        else:
            # 默認為指標數據
            return DataType.METRICS
    
    @staticmethod
    def _standardize_multimodal(data: Dict[str, Any], 
                               node_names: Optional[List[str]] = None,
                               inject_time: Optional[float] = None) -> StandardizedData:
        """標準化多模態數據"""
        # 提取各種模態的數據
        metrics_data = data.get('metrics', None)
        logs_data = data.get('logs', None)
        traces_data = data.get('traces', None)
        extracted_node_names = data.get('node_names', node_names or [])
        
        features_list = []
        feature_names_list = []
        
        # 處理指標數據
        if metrics_data is not None:
            metrics_std = UnifiedDataInterface._standardize_metrics(metrics_data, extracted_node_names)
            features_list.append(metrics_std.data)
            feature_names_list.extend([f'metrics_{name}' for name in metrics_std.feature_names])
        
        # 處理日誌數據
        if logs_data is not None:
            logs_std = UnifiedDataInterface._standardize_logs(logs_data, extracted_node_names)
            features_list.append(logs_std.data)
            feature_names_list.extend([f'logs_{name}' for name in logs_std.feature_names])
        
        # 處理鏈路追蹤數據
        if traces_data is not None:
            traces_std = UnifiedDataInterface._standardize_traces(traces_data, extracted_node_names, inject_time)
            features_list.append(traces_std.data)
            feature_names_list.extend([f'traces_{name}' for name in traces_std.feature_names])
        
        # 合併特徵
        if features_list:
            # 確保所有特徵矩陣具有相同的樣本數
            max_samples = max(f.shape[0] for f in features_list)
            aligned_features = []
            
            for features in features_list:
                if features.shape[0] < max_samples:
                    # 重複最後一行以匹配樣本數
                    padding = np.tile(features[-1:], (max_samples - features.shape[0], 1))
                    features = np.vstack([features, padding])
                elif features.shape[0] > max_samples:
                    # 截斷到最大樣本數
                    features = features[:max_samples]
                aligned_features.append(features)
            
            combined_features = np.hstack(aligned_features)
        else:
            # 沒有有效數據，創建默認特徵
            combined_features = np.array([[0.0]])
            feature_names_list = ['default_feature']
        
        return StandardizedData(
            data=combined_features,
            feature_names=feature_names_list,
            node_names=extracted_node_names or ['default_node'],
            data_type=DataType.MULTIMODAL,
            original_shape=combined_features.shape,
            metadata={
                'inject_time': inject_time,
                'has_metrics': metrics_data is not None,
                'has_logs': logs_data is not None,
                'has_traces': traces_data is not None
            }
        )
    
    @staticmethod
    def _standardize_metrics(data: Any, node_names: Optional[List[str]] = None) -> StandardizedData:
        """標準化指標數據"""
        # 轉換為DataFrame
        if isinstance(data, pd.DataFrame):
            df = data.select_dtypes(include=[np.number])
        elif isinstance(data, np.ndarray):
            df = pd.DataFrame(data) if data.ndim == 2 else pd.DataFrame({'metric': data})
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame({'metric': [float(data)] if np.isscalar(data) else data})
        
        # 處理缺失值
        df = df.fillna(method='ffill').fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
        
        # 提取特徵名稱和節點名稱
        feature_names = list(df.columns)
        if not node_names:
            node_names = feature_names if len(feature_names) > 0 else ['metric_node']
        
        return StandardizedData(
            data=df.values,
            feature_names=feature_names,
            node_names=node_names,
            data_type=DataType.METRICS,
            original_shape=df.shape,
            metadata={'source': 'metrics'}
        )
    
    @staticmethod
    def _standardize_logs(data: Any, node_names: Optional[List[str]] = None) -> StandardizedData:
        """標準化日誌數據"""
        # 基本日誌特徵提取
        if isinstance(data, list):
            text_data = ' '.join(str(item) for item in data)
        elif isinstance(data, str):
            text_data = data
        elif isinstance(data, pd.DataFrame):
            text_columns = [col for col in data.columns if 'message' in col.lower() or 'log' in col.lower()]
            if text_columns:
                text_data = ' '.join(data[text_columns[0]].astype(str))
            else:
                text_data = str(data.iloc[0, 0]) if not data.empty else ""
        else:
            text_data = str(data)
        
        # 簡單的文本特徵
        features = np.array([[
            len(text_data),                           # 文本長度
            len(text_data.split()),                   # 單詞數量
            text_data.lower().count('error'),         # 錯誤計數
            text_data.lower().count('warning'),       # 警告計數
            text_data.lower().count('exception'),     # 異常計數
            len(set(text_data.split()))               # 唯一單詞數
        ]])
        
        feature_names = ['text_length', 'word_count', 'error_count', 'warning_count', 'exception_count', 'unique_words']
        
        if not node_names:
            node_names = ['log_node']
        
        return StandardizedData(
            data=features,
            feature_names=feature_names,
            node_names=node_names,
            data_type=DataType.LOGS,
            original_shape=features.shape,
            metadata={'source': 'logs', 'text_length': len(text_data)}
        )
    
    @staticmethod
    def _standardize_traces(data: Any, 
                           node_names: Optional[List[str]] = None,
                           inject_time: Optional[float] = None) -> StandardizedData:
        """標準化鏈路追蹤數據"""
        if isinstance(data, pd.DataFrame) and not data.empty:
            # 提取基本追蹤特徵
            features = []
            
            # 基本統計特徵
            features.extend([
                len(data),                                    # 追蹤數量
                data.get('duration', [0]).mean(),            # 平均持續時間
                data.get('duration', [0]).std(),             # 持續時間標準差
                data.get('duration', [0]).max(),             # 最大持續時間
            ])
            
            # 服務相關特徵
            if 'serviceName' in data.columns:
                unique_services = data['serviceName'].nunique()
                features.append(unique_services)
            else:
                features.append(1)
            
            # 錯誤相關特徵
            if 'tags' in data.columns:
                error_traces = data['tags'].astype(str).str.contains('error', case=False, na=False).sum()
                features.append(error_traces)
            else:
                features.append(0)
            
            features = np.array([features])
            feature_names = ['trace_count', 'avg_duration', 'duration_std', 'max_duration', 'unique_services', 'error_traces']
            
            # 提取節點名稱
            if not node_names and 'serviceName' in data.columns:
                node_names = data['serviceName'].unique().tolist()
        else:
            # 空數據的默認特徵
            features = np.array([[0, 0, 0, 0, 0, 0]])
            feature_names = ['trace_count', 'avg_duration', 'duration_std', 'max_duration', 'unique_services', 'error_traces']
        
        if not node_names:
            node_names = ['trace_node']
        
        return StandardizedData(
            data=features,
            feature_names=feature_names,
            node_names=node_names,
            data_type=DataType.TRACES,
            original_shape=features.shape,
            metadata={'source': 'traces', 'inject_time': inject_time}
        )
    
    @staticmethod
    def _standardize_generic(data: Any, 
                            node_names: Optional[List[str]] = None,
                            data_type: DataType = DataType.UNKNOWN) -> StandardizedData:
        """標準化通用數據"""
        # 嘗試轉換為數值數組
        if isinstance(data, (list, tuple)):
            array_data = np.array(data, dtype=float)
        elif isinstance(data, np.ndarray):
            array_data = data.astype(float)
        elif isinstance(data, pd.DataFrame):
            array_data = data.select_dtypes(include=[np.number]).values
        elif np.isscalar(data):
            array_data = np.array([[float(data)]])
        else:
            # 最後回退
            array_data = np.array([[0.0]])
        
        # 確保是二維數組
        if array_data.ndim == 1:
            array_data = array_data.reshape(1, -1)
        elif array_data.ndim == 0:
            array_data = array_data.reshape(1, 1)
        
        feature_names = [f'feature_{i}' for i in range(array_data.shape[1])]
        
        if not node_names:
            node_names = [f'node_{i}' for i in range(array_data.shape[0])]
        
        return StandardizedData(
            data=array_data,
            feature_names=feature_names,
            node_names=node_names,
            data_type=data_type,
            original_shape=array_data.shape,
            metadata={'source': 'generic'}
        )
    
    @staticmethod
    def get_dimensions(data: Any) -> Dict[str, int]:
        """獲取數據維度信息"""
        if isinstance(data, np.ndarray):
            return {
                'samples': data.shape[0] if data.ndim > 0 else 1,
                'features': data.shape[1] if data.ndim > 1 else 1,
                'ndim': data.ndim
            }
        elif isinstance(data, pd.DataFrame):
            return {
                'samples': len(data),
                'features': len(data.columns),
                'ndim': 2
            }
        elif isinstance(data, dict):
            return {
                'keys': len(data),
                'ndim': 1
            }
        else:
            return {
                'samples': 1,
                'features': 1,
                'ndim': 0
            }
    
    @staticmethod
    def validate_data(data: Any) -> Dict[str, Any]:
        """驗證數據完整性"""
        validation_result = {
            'is_valid': True,
            'issues': [],
            'suggestions': []
        }
        
        try:
            # 基本類型檢查
            if data is None:
                validation_result['is_valid'] = False
                validation_result['issues'].append("數據為空")
                return validation_result
            
            # 數值數據檢查
            if isinstance(data, (np.ndarray, pd.DataFrame)):
                if isinstance(data, pd.DataFrame):
                    numeric_data = data.select_dtypes(include=[np.number])
                    if numeric_data.empty:
                        validation_result['issues'].append("DataFrame不包含數值數據")
                        validation_result['suggestions'].append("確保DataFrame包含數值列")
                    data_to_check = numeric_data.values
                else:
                    data_to_check = data
                
                if data_to_check.size > 0:
                    # 檢查無效值
                    nan_count = np.isnan(data_to_check).sum()
                    inf_count = np.isinf(data_to_check).sum()
                    
                    if nan_count > 0:
                        validation_result['issues'].append(f"包含 {nan_count} 個 NaN 值")
                        validation_result['suggestions'].append("使用 fillna() 處理缺失值")
                    
                    if inf_count > 0:
                        validation_result['issues'].append(f"包含 {inf_count} 個無窮大值")
                        validation_result['suggestions'].append("使用 replace([np.inf, -np.inf], value) 處理無窮大值")
            
            # 如果有問題但不是致命的，仍然認為數據有效
            if validation_result['issues'] and len(validation_result['issues']) < 3:
                validation_result['suggestions'].append("數據可以使用但建議清理")
            elif len(validation_result['issues']) >= 3:
                validation_result['is_valid'] = False
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"驗證過程中出錯: {str(e)}")
        
        return validation_result 