"""
Unified data interface - standardize all input formats
Solve the problem of inconsistent handling of the same data types across different files
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
    """Data type enumeration"""
    METRICS = "metrics"
    LOGS = "logs" 
    TRACES = "traces"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"


@dataclass
class StandardizedData:
    """Standardized data object"""
    data: np.ndarray
    feature_names: List[str]
    node_names: List[str]
    data_type: DataType
    original_shape: Tuple[int, ...]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Data validation and cleanup"""
        # Ensure data is numpy array
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data)
        
        # Handle invalid values
        self.data = np.nan_to_num(self.data, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Ensure feature names list length is correct
        if len(self.feature_names) != self.data.shape[1] if self.data.ndim > 1 else 1:
            self.feature_names = [f'feature_{i}' for i in range(self.data.shape[1] if self.data.ndim > 1 else 1)]
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Data shape"""
        return self.data.shape
    
    @property
    def num_samples(self) -> int:
        """Number of samples"""
        return self.data.shape[0] if self.data.ndim > 0 else 1
    
    @property 
    def num_features(self) -> int:
        """Number of features"""
        return self.data.shape[1] if self.data.ndim > 1 else 1
    
    def to_tensor(self, device: str = 'cpu') -> torch.Tensor:
        """Convert to PyTorch tensor - ensure type safety"""
        # Ensure data type consistency, avoid numpy.float32 to torch.FloatTensor mismatch
        if isinstance(self.data, np.ndarray):
            # First convert to float64, then to torch.float32
            data_safe = self.data.astype(np.float64)
        else:
            data_safe = self.data
        return torch.tensor(data_safe, dtype=torch.float32, device=device)


class UnifiedDataInterface:
    """Unified data interface - standardize all input formats"""
    
    @staticmethod
    def standardize_input(data: Any, 
                         data_type: Union[str, DataType] = 'auto',
                         node_names: Optional[List[str]] = None,
                         inject_time: Optional[float] = None) -> StandardizedData:
        """
        Standardize input data format
        
        Args:
            data: Input data in any format
            data_type: Data type ('metrics', 'logs', 'traces', 'multimodal', 'auto')
            node_names: List of node names
            inject_time: Fault injection time
            
        Returns:
            StandardizedData: Standardized data object
        """
        # Convert data type
        if isinstance(data_type, str):
            try:
                data_type = DataType(data_type)
            except ValueError:
                data_type = DataType.UNKNOWN
        
        # Auto-detect data type
        if data_type == DataType.UNKNOWN or data_type.value == 'auto':
            data_type = UnifiedDataInterface._detect_data_type(data)
        
        # Standardize based on data type
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
        """Auto-detect data type"""
        if isinstance(data, dict):
            # Check if contains keywords for multimodal data
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
            # Detect based on column names
            columns = [str(col).lower() for col in data.columns]
            if any('trace' in col or 'span' in col for col in columns):
                return DataType.TRACES
            elif any('log' in col or 'message' in col for col in columns):
                return DataType.LOGS
            else:
                return DataType.METRICS
        else:
            # Default to metrics data
            return DataType.METRICS
    
    @staticmethod
    def _standardize_multimodal(data: Dict[str, Any], 
                               node_names: Optional[List[str]] = None,
                               inject_time: Optional[float] = None) -> StandardizedData:
        """Standardize multimodal data"""
        # Extract data from various modalities
        metrics_data = data.get('metrics', None)
        logs_data = data.get('logs', None)
        traces_data = data.get('traces', None)
        extracted_node_names = data.get('node_names', node_names or [])
        
        features_list = []
        feature_names_list = []
        
        # Process metrics data
        if metrics_data is not None:
            metrics_std = UnifiedDataInterface._standardize_metrics(metrics_data, extracted_node_names)
            features_list.append(metrics_std.data)
            feature_names_list.extend([f'metrics_{name}' for name in metrics_std.feature_names])
        
        # Process logs data
        if logs_data is not None:
            logs_std = UnifiedDataInterface._standardize_logs(logs_data, extracted_node_names)
            features_list.append(logs_std.data)
            feature_names_list.extend([f'logs_{name}' for name in logs_std.feature_names])
        
        # Process traces data
        if traces_data is not None:
            traces_std = UnifiedDataInterface._standardize_traces(traces_data, extracted_node_names, inject_time)
            features_list.append(traces_std.data)
            feature_names_list.extend([f'traces_{name}' for name in traces_std.feature_names])
        
        # Merge features
        if features_list:
            # Ensure all feature matrices have the same number of samples
            max_samples = max(f.shape[0] for f in features_list)
            aligned_features = []
            
            for features in features_list:
                if features.shape[0] < max_samples:
                    # Repeat last row to match number of samples
                    padding = np.tile(features[-1:], (max_samples - features.shape[0], 1))
                    features = np.vstack([features, padding])
                elif features.shape[0] > max_samples:
                    # Truncate to maximum number of samples
                    features = features[:max_samples]
                aligned_features.append(features)
            
            combined_features = np.hstack(aligned_features)
        else:
            # No valid data, create default features
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
        """Standardize metrics data"""
        # Convert to DataFrame
        if isinstance(data, pd.DataFrame):
            df = data.select_dtypes(include=[np.number])
        elif isinstance(data, np.ndarray):
            df = pd.DataFrame(data) if data.ndim == 2 else pd.DataFrame({'metric': data})
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame({'metric': [float(data)] if np.isscalar(data) else data})
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
        
        # Extract feature names and node names
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
        """Standardize logs data"""
        # Basic log feature extraction
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
        
        # Simple text features
        features = np.array([[
            len(text_data),                           # Text length
            len(text_data.split()),                   # Word count
            text_data.lower().count('error'),         # Error count
            text_data.lower().count('warning'),       # Warning count
            text_data.lower().count('exception'),     # Exception count
            len(set(text_data.split()))               # Unique word count
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
        """Standardize traces data"""
        if isinstance(data, pd.DataFrame) and not data.empty:
            # Extract basic trace features
            features = []
            
            # Basic statistical features
            features.extend([
                len(data),                                    # Number of traces
                data.get('duration', [0]).mean(),            # Average duration
                data.get('duration', [0]).std(),             # Duration standard deviation
                data.get('duration', [0]).max(),             # Maximum duration
            ])
            
            # Service-related features
            if 'serviceName' in data.columns:
                unique_services = data['serviceName'].nunique()
                features.append(unique_services)
            else:
                features.append(1)
            
            # Error-related features
            if 'tags' in data.columns:
                error_traces = data['tags'].astype(str).str.contains('error', case=False, na=False).sum()
                features.append(error_traces)
            else:
                features.append(0)
            
            features = np.array([features])
            feature_names = ['trace_count', 'avg_duration', 'duration_std', 'max_duration', 'unique_services', 'error_traces']
            
            # Extract node names
            if not node_names and 'serviceName' in data.columns:
                node_names = data['serviceName'].unique().tolist()
        else:
            # Default features for empty data
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
        """Standardize generic data"""
        # Try to convert to numeric array
        if isinstance(data, (list, tuple)):
            array_data = np.array(data, dtype=float)
        elif isinstance(data, np.ndarray):
            array_data = data.astype(float)
        elif isinstance(data, pd.DataFrame):
            array_data = data.select_dtypes(include=[np.number]).values
        elif np.isscalar(data):
            array_data = np.array([[float(data)]])
        else:
            # Final fallback
            array_data = np.array([[0.0]])
        
        # Ensure it's a 2D array
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
        """Get data dimension information"""
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
        """Validate data integrity"""
        validation_result = {
            'is_valid': True,
            'issues': [],
            'suggestions': []
        }
        
        try:
            # Basic type check
            if data is None:
                validation_result['is_valid'] = False
                validation_result['issues'].append("Data is empty")
                return validation_result
            
            # Numeric data check
            if isinstance(data, (np.ndarray, pd.DataFrame)):
                if isinstance(data, pd.DataFrame):
                    numeric_data = data.select_dtypes(include=[np.number])
                    if numeric_data.empty:
                        validation_result['issues'].append("DataFrame does not contain numeric data")
                        validation_result['suggestions'].append("Ensure DataFrame contains numeric columns")
                    data_to_check = numeric_data.values
                else:
                    data_to_check = data
                
                if data_to_check.size > 0:
                    # Check for invalid values
                    nan_count = np.isnan(data_to_check).sum()
                    inf_count = np.isinf(data_to_check).sum()
                    
                    if nan_count > 0:
                        validation_result['issues'].append(f"Contains {nan_count} NaN values")
                        validation_result['suggestions'].append("Use fillna() to handle missing values")
                    
                    if inf_count > 0:
                        validation_result['issues'].append(f"Contains {inf_count} infinite values")
                        validation_result['suggestions'].append("Use replace([np.inf, -np.inf], value) to handle infinite values")
            
            # If there are issues but not fatal, still consider data valid
            if validation_result['issues'] and len(validation_result['issues']) < 3:
                validation_result['suggestions'].append("Data is usable but cleaning is recommended")
            elif len(validation_result['issues']) >= 3:
                validation_result['is_valid'] = False
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"Error during validation: {str(e)}")
        
        return validation_result 