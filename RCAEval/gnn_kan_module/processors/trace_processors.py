"""
統一鏈路追蹤處理器
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Union

from ..core.base_classes import BaseFeatureProcessor
from ..core.data_interface import StandardizedData


class UnifiedTraceProcessor(BaseFeatureProcessor):
    """統一鏈路追蹤處理器 - 包含完整的TracerCA風格特徵提取邏輯"""
    
    def __init__(self, target_dim: int = 64, method: str = 'tracer_ca', **kwargs):
        super().__init__(target_dim, method, **kwargs)

    def fit(self, data: Union[StandardizedData, Any], **kwargs) -> 'UnifiedTraceProcessor':
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[StandardizedData, Any], **kwargs) -> Tuple[np.ndarray, List[str]]:
        """轉換鏈路追蹤數據為特徵"""
        inject_time = kwargs.get('inject_time')
        
        # 標準化輸入數據
        if not isinstance(data, StandardizedData):
            from ..core.data_interface import UnifiedDataInterface
            data_interface = UnifiedDataInterface(data_type='traces')
            standardized_data = data_interface.standardize(data)
            trace_df = standardized_data.data
        elif isinstance(data.data, pd.DataFrame):
            trace_df = data.data
        else:
            trace_df = pd.DataFrame(data.data)

        features, names, _ = self.enhanced_trace_processing(trace_df, inject_time)
        return features, names

    def enhanced_trace_processing(self, trace_data, inject_time=None):
        """
        增強的TracerCA風格trace處理 - 專注於最有效的特徵
        """
        if trace_data is None or trace_data.empty:
            return np.array([]), [], None
        
        # ... (Implementation from feature_processing.py)
        # Standardize column names
        column_mapping = {
            'service_name': 'serviceName', 'service': 'serviceName',
            'operation_name': 'operationName', 'operation': 'operationName',
            'method_name': 'operationName', 'method': 'operationName',
            'start_time': 'startTime', 'timestamp': 'startTime',
            'time': 'startTime', 'trace_id': 'traceID', 'span_id': 'spanID'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in trace_data.columns and new_col not in trace_data.columns:
                trace_data.rename(columns={old_col: new_col}, inplace=True)
        
        if 'serviceName' not in trace_data.columns: trace_data['serviceName'] = 'default_service'
        if 'operationName' not in trace_data.columns: trace_data['operationName'] = 'default_operation'
        if 'duration' not in trace_data.columns: trace_data['duration'] = np.random.lognormal(2, 1, len(trace_data))
        
        trace_data['operation'] = trace_data['serviceName'].astype(str) + "_" + trace_data['operationName'].astype(str)
        service_graph = self._build_enhanced_service_graph(trace_data)
        
        operations = trace_data['operation'].unique()
        operation_features = []
        
        for op in operations:
            op_data = trace_data[trace_data['operation'] == op]
            duration_stats = op_data['duration'].describe()
            
            support, confidence, ji, latency_change, call_rate_change = 0, 0, 0, 0, 0
            if inject_time and 'startTime' in op_data.columns:
                normal_data = op_data[op_data['startTime'] < inject_time]
                anomal_data = op_data[op_data['startTime'] >= inject_time]
                
                if not normal_data.empty and not anomal_data.empty:
                    normal_latency, normal_std = normal_data['duration'].mean(), normal_data['duration'].std()
                    anomal_latency = anomal_data['duration'].mean()
                    threshold = normal_latency + 3 * normal_std
                    
                    abnormal_spans = (anomal_data['duration'] > threshold).sum()
                    support = abnormal_spans / max(len(anomal_data), 1)
                    confidence = abnormal_spans / max(len(op_data), 1)
                    ji = (2 * support * confidence) / max(support + confidence, 1e-10)
                    latency_change = (anomal_latency - normal_latency) / max(normal_latency, 1e-8)
                    call_rate_change = (len(anomal_data) - len(normal_data)) / len(op_data)
            
            features = [
                support, confidence, ji, latency_change, call_rate_change,
                duration_stats.get('mean', 0), duration_stats.get('std', 0),
                duration_stats.get('max', 0), duration_stats.get('count', 0),
                duration_stats.get('75%', 0) - duration_stats.get('25%', 0)
            ]
            operation_features.append(features)
        
        trace_features = np.nan_to_num(np.array(operation_features)) if operation_features else np.array([])
        return trace_features, list(operations), service_graph

    def _build_enhanced_service_graph(self, trace_data):
        """構建增強的服務依賴圖"""
        try:
            import networkx as nx
            G = nx.DiGraph()
            if 'serviceName' not in trace_data.columns: return G
            
            for service in trace_data['serviceName'].unique():
                G.add_node(service)
            
            if 'traceID' in trace_data.columns and 'startTime' in trace_data.columns:
                for _, trace_spans in trace_data.groupby('traceID'):
                    sorted_spans = trace_spans.sort_values('startTime')
                    for i in range(len(sorted_spans) - 1):
                        u, v = sorted_spans.iloc[i]['serviceName'], sorted_spans.iloc[i+1]['serviceName']
                        if u != v:
                            if G.has_edge(u, v):
                                G[u][v]['weight'] += 1
                            else:
                                G.add_edge(u, v, weight=1)
            return G
        except ImportError:
            return None

def extract_trace_features(trace_data, inject_time=None, target_dim=64, **kwargs):
    """向後兼容的鏈路追蹤特徵提取函數 - 重定向到 UnifiedTraceProcessor"""
    processor = UnifiedTraceProcessor(target_dim=target_dim)
    return processor.process(trace_data, inject_time=inject_time) 