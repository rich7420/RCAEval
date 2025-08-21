"""
Jaeger API client for trace collection with error and latency processing.
Implements requirements 4.3 and 6.4 for traces data collection.
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
import logging
from urllib.parse import urljoin
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class JaegerClient:
    """Client for collecting traces from Jaeger with advanced processing capabilities."""
    
    def __init__(self, base_url: str = "http://localhost:16686", timeout: int = 30):
        """
        Initialize Jaeger client.
        
        Args:
            base_url: Jaeger server URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to Jaeger API."""
        url = urljoin(self.base_url, endpoint)
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Jaeger API request failed: {e}")
            raise
            
    def get_services(self) -> List[str]:
        """Get list of available services."""
        data = self._make_request('/api/services', {})
        return data.get('data', [])
        
    def get_operations(self, service: str) -> List[str]:
        """Get operations for a service."""
        params = {'service': service}
        data = self._make_request('/api/operations', params)
        
        operations = []
        for op in data.get('data', []):
            if isinstance(op, dict):
                operations.append(op.get('operationName', ''))
            else:
                operations.append(str(op))
                
        return operations
        
    def find_traces(self, service: str, start_time: datetime, end_time: datetime,
                   operation: Optional[str] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Find traces for a service within time range.
        
        Args:
            service: Service name
            start_time: Start timestamp
            end_time: End timestamp
            operation: Optional operation name filter
            limit: Maximum number of traces
            
        Returns:
            List of trace summaries
        """
        # Convert to microseconds for Jaeger
        start_us = int(start_time.timestamp() * 1e6)
        end_us = int(end_time.timestamp() * 1e6)
        
        params = {
            'service': service,
            'start': start_us,
            'end': end_us,
            'limit': limit
        }
        
        if operation:
            params['operation'] = operation
            
        data = self._make_request('/api/traces', params)
        return data.get('data', [])
        
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed trace by ID.
        
        Args:
            trace_id: Trace ID
            
        Returns:
            Detailed trace data
        """
        endpoint = f'/api/traces/{trace_id}'
        data = self._make_request(endpoint, {})
        
        traces = data.get('data', [])
        return traces[0] if traces else None
        
    def collect_traces_range(self, services: List[str], start_time: datetime, 
                           end_time: datetime) -> pd.DataFrame:
        """
        Collect traces for multiple services over time range.
        
        Args:
            services: List of service names
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            DataFrame with trace data
        """
        all_traces = []
        
        for service in services:
            try:
                logger.info(f"Collecting traces for service: {service}")
                traces = self.find_traces(service, start_time, end_time)
                
                for trace_summary in traces:
                    # Get detailed trace data
                    trace_id = trace_summary.get('traceID')
                    if not trace_id:
                        continue
                        
                    detailed_trace = self.get_trace(trace_id)
                    if detailed_trace:
                        trace_df = self._parse_trace_to_dataframe(detailed_trace)
                        all_traces.append(trace_df)
                        
            except Exception as e:
                logger.warning(f"Failed to collect traces for service {service}: {e}")
                continue
                
        if not all_traces:
            logger.warning("No traces collected")
            return pd.DataFrame()
            
        return pd.concat(all_traces, ignore_index=True)
        
    def _parse_trace_to_dataframe(self, trace_data: Dict[str, Any]) -> pd.DataFrame:
        """Parse Jaeger trace data into DataFrame format."""
        rows = []
        
        trace_id = trace_data.get('traceID', '')
        spans = trace_data.get('spans', [])
        
        for span in spans:
            span_id = span.get('spanID', '')
            parent_span_id = span.get('references', [{}])[0].get('spanID', '') if span.get('references') else ''
            
            process = span.get('process', {})
            service_name = process.get('serviceName', 'unknown')
            
            operation_name = span.get('operationName', '')
            start_time = span.get('startTime', 0) // 1000000  # Convert to seconds
            duration_ms = span.get('duration', 0) / 1000  # Convert to milliseconds
            
            # Check for errors
            tags = span.get('tags', [])
            has_error = any(tag.get('key') == 'error' and tag.get('value') is True for tag in tags)
            
            # Extract additional information
            status_code = None
            http_method = None
            http_url = None
            
            for tag in tags:
                key = tag.get('key', '')
                value = tag.get('value')
                
                if key == 'http.status_code':
                    status_code = value
                elif key == 'http.method':
                    http_method = value
                elif key == 'http.url':
                    http_url = value
                    
            rows.append({
                'trace_id': trace_id,
                'span_id': span_id,
                'parent_span_id': parent_span_id,
                'service': service_name,
                'operation': operation_name,
                'start_time': start_time,
                'duration_ms': duration_ms,
                'error': has_error,
                'status_code': status_code,
                'http_method': http_method,
                'http_url': http_url
            })
            
        return pd.DataFrame(rows)


class TraceProcessor:
    """Process and analyze trace data for error and latency extraction."""
    
    def __init__(self):
        """Initialize trace processor."""
        self.error_patterns = [
            'error', 'exception', 'fail', 'timeout', 'abort',
            'denied', 'forbidden', 'unauthorized', 'invalid'
        ]
        
    def extract_error_traces(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract traces that contain errors.
        
        Args:
            df: DataFrame with trace data
            
        Returns:
            DataFrame with error traces
        """
        if df.empty:
            return df
            
        # Find traces with explicit error flags
        error_traces = df[df['error'] == True].copy()
        
        # Find traces with error status codes
        http_error_traces = df[
            (df['status_code'].notna()) & 
            (df['status_code'] >= 400)
        ].copy()
        
        # Find traces with error-related operations
        operation_error_traces = df[
            df['operation'].str.contains('|'.join(self.error_patterns), case=False, na=False)
        ].copy()
        
        # Combine all error traces
        all_error_traces = pd.concat([
            error_traces, http_error_traces, operation_error_traces
        ], ignore_index=True).drop_duplicates()
        
        return all_error_traces
        
    def extract_latency_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract latency metrics from traces.
        
        Args:
            df: DataFrame with trace data
            
        Returns:
            DataFrame with latency metrics
        """
        if df.empty:
            return pd.DataFrame()
            
        # Calculate latency metrics per service and operation
        latency_metrics = []
        
        for (service, operation), group in df.groupby(['service', 'operation']):
            durations = group['duration_ms'].dropna()
            
            if len(durations) == 0:
                continue
                
            metrics = {
                'service': service,
                'operation': operation,
                'count': len(durations),
                'mean_latency_ms': durations.mean(),
                'median_latency_ms': durations.median(),
                'p95_latency_ms': durations.quantile(0.95),
                'p99_latency_ms': durations.quantile(0.99),
                'max_latency_ms': durations.max(),
                'min_latency_ms': durations.min(),
                'std_latency_ms': durations.std()
            }
            
            latency_metrics.append(metrics)
            
        return pd.DataFrame(latency_metrics)
        
    def analyze_trace_dependencies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze service dependencies from traces.
        
        Args:
            df: DataFrame with trace data
            
        Returns:
            Dict with dependency analysis
        """
        if df.empty:
            return {}
            
        dependencies = defaultdict(set)
        service_calls = defaultdict(int)
        
        # Group by trace to analyze dependencies
        for trace_id, trace_group in df.groupby('trace_id'):
            trace_spans = trace_group.sort_values('start_time')
            
            # Build dependency graph for this trace
            for i, (_, span) in enumerate(trace_spans.iterrows()):
                service = span['service']
                
                # Find parent service
                parent_span_id = span['parent_span_id']
                if parent_span_id:
                    parent_spans = trace_spans[trace_spans['span_id'] == parent_span_id]
                    if not parent_spans.empty:
                        parent_service = parent_spans.iloc[0]['service']
                        if parent_service != service:
                            dependencies[parent_service].add(service)
                            service_calls[(parent_service, service)] += 1
                            
        # Convert to serializable format
        dependency_graph = {}
        for service, deps in dependencies.items():
            dependency_graph[service] = list(deps)
            
        # Calculate call frequencies
        call_frequencies = {}
        for (caller, callee), count in service_calls.items():
            call_frequencies[f"{caller} -> {callee}"] = count
            
        return {
            'dependencies': dependency_graph,
            'call_frequencies': call_frequencies,
            'total_services': len(set(df['service'])),
            'total_traces': df['trace_id'].nunique()
        }
        
    def detect_slow_traces(self, df: pd.DataFrame, percentile: float = 0.95) -> pd.DataFrame:
        """
        Detect slow traces based on duration percentiles.
        
        Args:
            df: DataFrame with trace data
            percentile: Percentile threshold for slow traces
            
        Returns:
            DataFrame with slow traces
        """
        if df.empty:
            return df
            
        # Calculate trace-level durations
        trace_durations = df.groupby('trace_id').agg({
            'duration_ms': 'sum',
            'service': lambda x: list(x.unique()),
            'start_time': 'min'
        }).reset_index()
        
        # Find slow traces
        threshold = trace_durations['duration_ms'].quantile(percentile)
        slow_traces = trace_durations[trace_durations['duration_ms'] >= threshold]
        
        # Get detailed spans for slow traces
        slow_trace_ids = slow_traces['trace_id'].tolist()
        slow_spans = df[df['trace_id'].isin(slow_trace_ids)].copy()
        
        # Add trace-level duration info
        slow_spans = slow_spans.merge(
            trace_durations[['trace_id', 'duration_ms']].rename(columns={'duration_ms': 'total_trace_duration_ms'}),
            on='trace_id'
        )
        
        return slow_spans.sort_values(['total_trace_duration_ms', 'trace_id', 'start_time'], ascending=[False, True, True])


class TraceCorrelator:
    """Correlate traces with other observability data."""
    
    def __init__(self, correlation_window: int = 60):
        """
        Initialize trace correlator.
        
        Args:
            correlation_window: Time window for correlation in seconds
        """
        self.correlation_window = correlation_window
        
    def correlate_with_logs(self, traces_df: pd.DataFrame, logs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Correlate traces with log data.
        
        Args:
            traces_df: DataFrame with trace data
            logs_df: DataFrame with log data
            
        Returns:
            DataFrame with correlation information
        """
        if traces_df.empty or logs_df.empty:
            return traces_df
            
        correlated_traces = traces_df.copy()
        correlated_traces['correlated_log_count'] = 0
        correlated_traces['correlated_error_logs'] = 0
        
        for idx, trace in traces_df.iterrows():
            trace_start = trace['start_time']
            trace_end = trace_start + (trace['duration_ms'] / 1000)
            service = trace['service']
            
            # Find logs in time window and same service
            time_mask = (
                (logs_df['timestamp'] >= trace_start - self.correlation_window) &
                (logs_df['timestamp'] <= trace_end + self.correlation_window)
            )
            service_mask = logs_df['service'] == service
            
            correlated_logs = logs_df[time_mask & service_mask]
            
            if not correlated_logs.empty:
                correlated_traces.loc[idx, 'correlated_log_count'] = len(correlated_logs)
                
                # Count error logs
                error_logs = correlated_logs[
                    correlated_logs['level'].isin(['ERROR', 'FATAL', 'CRITICAL'])
                ]
                correlated_traces.loc[idx, 'correlated_error_logs'] = len(error_logs)
                
        return correlated_traces
        
    def correlate_with_metrics(self, traces_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Correlate traces with metrics data.
        
        Args:
            traces_df: DataFrame with trace data
            metrics_df: DataFrame with metrics data
            
        Returns:
            DataFrame with correlation information
        """
        if traces_df.empty or metrics_df.empty:
            return traces_df
            
        correlated_traces = traces_df.copy()
        correlated_traces['avg_cpu_during_trace'] = np.nan
        correlated_traces['avg_memory_during_trace'] = np.nan
        
        for idx, trace in traces_df.iterrows():
            trace_start = trace['start_time']
            trace_end = trace_start + (trace['duration_ms'] / 1000)
            service = trace['service']
            
            # Find metrics in time window and same service
            time_mask = (
                (metrics_df['timestamp'] >= trace_start) &
                (metrics_df['timestamp'] <= trace_end)
            )
            service_mask = metrics_df['service'] == service
            
            correlated_metrics = metrics_df[time_mask & service_mask]
            
            if not correlated_metrics.empty:
                # Calculate average CPU usage
                cpu_metrics = correlated_metrics[
                    correlated_metrics['metric_name'].str.contains('cpu', case=False, na=False)
                ]
                if not cpu_metrics.empty:
                    correlated_traces.loc[idx, 'avg_cpu_during_trace'] = cpu_metrics['value'].mean()
                    
                # Calculate average memory usage
                memory_metrics = correlated_metrics[
                    correlated_metrics['metric_name'].str.contains('memory', case=False, na=False)
                ]
                if not memory_metrics.empty:
                    correlated_traces.loc[idx, 'avg_memory_during_trace'] = memory_metrics['value'].mean()
                    
        return correlated_traces


class TracesCollector:
    """High-level traces collector with built-in processing."""
    
    def __init__(self, jaeger_url: str = "http://localhost:16686"):
        """Initialize traces collector."""
        self.client = JaegerClient(jaeger_url)
        self.processor = TraceProcessor()
        self.correlator = TraceCorrelator()
        
    def collect_service_traces(self, start_time: datetime, end_time: datetime,
                             services: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Collect traces for specified services.
        
        Args:
            start_time: Collection start time
            end_time: Collection end time
            services: List of services to collect (None for all available)
            
        Returns:
            DataFrame with processed traces data
        """
        if not services:
            try:
                services = self.client.get_services()
                logger.info(f"Auto-discovered services: {services}")
            except Exception as e:
                logger.error(f"Failed to get services: {e}")
                return pd.DataFrame()
                
        if not services:
            logger.warning("No services found")
            return pd.DataFrame()
            
        try:
            logger.info(f"Collecting traces for services: {services}")
            df = self.client.collect_traces_range(services, start_time, end_time)
            
            if df.empty:
                logger.warning("No traces collected")
                return df
                
            return df.sort_values(['trace_id', 'start_time'])
            
        except Exception as e:
            logger.error(f"Failed to collect traces: {e}")
            return pd.DataFrame()
            
    def analyze_traces(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive trace analysis.
        
        Args:
            df: DataFrame with trace data
            
        Returns:
            Dict with analysis results
        """
        if df.empty:
            return {}
            
        analysis = {}
        
        # Basic statistics
        analysis['total_traces'] = df['trace_id'].nunique()
        analysis['total_spans'] = len(df)
        analysis['services'] = df['service'].unique().tolist()
        analysis['operations'] = df['operation'].unique().tolist()
        
        # Error analysis
        error_traces = self.processor.extract_error_traces(df)
        analysis['error_traces_count'] = error_traces['trace_id'].nunique()
        analysis['error_rate'] = analysis['error_traces_count'] / analysis['total_traces'] if analysis['total_traces'] > 0 else 0
        
        # Latency analysis
        latency_metrics = self.processor.extract_latency_metrics(df)
        analysis['latency_metrics'] = latency_metrics.to_dict('records')
        
        # Dependency analysis
        dependencies = self.processor.analyze_trace_dependencies(df)
        analysis['dependencies'] = dependencies
        
        # Slow traces
        slow_traces = self.processor.detect_slow_traces(df)
        analysis['slow_traces_count'] = slow_traces['trace_id'].nunique()
        
        return analysis
        
    def export_for_csv(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Export traces data in CSV-compatible format.
        
        Args:
            df: DataFrame with trace data
            
        Returns:
            Tuple of (traces_df, error_traces_df, latency_df)
        """
        if df.empty:
            empty_df = pd.DataFrame()
            return empty_df, empty_df, empty_df
            
        # Main traces data
        traces_df = df[['trace_id', 'span_id', 'parent_span_id', 'service', 
                       'operation', 'start_time', 'duration_ms', 'error']].copy()
        
        # Error traces
        error_traces_df = self.processor.extract_error_traces(df)
        
        # Latency metrics
        latency_df = self.processor.extract_latency_metrics(df)
        
        return traces_df, error_traces_df, latency_df