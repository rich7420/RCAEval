"""
Prometheus API client for metrics collection with time-series processing and filtering.
Implements requirements 4.1 and 6.4 for metrics data collection.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import time
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class PrometheusClient:
    """Client for collecting metrics from Prometheus with advanced processing capabilities."""
    
    def __init__(self, base_url: str = "http://localhost:9090", timeout: int = 30):
        """
        Initialize Prometheus client.
        
        Args:
            base_url: Prometheus server URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to Prometheus API."""
        url = urljoin(self.base_url, endpoint)
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Prometheus API request failed: {e}")
            raise
            
    def query_range(self, query: str, start_time: datetime, end_time: datetime, 
                   step: str = "15s") -> pd.DataFrame:
        """
        Query Prometheus for time-series data over a range.
        
        Args:
            query: PromQL query string
            start_time: Start timestamp
            end_time: End timestamp  
            step: Query resolution step
            
        Returns:
            DataFrame with timestamp, metric_name, value, labels columns
        """
        params = {
            'query': query,
            'start': start_time.timestamp(),
            'end': end_time.timestamp(),
            'step': step
        }
        
        data = self._make_request('/api/v1/query_range', params)
        
        if data['status'] != 'success':
            raise ValueError(f"Prometheus query failed: {data.get('error', 'Unknown error')}")
            
        return self._parse_range_response(data['data']['result'], query)
        
    def _parse_range_response(self, result: List[Dict], query: str) -> pd.DataFrame:
        """Parse Prometheus range query response into DataFrame."""
        rows = []
        
        for series in result:
            metric_info = series['metric']
            values = series['values']
            
            # Extract service name from labels
            service = self._extract_service_name(metric_info)
            
            # Extract metric name from query or labels
            metric_name = self._extract_metric_name(query, metric_info)
            
            # Convert labels to JSON string
            labels_json = json.dumps(metric_info, sort_keys=True)
            
            for timestamp, value in values:
                rows.append({
                    'timestamp': int(timestamp),
                    'service': service,
                    'metric_name': metric_name,
                    'value': float(value) if value != 'NaN' else np.nan,
                    'labels': labels_json
                })
                
        return pd.DataFrame(rows)
        
    def _extract_service_name(self, metric_info: Dict[str, str]) -> str:
        """Extract service name from metric labels."""
        # Try common service label names
        for label in ['service', 'job', 'container', 'pod', 'service_name']:
            if label in metric_info:
                return metric_info[label]
        
        # Fallback to instance or unknown
        return metric_info.get('instance', 'unknown')
        
    def _extract_metric_name(self, query: str, metric_info: Dict[str, str]) -> str:
        """Extract metric name from query or labels."""
        # Try to get __name__ from labels first
        if '__name__' in metric_info:
            return metric_info['__name__']
            
        # Extract from query (simple heuristic)
        # Look for metric name at start of query
        import re
        match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)', query.strip())
        if match:
            return match.group(1)
            
        return 'unknown_metric'


class MetricsProcessor:
    """Process and filter time-series metrics data."""
    
    def __init__(self, sampling_rate: str = "15s"):
        """
        Initialize metrics processor.
        
        Args:
            sampling_rate: Default sampling rate for aggregation
        """
        self.sampling_rate = sampling_rate
        
    def filter_by_time_range(self, df: pd.DataFrame, start_time: datetime, 
                           end_time: datetime) -> pd.DataFrame:
        """Filter metrics by time range."""
        if df.empty:
            return df
            
        # Convert timestamp to datetime for filtering
        df_copy = df.copy()
        df_copy['datetime'] = pd.to_datetime(df_copy['timestamp'], unit='s')
        
        mask = (df_copy['datetime'] >= start_time) & (df_copy['datetime'] <= end_time)
        return df_copy[mask].drop('datetime', axis=1)
        
    def filter_by_services(self, df: pd.DataFrame, services: List[str]) -> pd.DataFrame:
        """Filter metrics by service names."""
        if df.empty or not services:
            return df
            
        return df[df['service'].isin(services)]
        
    def aggregate_metrics(self, df: pd.DataFrame, agg_window: str = "1min", 
                         agg_func: str = "mean") -> pd.DataFrame:
        """
        Aggregate metrics over time windows.
        
        Args:
            df: Input DataFrame
            agg_window: Aggregation window (e.g., "1min", "5min")
            agg_func: Aggregation function ("mean", "max", "min", "sum")
        """
        if df.empty:
            return df
            
        df_copy = df.copy()
        df_copy['datetime'] = pd.to_datetime(df_copy['timestamp'], unit='s')
        
        # Group by service, metric_name, and time window
        grouper = pd.Grouper(key='datetime', freq=agg_window)
        
        agg_df = df_copy.groupby(['service', 'metric_name', grouper]).agg({
            'value': agg_func,
            'labels': 'first'  # Keep first labels entry
        }).reset_index()
        
        # Convert back to timestamp
        agg_df['timestamp'] = agg_df['datetime'].astype(int) // 10**9
        agg_df = agg_df.drop('datetime', axis=1)
        
        return agg_df
        
    def remove_outliers(self, df: pd.DataFrame, method: str = "iqr", 
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from metrics data.
        
        Args:
            df: Input DataFrame
            method: Outlier detection method ("iqr", "zscore")
            threshold: Threshold for outlier detection
        """
        if df.empty:
            return df
            
        df_clean = df.copy()
        
        for (service, metric), group in df_clean.groupby(['service', 'metric_name']):
            if method == "iqr":
                Q1 = group['value'].quantile(0.25)
                Q3 = group['value'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask = (group['value'] < lower_bound) | (group['value'] > upper_bound)
                
            elif method == "zscore":
                z_scores = np.abs((group['value'] - group['value'].mean()) / group['value'].std())
                outlier_mask = z_scores > threshold
                
            else:
                continue
                
            # Remove outliers
            df_clean = df_clean.drop(group[outlier_mask].index)
            
        return df_clean.reset_index(drop=True)
        
    def interpolate_missing_values(self, df: pd.DataFrame, method: str = "linear") -> pd.DataFrame:
        """
        Interpolate missing values in time series.
        
        Args:
            df: Input DataFrame
            method: Interpolation method ("linear", "forward", "backward")
        """
        if df.empty:
            return df
            
        df_interp = df.copy()
        
        for (service, metric), group in df_interp.groupby(['service', 'metric_name']):
            if method == "linear":
                df_interp.loc[group.index, 'value'] = group['value'].interpolate(method='linear')
            elif method == "forward":
                df_interp.loc[group.index, 'value'] = group['value'].fillna(method='ffill')
            elif method == "backward":
                df_interp.loc[group.index, 'value'] = group['value'].fillna(method='bfill')
                
        return df_interp


class MetricsCollector:
    """High-level metrics collector with built-in processing."""
    
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        """Initialize metrics collector."""
        self.client = PrometheusClient(prometheus_url)
        self.processor = MetricsProcessor()
        
    def collect_system_metrics(self, start_time: datetime, end_time: datetime,
                             services: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Collect comprehensive system metrics for services.
        
        Args:
            start_time: Collection start time
            end_time: Collection end time
            services: List of services to collect (None for all)
            
        Returns:
            DataFrame with processed metrics data
        """
        # Define key system metrics queries - fallback to available metrics
        queries = {
            # Try container metrics first, fallback to process metrics
            'cpu_usage_seconds': 'rate(container_cpu_usage_seconds_total[1m]) or rate(process_cpu_seconds_total[1m])',
            'memory_usage_bytes': 'container_memory_usage_bytes or process_resident_memory_bytes',
            'network_rx_bytes': 'rate(container_network_receive_bytes_total[1m])',
            'network_tx_bytes': 'rate(container_network_transmit_bytes_total[1m])',
            
            # Process-level metrics (more likely to be available)
            'process_cpu_seconds': 'rate(process_cpu_seconds_total[1m])',
            'process_memory_bytes': 'process_resident_memory_bytes',
            'process_open_fds': 'process_open_fds',
            
            # Go runtime metrics (for Go services)
            'go_memstats_alloc_bytes': 'go_memstats_alloc_bytes',
            'go_goroutines': 'go_goroutines',
            'go_gc_duration': 'rate(go_gc_duration_seconds_sum[1m])',
            
            # HTTP metrics (if available)
            'http_requests_total': 'rate(http_requests_total[1m])',
            'http_request_duration': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[1m]))',
            
            # Generic up metric
            'service_up': 'up'
        }
        
        all_metrics = []
        
        for metric_name, query in queries.items():
            try:
                logger.info(f"Collecting metric: {metric_name}")
                df = self.client.query_range(query, start_time, end_time)
                
                if not df.empty:
                    # Filter by services if specified
                    if services:
                        df = self.processor.filter_by_services(df, services)
                    
                    all_metrics.append(df)
                    
            except Exception as e:
                logger.warning(f"Failed to collect metric {metric_name}: {e}")
                continue
                
        if not all_metrics:
            logger.warning("No metrics collected")
            return pd.DataFrame()
            
        # Combine all metrics
        combined_df = pd.concat(all_metrics, ignore_index=True)
        
        # Apply processing
        combined_df = self.processor.remove_outliers(combined_df)
        combined_df = self.processor.interpolate_missing_values(combined_df)
        
        return combined_df.sort_values(['timestamp', 'service', 'metric_name'])
        
    def collect_custom_metrics(self, queries: Dict[str, str], start_time: datetime,
                             end_time: datetime) -> pd.DataFrame:
        """
        Collect custom metrics using provided PromQL queries.
        
        Args:
            queries: Dict of metric_name -> PromQL query
            start_time: Collection start time
            end_time: Collection end time
            
        Returns:
            DataFrame with collected metrics
        """
        all_metrics = []
        
        for metric_name, query in queries.items():
            try:
                df = self.client.query_range(query, start_time, end_time)
                if not df.empty:
                    all_metrics.append(df)
            except Exception as e:
                logger.warning(f"Failed to collect custom metric {metric_name}: {e}")
                
        if not all_metrics:
            return pd.DataFrame()
            
        return pd.concat(all_metrics, ignore_index=True)