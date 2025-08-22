"""
Traces CSV exporter for RE2-compatible format.
Implements requirement 4.3 for traces data export.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
import os

logger = logging.getLogger(__name__)


class TracesCSVExporter:
    """Export traces data to CSV format compatible with RE2 datasets."""
    
    def __init__(self):
        """Initialize traces CSV exporter."""
        self.required_columns = ['trace_id', 'span_id', 'service', 'operation', 'start_time', 'duration_ms']
        
    def export_traces_csv(self, traces_df: pd.DataFrame, output_path: str) -> bool:
        """
        Export traces DataFrame to traces.csv file.
        
        Args:
            traces_df: DataFrame with traces data
            output_path: Path to output traces.csv file
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            if traces_df.empty:
                logger.warning("Empty traces DataFrame provided")
                # Create empty CSV with headers
                empty_df = pd.DataFrame(columns=['trace_id', 'span_id', 'parent_span_id', 'service', 'operation', 'start_time', 'duration_ms', 'error'])
                empty_df.to_csv(output_path, index=False)
                return True
                
            # Validate required columns
            missing_cols = [col for col in self.required_columns if col not in traces_df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
                
            # Transform to RE2 traces.csv format
            traces_csv_df = self._transform_to_traces_csv_format(traces_df)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Export to CSV
            traces_csv_df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(traces_csv_df)} trace records to {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to export traces to CSV: {e}")
            return False
            
    def export_tracets_err_csv(self, traces_df: pd.DataFrame, output_path: str) -> bool:
        """
        Export traces DataFrame to tracets_err.csv file (error time series).
        
        Args:
            traces_df: DataFrame with traces data
            output_path: Path to output tracets_err.csv file
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            if traces_df.empty:
                logger.warning("Empty traces DataFrame provided")
                # Create empty CSV with headers
                empty_df = pd.DataFrame(columns=['time', 'service', 'error_count', 'total_traces', 'error_rate'])
                empty_df.to_csv(output_path, index=False)
                return True
                
            # Transform to RE2 tracets_err.csv format
            tracets_err_df = self._transform_to_tracets_err_format(traces_df)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Export to CSV
            tracets_err_df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(tracets_err_df)} trace error time series records to {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to export tracets_err to CSV: {e}")
            return False
            
    def export_tracets_lat_csv(self, traces_df: pd.DataFrame, output_path: str) -> bool:
        """
        Export traces DataFrame to tracets_lat.csv file (latency time series).
        
        Args:
            traces_df: DataFrame with traces data
            output_path: Path to output tracets_lat.csv file
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            if traces_df.empty:
                logger.warning("Empty traces DataFrame provided")
                # Create empty CSV with headers
                empty_df = pd.DataFrame(columns=['time', 'service', 'avg_latency_ms', 'p95_latency_ms', 'p99_latency_ms', 'max_latency_ms'])
                empty_df.to_csv(output_path, index=False)
                return True
                
            # Transform to RE2 tracets_lat.csv format
            tracets_lat_df = self._transform_to_tracets_lat_format(traces_df)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Export to CSV
            tracets_lat_df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(tracets_lat_df)} trace latency time series records to {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to export tracets_lat to CSV: {e}")
            return False
            
    def _transform_to_traces_csv_format(self, traces_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform traces DataFrame to RE2 traces.csv format.
        
        Expected columns: trace_id, span_id, parent_span_id, service, operation, start_time, duration_ms, error
        """
        traces_csv = traces_df.copy()
        
        # Ensure required columns exist
        required_cols = ['trace_id', 'span_id', 'parent_span_id', 'service', 'operation', 'start_time', 'duration_ms', 'error']
        
        for col in required_cols:
            if col not in traces_csv.columns:
                if col == 'parent_span_id':
                    traces_csv[col] = ''  # Empty string for root spans
                elif col == 'error':
                    traces_csv[col] = False  # Default to no error
                else:
                    traces_csv[col] = ''
                    
        # Format start_time as ISO string if it's numeric
        if traces_csv['start_time'].dtype in ['int64', 'float64']:
            traces_csv['start_time'] = pd.to_datetime(traces_csv['start_time'], unit='s').dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
        
        # Ensure duration_ms is numeric
        traces_csv['duration_ms'] = pd.to_numeric(traces_csv['duration_ms'], errors='coerce').fillna(0)
        
        # Select and order columns for RE2 format
        traces_csv = traces_csv[required_cols]
        
        # Sort by trace_id and start_time
        traces_csv = traces_csv.sort_values(['trace_id', 'start_time'])
        
        return traces_csv
        
    def _transform_to_tracets_err_format(self, traces_df: pd.DataFrame, 
                                       time_window: str = '1min') -> pd.DataFrame:
        """
        Transform traces DataFrame to RE2 tracets_err.csv format (error time series).
        
        Args:
            traces_df: DataFrame with traces data
            time_window: Time window for aggregation
            
        Returns:
            DataFrame in tracets_err.csv format
        """
        if traces_df.empty:
            return pd.DataFrame()
            
        # Convert start_time to datetime for grouping
        traces_df = traces_df.copy()
        
        # Handle different start_time formats
        if traces_df['start_time'].dtype == 'object':
            # Try to parse ISO format
            try:
                traces_df['datetime'] = pd.to_datetime(traces_df['start_time'])
            except:
                # Fallback to numeric timestamp
                traces_df['datetime'] = pd.to_datetime(traces_df['start_time'], unit='s', errors='coerce')
        else:
            traces_df['datetime'] = pd.to_datetime(traces_df['start_time'], unit='s', errors='coerce')
            
        # Remove rows with invalid datetime
        traces_df = traces_df.dropna(subset=['datetime'])
        
        if traces_df.empty:
            return pd.DataFrame()
            
        # Group by time window and service
        grouper = pd.Grouper(key='datetime', freq=time_window)
        
        # Aggregate error data
        agg_data = []
        
        for (service, time_group), group in traces_df.groupby(['service', grouper]):
            if pd.isna(time_group):
                continue
                
            total_traces = len(group)
            error_traces = len(group[group.get('error', False) == True])
            error_rate = error_traces / total_traces if total_traces > 0 else 0
            
            record = {
                'time': int(time_group.timestamp()),
                'service': service,
                'error_count': error_traces,
                'total_traces': total_traces,
                'error_rate': error_rate
            }
            
            agg_data.append(record)
            
        if not agg_data:
            return pd.DataFrame()
            
        tracets_err_df = pd.DataFrame(agg_data)
        
        # Sort by time and service
        tracets_err_df = tracets_err_df.sort_values(['time', 'service'])
        
        return tracets_err_df
        
    def _transform_to_tracets_lat_format(self, traces_df: pd.DataFrame, 
                                       time_window: str = '1min') -> pd.DataFrame:
        """
        Transform traces DataFrame to RE2 tracets_lat.csv format (latency time series).
        
        Args:
            traces_df: DataFrame with traces data
            time_window: Time window for aggregation
            
        Returns:
            DataFrame in tracets_lat.csv format
        """
        if traces_df.empty:
            return pd.DataFrame()
            
        # Convert start_time to datetime for grouping
        traces_df = traces_df.copy()
        
        # Handle different start_time formats
        if traces_df['start_time'].dtype == 'object':
            try:
                traces_df['datetime'] = pd.to_datetime(traces_df['start_time'])
            except:
                traces_df['datetime'] = pd.to_datetime(traces_df['start_time'], unit='s', errors='coerce')
        else:
            traces_df['datetime'] = pd.to_datetime(traces_df['start_time'], unit='s', errors='coerce')
            
        # Remove rows with invalid datetime or duration
        traces_df = traces_df.dropna(subset=['datetime', 'duration_ms'])
        
        if traces_df.empty:
            return pd.DataFrame()
            
        # Ensure duration_ms is numeric
        traces_df['duration_ms'] = pd.to_numeric(traces_df['duration_ms'], errors='coerce')
        traces_df = traces_df.dropna(subset=['duration_ms'])
        
        # Group by time window and service
        grouper = pd.Grouper(key='datetime', freq=time_window)
        
        # Aggregate latency data
        agg_data = []
        
        for (service, time_group), group in traces_df.groupby(['service', grouper]):
            if pd.isna(time_group) or group.empty:
                continue
                
            durations = group['duration_ms']
            
            record = {
                'time': int(time_group.timestamp()),
                'service': service,
                'avg_latency_ms': durations.mean(),
                'median_latency_ms': durations.median(),
                'p95_latency_ms': durations.quantile(0.95),
                'p99_latency_ms': durations.quantile(0.99),
                'max_latency_ms': durations.max(),
                'min_latency_ms': durations.min(),
                'trace_count': len(group)
            }
            
            agg_data.append(record)
            
        if not agg_data:
            return pd.DataFrame()
            
        tracets_lat_df = pd.DataFrame(agg_data)
        
        # Sort by time and service
        tracets_lat_df = tracets_lat_df.sort_values(['time', 'service'])
        
        return tracets_lat_df
        
    def create_service_specific_traces(self, traces_df: pd.DataFrame, service: str,
                                     output_dir: str) -> Tuple[bool, bool, bool]:
        """
        Create service-specific traces.csv, tracets_err.csv, and tracets_lat.csv files.
        
        Args:
            traces_df: DataFrame with traces data
            service: Service name to filter by
            output_dir: Output directory path
            
        Returns:
            Tuple of (traces_csv_success, tracets_err_success, tracets_lat_success)
        """
        try:
            # Filter traces for specific service
            service_traces = traces_df[traces_df['service'] == service].copy()
            
            if service_traces.empty:
                logger.warning(f"No traces found for service: {service}")
                return False, False, False
                
            # Create output paths
            traces_path = os.path.join(output_dir, f"{service}_traces.csv")
            tracets_err_path = os.path.join(output_dir, f"{service}_tracets_err.csv")
            tracets_lat_path = os.path.join(output_dir, f"{service}_tracets_lat.csv")
            
            # Export service-specific traces
            traces_success = self.export_traces_csv(service_traces, traces_path)
            err_success = self.export_tracets_err_csv(service_traces, tracets_err_path)
            lat_success = self.export_tracets_lat_csv(service_traces, tracets_lat_path)
            
            return traces_success, err_success, lat_success
            
        except Exception as e:
            logger.error(f"Failed to create service-specific traces for {service}: {e}")
            return False, False, False
            
    def analyze_trace_patterns(self, traces_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze trace patterns for export insights.
        
        Args:
            traces_df: DataFrame with traces data
            
        Returns:
            Dict with trace pattern analysis
        """
        if traces_df.empty:
            return {}
            
        try:
            analysis = {}
            
            # Service call patterns
            service_calls = traces_df.groupby(['service', 'operation']).size().reset_index(name='count')
            analysis['service_operations'] = service_calls.to_dict('records')
            
            # Error patterns
            if 'error' in traces_df.columns:
                error_patterns = traces_df[traces_df['error'] == True].groupby(['service', 'operation']).size().reset_index(name='error_count')
                analysis['error_patterns'] = error_patterns.to_dict('records')
            
            # Latency patterns
            if 'duration_ms' in traces_df.columns:
                latency_stats = traces_df.groupby('service')['duration_ms'].agg(['mean', 'median', 'std', 'max']).reset_index()
                analysis['latency_patterns'] = latency_stats.to_dict('records')
            
            # Trace complexity (spans per trace)
            trace_complexity = traces_df.groupby('trace_id').size().describe()
            analysis['trace_complexity'] = trace_complexity.to_dict()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze trace patterns: {e}")
            return {}
            
    def validate_traces_csv_format(self, csv_path: str) -> bool:
        """
        Validate exported traces CSV format.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            True if format is valid, False otherwise
        """
        try:
            if not os.path.exists(csv_path):
                logger.error(f"CSV file does not exist: {csv_path}")
                return False
                
            # Read CSV and check format
            df = pd.read_csv(csv_path)
            
            if df.empty:
                logger.warning(f"CSV file is empty: {csv_path}")
                return True  # Empty is valid
                
            # Check for required columns based on filename
            if 'traces.csv' in csv_path:
                required_cols = ['trace_id', 'span_id', 'service', 'operation']
            elif 'tracets_err.csv' in csv_path:
                required_cols = ['time', 'service', 'error_count']
            elif 'tracets_lat.csv' in csv_path:
                required_cols = ['time', 'service', 'avg_latency_ms']
            else:
                required_cols = ['trace_id', 'service']
                
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
                
            logger.info(f"Traces CSV format validation passed: {csv_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate traces CSV format: {e}")
            return False
            
    def get_export_summary(self, traces_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary of traces data for export.
        
        Args:
            traces_df: DataFrame with traces data
            
        Returns:
            Dict with export summary
        """
        if traces_df.empty:
            return {
                'total_spans': 0,
                'total_traces': 0,
                'services': [],
                'operations': [],
                'time_range': None
            }
            
        try:
            summary = {
                'total_spans': len(traces_df),
                'total_traces': traces_df['trace_id'].nunique(),
                'services': sorted(traces_df['service'].unique().tolist()),
                'operations': sorted(traces_df['operation'].unique().tolist()),
                'spans_per_service': traces_df['service'].value_counts().to_dict(),
                'spans_per_operation': traces_df['operation'].value_counts().to_dict()
            }
            
            # Add time range if start_time is available
            if 'start_time' in traces_df.columns:
                try:
                    if traces_df['start_time'].dtype == 'object':
                        timestamps = pd.to_datetime(traces_df['start_time']).astype(int) // 10**9
                    else:
                        timestamps = traces_df['start_time']
                        
                    summary['time_range'] = {
                        'start': int(timestamps.min()),
                        'end': int(timestamps.max()),
                        'duration_seconds': int(timestamps.max() - timestamps.min())
                    }
                except:
                    summary['time_range'] = None
            
            # Add error statistics if available
            if 'error' in traces_df.columns:
                error_count = len(traces_df[traces_df['error'] == True])
                summary['error_count'] = error_count
                summary['error_rate'] = error_count / len(traces_df) if len(traces_df) > 0 else 0
            
            # Add latency statistics if available
            if 'duration_ms' in traces_df.columns:
                durations = pd.to_numeric(traces_df['duration_ms'], errors='coerce').dropna()
                if not durations.empty:
                    summary['latency_stats'] = {
                        'mean_ms': durations.mean(),
                        'median_ms': durations.median(),
                        'p95_ms': durations.quantile(0.95),
                        'p99_ms': durations.quantile(0.99),
                        'max_ms': durations.max()
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate export summary: {e}")
            return {}
            
    def export_all_formats(self, traces_df: pd.DataFrame, output_dir: str,
                          filename_prefix: str = "") -> Tuple[bool, bool, bool]:
        """
        Export all trace CSV formats (traces.csv, tracets_err.csv, tracets_lat.csv).
        
        Args:
            traces_df: DataFrame with traces data
            output_dir: Output directory path
            filename_prefix: Optional prefix for filenames
            
        Returns:
            Tuple of (traces_csv_success, tracets_err_success, tracets_lat_success)
        """
        try:
            # Create output paths
            prefix = f"{filename_prefix}_" if filename_prefix else ""
            traces_path = os.path.join(output_dir, f"{prefix}traces.csv")
            tracets_err_path = os.path.join(output_dir, f"{prefix}tracets_err.csv")
            tracets_lat_path = os.path.join(output_dir, f"{prefix}tracets_lat.csv")
            
            # Export all formats
            traces_success = self.export_traces_csv(traces_df, traces_path)
            err_success = self.export_tracets_err_csv(traces_df, tracets_err_path)
            lat_success = self.export_tracets_lat_csv(traces_df, tracets_lat_path)
            
            return traces_success, err_success, lat_success
            
        except Exception as e:
            logger.error(f"Failed to export all trace formats: {e}")
            return False, False, False