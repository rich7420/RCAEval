"""
Logs CSV exporter for RE2-compatible format.
Implements requirement 4.2 for logs data export.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
import os
import json
import re

logger = logging.getLogger(__name__)


class LogsCSVExporter:
    """Export logs data to CSV format compatible with RE2 datasets."""
    
    def __init__(self):
        """Initialize logs CSV exporter."""
        self.required_columns = ['timestamp', 'service', 'level', 'message']
        
    def export_logs_csv(self, logs_df: pd.DataFrame, output_path: str) -> bool:
        """
        Export logs DataFrame to logs.csv file.
        
        Args:
            logs_df: DataFrame with logs data
            output_path: Path to output logs.csv file
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            if logs_df.empty:
                logger.warning("Empty logs DataFrame provided")
                # Create empty CSV with headers
                empty_df = pd.DataFrame(columns=['time', 'timestamp', 'container_name', 'message', 'level', 'req_path', 'error', 'cluster_id', 'log_template'])
                empty_df.to_csv(output_path, index=False)
                return True
                
            # Validate required columns
            missing_cols = [col for col in self.required_columns if col not in logs_df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
                
            # Transform to RE2 logs.csv format
            logs_csv_df = self._transform_to_logs_csv_format(logs_df)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Export to CSV
            logs_csv_df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(logs_csv_df)} log records to {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to export logs to CSV: {e}")
            return False
            
    def export_logts_csv(self, logs_df: pd.DataFrame, output_path: str) -> bool:
        """
        Export logs DataFrame to logts.csv file (time series format).
        
        Args:
            logs_df: DataFrame with logs data
            output_path: Path to output logts.csv file
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            if logs_df.empty:
                logger.warning("Empty logs DataFrame provided")
                # Create empty CSV with headers
                empty_df = pd.DataFrame(columns=['time', 'service', 'log_count', 'error_count', 'warning_count', 'info_count'])
                empty_df.to_csv(output_path, index=False)
                return True
                
            # Transform to RE2 logts.csv format (time series aggregation)
            logts_csv_df = self._transform_to_logts_csv_format(logs_df)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Export to CSV
            logts_csv_df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(logts_csv_df)} log time series records to {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to export logts to CSV: {e}")
            return False
            
    def _transform_to_logs_csv_format(self, logs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform logs DataFrame to RE2 logs.csv format.
        
        Expected columns: time, timestamp, container_name, message, level, req_path, error, cluster_id, log_template
        """
        logs_csv = logs_df.copy()
        
        # Create time column (formatted timestamp)
        logs_csv['time'] = logs_csv['timestamp'].apply(self._format_timestamp_for_display)
        
        # Map service to container_name
        logs_csv['container_name'] = logs_csv['service']
        
        # Extract request path from message if available
        logs_csv['req_path'] = logs_csv['message'].apply(self._extract_request_path)
        
        # Determine if log indicates an error
        logs_csv['error'] = logs_csv['level'].apply(lambda x: x.upper() in ['ERROR', 'FATAL', 'CRITICAL'])
        
        # Add cluster_id (default to 1 for single cluster)
        logs_csv['cluster_id'] = 1
        
        # Generate log template (simplified version)
        logs_csv['log_template'] = logs_csv.apply(self._generate_log_template, axis=1)
        
        # Select and order columns for RE2 format
        re2_columns = ['time', 'timestamp', 'container_name', 'message', 'level', 'req_path', 'error', 'cluster_id', 'log_template']
        
        # Ensure all columns exist
        for col in re2_columns:
            if col not in logs_csv.columns:
                logs_csv[col] = ''
                
        logs_csv = logs_csv[re2_columns]
        
        # Sort by timestamp
        logs_csv = logs_csv.sort_values('timestamp')
        
        return logs_csv
        
    def _transform_to_logts_csv_format(self, logs_df: pd.DataFrame, 
                                     time_window: str = '1min') -> pd.DataFrame:
        """
        Transform logs DataFrame to RE2 logts.csv format (time series aggregation).
        
        Args:
            logs_df: DataFrame with logs data
            time_window: Time window for aggregation
            
        Returns:
            DataFrame in logts.csv format
        """
        if logs_df.empty:
            return pd.DataFrame()
            
        # Convert timestamp to datetime for grouping
        logs_df = logs_df.copy()
        logs_df['datetime'] = pd.to_datetime(logs_df['timestamp'], unit='s')
        
        # Group by time window and service
        grouper = pd.Grouper(key='datetime', freq=time_window)
        
        # Aggregate log counts by level
        agg_data = []
        
        for (service, time_group), group in logs_df.groupby(['service', grouper]):
            if pd.isna(time_group):
                continue
                
            # Count logs by level
            level_counts = group['level'].value_counts()
            
            record = {
                'time': int(time_group.timestamp()),
                'service': service,
                'log_count': len(group),
                'error_count': level_counts.get('ERROR', 0) + level_counts.get('FATAL', 0) + level_counts.get('CRITICAL', 0),
                'warning_count': level_counts.get('WARNING', 0) + level_counts.get('WARN', 0),
                'info_count': level_counts.get('INFO', 0),
                'debug_count': level_counts.get('DEBUG', 0),
                'trace_count': level_counts.get('TRACE', 0)
            }
            
            agg_data.append(record)
            
        if not agg_data:
            return pd.DataFrame()
            
        logts_df = pd.DataFrame(agg_data)
        
        # Sort by time and service
        logts_df = logts_df.sort_values(['time', 'service'])
        
        return logts_df
        
    def _format_timestamp_for_display(self, timestamp: int) -> str:
        """Format timestamp for display in time column."""
        try:
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime("%H:%M")
        except:
            return "00:00"
            
    def _extract_request_path(self, message: str) -> str:
        """Extract request path from log message."""
        if not isinstance(message, str):
            return ""
            
        # Common patterns for request paths
        path_patterns = [
            r'(?:GET|POST|PUT|DELETE|PATCH)\s+([^\s]+)',  # HTTP method + path
            r'path[=:]\s*([^\s,}]+)',  # path= or path:
            r'uri[=:]\s*([^\s,}]+)',   # uri= or uri:
            r'url[=:]\s*([^\s,}]+)',   # url= or url:
            r'"path"\s*:\s*"([^"]+)"', # JSON path
            r'/[a-zA-Z0-9/_-]+',       # Simple path pattern
        ]
        
        for pattern in path_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                if len(match.groups()) > 0:
                    return match.group(1)
                else:
                    return match.group(0)
                    
        return ""
        
    def _generate_log_template(self, row: pd.Series) -> str:
        """Generate simplified log template."""
        container = row.get('container_name', '')
        message = row.get('message', '')
        
        # Create a simplified template by truncating the message
        if len(message) > 50:
            template = f"{container} {message[:50]}"
        else:
            template = f"{container} {message}"
            
        return template
        
    def create_service_specific_logs(self, logs_df: pd.DataFrame, service: str,
                                   output_dir: str) -> Tuple[bool, bool]:
        """
        Create service-specific logs.csv and logts.csv files.
        
        Args:
            logs_df: DataFrame with logs data
            service: Service name to filter by
            output_dir: Output directory path
            
        Returns:
            Tuple of (logs_csv_success, logts_csv_success)
        """
        try:
            # Filter logs for specific service
            service_logs = logs_df[logs_df['service'] == service].copy()
            
            if service_logs.empty:
                logger.warning(f"No logs found for service: {service}")
                return False, False
                
            # Create output paths
            logs_path = os.path.join(output_dir, f"{service}_logs.csv")
            logts_path = os.path.join(output_dir, f"{service}_logts.csv")
            
            # Export service-specific logs
            logs_success = self.export_logs_csv(service_logs, logs_path)
            logts_success = self.export_logts_csv(service_logs, logts_path)
            
            return logs_success, logts_success
            
        except Exception as e:
            logger.error(f"Failed to create service-specific logs for {service}: {e}")
            return False, False
            
    def aggregate_logs_by_template(self, logs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate logs by template for analysis.
        
        Args:
            logs_df: DataFrame with logs data
            
        Returns:
            DataFrame with template aggregation
        """
        if logs_df.empty:
            return pd.DataFrame()
            
        try:
            # Group by service and extract patterns
            template_data = []
            
            for service, group in logs_df.groupby('service'):
                messages = group['message'].tolist()
                
                # Simple template extraction (could be enhanced)
                template_counts = {}
                
                for message in messages:
                    # Create simple template by replacing numbers and IDs
                    template = re.sub(r'\b\d+\b', '<NUM>', message)
                    template = re.sub(r'\b[a-fA-F0-9]{8,}\b', '<ID>', template)
                    template = re.sub(r'\b\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}', '<TIMESTAMP>', template)
                    
                    template_counts[template] = template_counts.get(template, 0) + 1
                    
                # Add to results
                for template, count in template_counts.items():
                    template_data.append({
                        'service': service,
                        'template': template,
                        'count': count,
                        'frequency': count / len(messages)
                    })
                    
            if not template_data:
                return pd.DataFrame()
                
            template_df = pd.DataFrame(template_data)
            template_df = template_df.sort_values(['service', 'count'], ascending=[True, False])
            
            return template_df
            
        except Exception as e:
            logger.error(f"Failed to aggregate logs by template: {e}")
            return pd.DataFrame()
            
    def validate_logs_csv_format(self, csv_path: str) -> bool:
        """
        Validate exported logs CSV format.
        
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
            if 'logs.csv' in csv_path:
                required_cols = ['time', 'timestamp', 'container_name', 'message', 'level']
            elif 'logts.csv' in csv_path:
                required_cols = ['time', 'service', 'log_count']
            else:
                required_cols = ['timestamp', 'message']
                
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
                
            logger.info(f"Logs CSV format validation passed: {csv_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate logs CSV format: {e}")
            return False
            
    def get_export_summary(self, logs_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary of logs data for export.
        
        Args:
            logs_df: DataFrame with logs data
            
        Returns:
            Dict with export summary
        """
        if logs_df.empty:
            return {
                'total_records': 0,
                'services': [],
                'levels': [],
                'time_range': None
            }
            
        try:
            summary = {
                'total_records': len(logs_df),
                'services': sorted(logs_df['service'].unique().tolist()),
                'levels': sorted(logs_df['level'].unique().tolist()),
                'time_range': {
                    'start': int(logs_df['timestamp'].min()),
                    'end': int(logs_df['timestamp'].max()),
                    'duration_seconds': int(logs_df['timestamp'].max() - logs_df['timestamp'].min())
                },
                'records_per_service': logs_df['service'].value_counts().to_dict(),
                'records_per_level': logs_df['level'].value_counts().to_dict(),
                'error_rate': len(logs_df[logs_df['level'].isin(['ERROR', 'FATAL', 'CRITICAL'])]) / len(logs_df)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate export summary: {e}")
            return {}
            
    def export_both_formats(self, logs_df: pd.DataFrame, output_dir: str,
                          filename_prefix: str = "") -> Tuple[bool, bool]:
        """
        Export both logs.csv and logts.csv formats.
        
        Args:
            logs_df: DataFrame with logs data
            output_dir: Output directory path
            filename_prefix: Optional prefix for filenames
            
        Returns:
            Tuple of (logs_csv_success, logts_csv_success)
        """
        try:
            # Create output paths
            prefix = f"{filename_prefix}_" if filename_prefix else ""
            logs_path = os.path.join(output_dir, f"{prefix}logs.csv")
            logts_path = os.path.join(output_dir, f"{prefix}logts.csv")
            
            # Export both formats
            logs_success = self.export_logs_csv(logs_df, logs_path)
            logts_success = self.export_logts_csv(logs_df, logts_path)
            
            return logs_success, logts_success
            
        except Exception as e:
            logger.error(f"Failed to export both log formats: {e}")
            return False, False