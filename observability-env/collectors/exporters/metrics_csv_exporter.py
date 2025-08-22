"""
Metrics CSV exporter for RE2-compatible format.
Implements requirement 4.1 for metrics data export.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import os

logger = logging.getLogger(__name__)


class MetricsCSVExporter:
    """Export metrics data to CSV format compatible with RE2 datasets."""
    
    def __init__(self):
        """Initialize metrics CSV exporter."""
        self.required_columns = ['timestamp', 'service', 'metric_name', 'value', 'labels']
        
    def export_to_csv(self, metrics_df: pd.DataFrame, output_path: str) -> bool:
        """
        Export metrics DataFrame to CSV file.
        
        Args:
            metrics_df: DataFrame with metrics data
            output_path: Path to output CSV file
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            if metrics_df.empty:
                logger.warning("Empty metrics DataFrame provided")
                # Create empty CSV with headers
                empty_df = pd.DataFrame(columns=['time'] + [f'service_{col}' for col in ['cpu', 'memory', 'network']])
                empty_df.to_csv(output_path, index=False)
                return True
                
            # Validate required columns
            missing_cols = [col for col in self.required_columns if col not in metrics_df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
                
            # Transform to RE2 format
            re2_df = self._transform_to_re2_format(metrics_df)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Export to CSV
            re2_df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(re2_df)} metrics records to {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to export metrics to CSV: {e}")
            return False
            
    def _transform_to_re2_format(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform metrics DataFrame to RE2-compatible format.
        
        The RE2 format uses a wide table with time as the first column
        and each metric as a separate column named by service_metric.
        """
        # Create pivot table with timestamp as index and service_metric as columns
        pivot_df = metrics_df.pivot_table(
            index='timestamp',
            columns=['service', 'metric_name'],
            values='value',
            aggfunc='mean'  # Average if multiple values at same timestamp
        )
        
        # Flatten column names to service_metric format
        pivot_df.columns = [f"{service}_{metric}" for service, metric in pivot_df.columns]
        
        # Reset index to make timestamp a column
        pivot_df = pivot_df.reset_index()
        
        # Rename timestamp to time for RE2 compatibility
        pivot_df = pivot_df.rename(columns={'timestamp': 'time'})
        
        # Fill NaN values with 0.0 (common in RE2 datasets)
        pivot_df = pivot_df.fillna(0.0)
        
        # Sort by time
        pivot_df = pivot_df.sort_values('time')
        
        return pivot_df
        
    def export_time_series_csv(self, metrics_df: pd.DataFrame, output_path: str,
                              time_column: str = 'time') -> bool:
        """
        Export metrics as time series CSV with custom time column format.
        
        Args:
            metrics_df: DataFrame with metrics data
            output_path: Path to output CSV file
            time_column: Name of time column
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            if metrics_df.empty:
                logger.warning("Empty metrics DataFrame provided")
                return False
                
            # Create time series format
            ts_df = metrics_df.copy()
            
            # Ensure timestamp is in the correct format
            if 'timestamp' in ts_df.columns:
                ts_df[time_column] = ts_df['timestamp']
                
            # Select relevant columns for time series
            columns_to_keep = [time_column, 'service', 'metric_name', 'value']
            available_columns = [col for col in columns_to_keep if col in ts_df.columns]
            ts_df = ts_df[available_columns]
            
            # Sort by time and service
            ts_df = ts_df.sort_values([time_column, 'service'])
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Export to CSV
            ts_df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(ts_df)} time series metrics records to {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to export time series metrics to CSV: {e}")
            return False
            
    def create_service_specific_csv(self, metrics_df: pd.DataFrame, service: str,
                                  output_dir: str) -> bool:
        """
        Create service-specific metrics CSV file.
        
        Args:
            metrics_df: DataFrame with metrics data
            service: Service name to filter by
            output_dir: Output directory path
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Filter metrics for specific service
            service_metrics = metrics_df[metrics_df['service'] == service].copy()
            
            if service_metrics.empty:
                logger.warning(f"No metrics found for service: {service}")
                return False
                
            # Create output path
            output_path = os.path.join(output_dir, f"{service}_metrics.csv")
            
            # Export service-specific metrics
            return self.export_to_csv(service_metrics, output_path)
            
        except Exception as e:
            logger.error(f"Failed to create service-specific CSV for {service}: {e}")
            return False
            
    def aggregate_metrics_by_time(self, metrics_df: pd.DataFrame, 
                                time_window: str = '1min') -> pd.DataFrame:
        """
        Aggregate metrics by time windows for export.
        
        Args:
            metrics_df: DataFrame with metrics data
            time_window: Time window for aggregation (e.g., '1min', '5min')
            
        Returns:
            Aggregated DataFrame
        """
        if metrics_df.empty:
            return metrics_df
            
        try:
            # Convert timestamp to datetime
            metrics_df = metrics_df.copy()
            metrics_df['datetime'] = pd.to_datetime(metrics_df['timestamp'], unit='s')
            
            # Group by time window, service, and metric
            grouper = pd.Grouper(key='datetime', freq=time_window)
            
            aggregated = metrics_df.groupby(['service', 'metric_name', grouper]).agg({
                'value': ['mean', 'max', 'min', 'count'],
                'timestamp': 'first'
            }).reset_index()
            
            # Flatten column names
            aggregated.columns = ['service', 'metric_name', 'datetime', 
                                'value_mean', 'value_max', 'value_min', 'value_count', 'timestamp']
            
            # Use mean as primary value
            aggregated['value'] = aggregated['value_mean']
            
            # Drop intermediate columns
            aggregated = aggregated[['timestamp', 'service', 'metric_name', 'value']]
            
            return aggregated.sort_values(['timestamp', 'service'])
            
        except Exception as e:
            logger.error(f"Failed to aggregate metrics: {e}")
            return metrics_df
            
    def validate_csv_format(self, csv_path: str) -> bool:
        """
        Validate exported CSV format.
        
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
                
            # Check for time column
            if 'time' not in df.columns and 'timestamp' not in df.columns:
                logger.error("CSV missing time/timestamp column")
                return False
                
            # Check for numeric data
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                logger.error("CSV contains no numeric data")
                return False
                
            logger.info(f"CSV format validation passed: {csv_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate CSV format: {e}")
            return False
            
    def get_export_summary(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary of metrics data for export.
        
        Args:
            metrics_df: DataFrame with metrics data
            
        Returns:
            Dict with export summary
        """
        if metrics_df.empty:
            return {
                'total_records': 0,
                'services': [],
                'metrics': [],
                'time_range': None
            }
            
        try:
            summary = {
                'total_records': len(metrics_df),
                'services': sorted(metrics_df['service'].unique().tolist()),
                'metrics': sorted(metrics_df['metric_name'].unique().tolist()),
                'time_range': {
                    'start': int(metrics_df['timestamp'].min()),
                    'end': int(metrics_df['timestamp'].max()),
                    'duration_seconds': int(metrics_df['timestamp'].max() - metrics_df['timestamp'].min())
                },
                'records_per_service': metrics_df['service'].value_counts().to_dict(),
                'records_per_metric': metrics_df['metric_name'].value_counts().to_dict()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate export summary: {e}")
            return {}