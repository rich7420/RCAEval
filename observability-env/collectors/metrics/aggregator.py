"""
Metrics aggregation and sampling logic for time-series data processing.
Implements advanced aggregation strategies for metrics collection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class MetricsAggregator:
    """Advanced metrics aggregation with multiple strategies."""
    
    def __init__(self, default_window: str = "1min"):
        """
        Initialize metrics aggregator.
        
        Args:
            default_window: Default aggregation window
        """
        self.default_window = default_window
        self.aggregation_functions = {
            'mean': np.mean,
            'median': np.median,
            'max': np.max,
            'min': np.min,
            'sum': np.sum,
            'std': np.std,
            'count': len,
            'p95': lambda x: np.percentile(x, 95),
            'p99': lambda x: np.percentile(x, 99)
        }
        
    def aggregate_by_time_window(self, df: pd.DataFrame, window: str = None,
                               functions: List[str] = None) -> pd.DataFrame:
        """
        Aggregate metrics by time windows with multiple functions.
        
        Args:
            df: Input DataFrame with timestamp, service, metric_name, value columns
            window: Time window for aggregation (e.g., "1min", "5min")
            functions: List of aggregation functions to apply
            
        Returns:
            Aggregated DataFrame
        """
        if df.empty:
            return df
            
        window = window or self.default_window
        functions = functions or ['mean']
        
        # Validate functions
        invalid_funcs = [f for f in functions if f not in self.aggregation_functions]
        if invalid_funcs:
            raise ValueError(f"Invalid aggregation functions: {invalid_funcs}")
            
        df_copy = df.copy()
        df_copy['datetime'] = pd.to_datetime(df_copy['timestamp'], unit='s')
        
        # Group by service, metric_name, and time window
        grouper = pd.Grouper(key='datetime', freq=window)
        grouped = df_copy.groupby(['service', 'metric_name', grouper])
        
        # Apply multiple aggregation functions
        agg_results = []
        
        for func_name in functions:
            func = self.aggregation_functions[func_name]
            
            if func_name == 'count':
                agg_df = grouped.size().reset_index(name='value')
            else:
                agg_df = grouped['value'].apply(func).reset_index()
                
            agg_df['aggregation'] = func_name
            agg_results.append(agg_df)
            
        # Combine results
        combined_df = pd.concat(agg_results, ignore_index=True)
        
        # Convert back to timestamp
        combined_df['timestamp'] = combined_df['datetime'].astype(int) // 10**9
        combined_df = combined_df.drop('datetime', axis=1)
        
        # Update metric names to include aggregation
        combined_df['metric_name'] = combined_df['metric_name'] + '_' + combined_df['aggregation']
        combined_df = combined_df.drop('aggregation', axis=1)
        
        return combined_df.sort_values(['timestamp', 'service', 'metric_name'])
        
    def downsample_metrics(self, df: pd.DataFrame, target_points: int = 1000) -> pd.DataFrame:
        """
        Downsample metrics to target number of data points per metric.
        
        Args:
            df: Input DataFrame
            target_points: Target number of points per metric series
            
        Returns:
            Downsampled DataFrame
        """
        if df.empty:
            return df
            
        downsampled_results = []
        
        for (service, metric), group in df.groupby(['service', 'metric_name']):
            if len(group) <= target_points:
                downsampled_results.append(group)
                continue
                
            # Calculate sampling interval
            total_points = len(group)
            step = max(1, total_points // target_points)
            
            # Sample every nth point
            sampled = group.iloc[::step].copy()
            downsampled_results.append(sampled)
            
        return pd.concat(downsampled_results, ignore_index=True)
        
    def create_rolling_aggregates(self, df: pd.DataFrame, windows: List[str] = None) -> pd.DataFrame:
        """
        Create rolling window aggregates for metrics.
        
        Args:
            df: Input DataFrame
            windows: List of rolling window sizes (e.g., ["5min", "15min", "1h"])
            
        Returns:
            DataFrame with rolling aggregates
        """
        if df.empty:
            return df
            
        windows = windows or ["5min", "15min", "1h"]
        
        df_copy = df.copy()
        df_copy['datetime'] = pd.to_datetime(df_copy['timestamp'], unit='s')
        df_copy = df_copy.sort_values(['service', 'metric_name', 'datetime'])
        
        rolling_results = []
        
        for (service, metric), group in df_copy.groupby(['service', 'metric_name']):
            group_indexed = group.set_index('datetime')
            
            for window in windows:
                # Create rolling mean
                rolling_mean = group_indexed['value'].rolling(window).mean()
                rolling_std = group_indexed['value'].rolling(window).std()
                
                # Create new rows for rolling metrics
                for suffix, values in [('_rolling_mean', rolling_mean), ('_rolling_std', rolling_std)]:
                    rolling_df = group.copy()
                    rolling_df['metric_name'] = metric + f'_{window}' + suffix
                    rolling_df['value'] = values.values
                    
                    # Remove NaN values from rolling calculations
                    rolling_df = rolling_df.dropna(subset=['value'])
                    rolling_results.append(rolling_df)
                    
        if not rolling_results:
            return df_copy.drop('datetime', axis=1)
            
        # Combine original and rolling metrics
        all_results = [df_copy] + rolling_results
        combined = pd.concat(all_results, ignore_index=True)
        
        return combined.drop('datetime', axis=1).sort_values(['timestamp', 'service', 'metric_name'])


class SamplingStrategy:
    """Intelligent sampling strategies for metrics data."""
    
    @staticmethod
    def uniform_sampling(df: pd.DataFrame, sample_rate: float = 0.1) -> pd.DataFrame:
        """
        Apply uniform random sampling to metrics data.
        
        Args:
            df: Input DataFrame
            sample_rate: Fraction of data to keep (0.0 to 1.0)
            
        Returns:
            Sampled DataFrame
        """
        if df.empty or sample_rate >= 1.0:
            return df
            
        return df.sample(frac=sample_rate, random_state=42).sort_values(['timestamp', 'service'])
        
    @staticmethod
    def stratified_sampling(df: pd.DataFrame, sample_rate: float = 0.1) -> pd.DataFrame:
        """
        Apply stratified sampling by service and metric.
        
        Args:
            df: Input DataFrame
            sample_rate: Fraction of data to keep per stratum
            
        Returns:
            Sampled DataFrame
        """
        if df.empty or sample_rate >= 1.0:
            return df
            
        sampled_groups = []
        
        for (service, metric), group in df.groupby(['service', 'metric_name']):
            n_samples = max(1, int(len(group) * sample_rate))
            sampled = group.sample(n=min(n_samples, len(group)), random_state=42)
            sampled_groups.append(sampled)
            
        return pd.concat(sampled_groups, ignore_index=True).sort_values(['timestamp', 'service'])
        
    @staticmethod
    def adaptive_sampling(df: pd.DataFrame, variance_threshold: float = 0.1) -> pd.DataFrame:
        """
        Apply adaptive sampling based on data variance.
        Keep more samples in high-variance regions.
        
        Args:
            df: Input DataFrame
            variance_threshold: Threshold for determining high-variance regions
            
        Returns:
            Adaptively sampled DataFrame
        """
        if df.empty:
            return df
            
        sampled_groups = []
        
        for (service, metric), group in df.groupby(['service', 'metric_name']):
            if len(group) < 10:  # Keep small groups as-is
                sampled_groups.append(group)
                continue
                
            # Calculate rolling variance
            group_sorted = group.sort_values('timestamp')
            rolling_var = group_sorted['value'].rolling(window=10, min_periods=1).var()
            
            # Determine sampling rate based on variance
            high_variance_mask = rolling_var > rolling_var.quantile(1 - variance_threshold)
            
            # Sample more from high-variance regions
            high_var_samples = group_sorted[high_variance_mask]
            low_var_samples = group_sorted[~high_variance_mask].sample(
                frac=0.3, random_state=42
            ) if len(group_sorted[~high_variance_mask]) > 0 else pd.DataFrame()
            
            combined_samples = pd.concat([high_var_samples, low_var_samples], ignore_index=True)
            sampled_groups.append(combined_samples)
            
        return pd.concat(sampled_groups, ignore_index=True).sort_values(['timestamp', 'service'])


class MetricsQualityAnalyzer:
    """Analyze and ensure quality of collected metrics."""
    
    def __init__(self):
        """Initialize quality analyzer."""
        self.quality_metrics = {}
        
    def analyze_completeness(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze data completeness by service and metric.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dict with completeness scores
        """
        if df.empty:
            return {}
            
        completeness = {}
        
        for (service, metric), group in df.groupby(['service', 'metric_name']):
            # Calculate missing value ratio
            missing_ratio = group['value'].isna().sum() / len(group)
            completeness_score = 1.0 - missing_ratio
            
            completeness[f"{service}_{metric}"] = completeness_score
            
        return completeness
        
    def detect_anomalies(self, df: pd.DataFrame, method: str = "zscore", 
                        threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect anomalies in metrics data.
        
        Args:
            df: Input DataFrame
            method: Anomaly detection method ("zscore", "iqr")
            threshold: Threshold for anomaly detection
            
        Returns:
            DataFrame with anomaly flags
        """
        if df.empty:
            return df
            
        df_with_anomalies = df.copy()
        df_with_anomalies['is_anomaly'] = False
        
        for (service, metric), group in df_with_anomalies.groupby(['service', 'metric_name']):
            if method == "zscore":
                z_scores = np.abs((group['value'] - group['value'].mean()) / group['value'].std())
                anomaly_mask = z_scores > threshold
                
            elif method == "iqr":
                Q1 = group['value'].quantile(0.25)
                Q3 = group['value'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                anomaly_mask = (group['value'] < lower_bound) | (group['value'] > upper_bound)
                
            else:
                continue
                
            df_with_anomalies.loc[group.index[anomaly_mask], 'is_anomaly'] = True
            
        return df_with_anomalies
        
    def calculate_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculate overall quality score for metrics dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if df.empty:
            return 0.0
            
        # Completeness score
        completeness_scores = list(self.analyze_completeness(df).values())
        avg_completeness = np.mean(completeness_scores) if completeness_scores else 0.0
        
        # Anomaly score (lower anomaly rate = higher quality)
        df_with_anomalies = self.detect_anomalies(df)
        anomaly_rate = df_with_anomalies['is_anomaly'].sum() / len(df_with_anomalies)
        anomaly_score = max(0.0, 1.0 - anomaly_rate)
        
        # Temporal consistency score (regular intervals)
        temporal_score = self._calculate_temporal_consistency(df)
        
        # Combined quality score
        quality_score = (avg_completeness * 0.4 + anomaly_score * 0.3 + temporal_score * 0.3)
        
        return min(1.0, max(0.0, quality_score))
        
    def _calculate_temporal_consistency(self, df: pd.DataFrame) -> float:
        """Calculate temporal consistency score."""
        if df.empty:
            return 0.0
            
        consistency_scores = []
        
        for (service, metric), group in df.groupby(['service', 'metric_name']):
            if len(group) < 3:
                continue
                
            # Calculate time intervals
            sorted_group = group.sort_values('timestamp')
            intervals = np.diff(sorted_group['timestamp'])
            
            if len(intervals) == 0:
                continue
                
            # Calculate coefficient of variation for intervals
            interval_cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 1.0
            
            # Lower CV = higher consistency
            consistency_score = max(0.0, 1.0 - interval_cv)
            consistency_scores.append(consistency_score)
            
        return np.mean(consistency_scores) if consistency_scores else 0.0