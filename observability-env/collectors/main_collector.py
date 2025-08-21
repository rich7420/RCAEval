"""
Main data collector that integrates metrics, logs, traces, and cluster info generation.
Implements the complete data collection and processing system for requirements 4.1, 4.2, 4.3, and 6.4.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import asyncio
import concurrent.futures
from pathlib import Path

from metrics import MetricsCollector, MetricsAggregator, SamplingStrategy, MetricsQualityAnalyzer
from logs import LogsCollector, LogClusterer, LogPatternAnalyzer
from traces import TracesCollector, TraceAnalyzer, TraceFlowAnalyzer
from exporters import ClusterInfoGenerator

logger = logging.getLogger(__name__)


class ObservabilityDataCollector:
    """Main collector that orchestrates all data collection components."""
    
    def __init__(self, prometheus_url: str = "http://localhost:9090",
                 loki_url: str = "http://localhost:3100",
                 jaeger_url: str = "http://localhost:16686"):
        """
        Initialize the main data collector.
        
        Args:
            prometheus_url: Prometheus server URL
            loki_url: Loki server URL
            jaeger_url: Jaeger server URL
        """
        self.metrics_collector = MetricsCollector(prometheus_url)
        self.logs_collector = LogsCollector(loki_url)
        self.traces_collector = TracesCollector(jaeger_url)
        
        self.metrics_aggregator = MetricsAggregator()
        self.log_clusterer = LogClusterer(method="drain")
        self.log_pattern_analyzer = LogPatternAnalyzer()
        self.trace_analyzer = TraceAnalyzer()
        self.trace_flow_analyzer = TraceFlowAnalyzer()
        self.cluster_info_generator = ClusterInfoGenerator()
        
        self.quality_analyzer = MetricsQualityAnalyzer()
        
    def collect_all_data(self, start_time: datetime, end_time: datetime,
                        services: Optional[List[str]] = None,
                        config: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
        """
        Collect all observability data (metrics, logs, traces).
        
        Args:
            start_time: Collection start time
            end_time: Collection end time
            services: List of services to collect (None for all)
            config: Collection configuration
            
        Returns:
            Dict with collected data DataFrames
        """
        config = config or {}
        
        logger.info(f"Starting data collection from {start_time} to {end_time}")
        
        # Collect data in parallel for better performance
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit collection tasks
            metrics_future = executor.submit(
                self._collect_metrics_with_config, start_time, end_time, services, config
            )
            logs_future = executor.submit(
                self._collect_logs_with_config, start_time, end_time, services, config
            )
            traces_future = executor.submit(
                self._collect_traces_with_config, start_time, end_time, services, config
            )
            
            # Wait for completion
            try:
                metrics_df = metrics_future.result(timeout=300)  # 5 minute timeout
                logs_df = logs_future.result(timeout=300)
                traces_df = traces_future.result(timeout=300)
                
            except concurrent.futures.TimeoutError:
                logger.error("Data collection timed out")
                raise
            except Exception as e:
                logger.error(f"Data collection failed: {e}")
                raise
                
        logger.info(f"Data collection completed. Metrics: {len(metrics_df)}, "
                   f"Logs: {len(logs_df)}, Traces: {len(traces_df)}")
        
        return {
            'metrics': metrics_df,
            'logs': logs_df,
            'traces': traces_df
        }
        
    def _collect_metrics_with_config(self, start_time: datetime, end_time: datetime,
                                   services: Optional[List[str]], config: Dict[str, Any]) -> pd.DataFrame:
        """Collect metrics with configuration."""
        try:
            metrics_config = config.get('metrics', {})
            
            # Collect raw metrics
            metrics_df = self.metrics_collector.collect_system_metrics(
                start_time, end_time, services
            )
            
            if metrics_df.empty:
                return metrics_df
                
            # Apply processing based on config
            if metrics_config.get('aggregate', False):
                window = metrics_config.get('aggregation_window', '1min')
                functions = metrics_config.get('aggregation_functions', ['mean'])
                metrics_df = self.metrics_aggregator.aggregate_by_time_window(
                    metrics_df, window, functions
                )
                
            if metrics_config.get('sample', False):
                sample_rate = metrics_config.get('sample_rate', 0.1)
                strategy = metrics_config.get('sampling_strategy', 'uniform')
                
                if strategy == 'uniform':
                    metrics_df = SamplingStrategy.uniform_sampling(metrics_df, sample_rate)
                elif strategy == 'stratified':
                    metrics_df = SamplingStrategy.stratified_sampling(metrics_df, sample_rate)
                elif strategy == 'adaptive':
                    metrics_df = SamplingStrategy.adaptive_sampling(metrics_df)
                    
            if metrics_config.get('remove_outliers', False):
                threshold = metrics_config.get('outlier_threshold', 1.5)
                method = metrics_config.get('outlier_method', 'iqr')
                metrics_df = self.metrics_collector.processor.remove_outliers(
                    metrics_df, method, threshold
                )
                
            return metrics_df
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
            return pd.DataFrame()
            
    def _collect_logs_with_config(self, start_time: datetime, end_time: datetime,
                                services: Optional[List[str]], config: Dict[str, Any]) -> pd.DataFrame:
        """Collect logs with configuration."""
        try:
            logs_config = config.get('logs', {})
            
            # Collect raw logs
            logs_df = self.logs_collector.collect_service_logs(
                start_time, end_time, services
            )
            
            if logs_df.empty:
                return logs_df
                
            # Apply log-specific processing
            if logs_config.get('extract_templates', True):
                # Extract templates and add template IDs
                templates = self.logs_collector.extract_log_templates(logs_df)
                logs_df = self.logs_collector.add_template_ids(logs_df, templates)
                
            return logs_df
            
        except Exception as e:
            logger.error(f"Logs collection failed: {e}")
            return pd.DataFrame()
            
    def _collect_traces_with_config(self, start_time: datetime, end_time: datetime,
                                  services: Optional[List[str]], config: Dict[str, Any]) -> pd.DataFrame:
        """Collect traces with configuration."""
        try:
            traces_config = config.get('traces', {})
            
            # Collect raw traces
            traces_df = self.traces_collector.collect_service_traces(
                start_time, end_time, services
            )
            
            if traces_df.empty:
                return traces_df
                
            # Apply trace-specific processing
            if traces_config.get('analyze_dependencies', True):
                # Build dependency graph
                self.trace_analyzer.build_service_dependency_graph(traces_df)
                
            return traces_df
            
        except Exception as e:
            logger.error(f"Traces collection failed: {e}")
            return pd.DataFrame()
            
    def generate_comprehensive_analysis(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate comprehensive analysis of collected data.
        
        Args:
            data: Dict with collected data DataFrames
            
        Returns:
            Dict with analysis results
        """
        analysis = {
            'metrics_analysis': {},
            'logs_analysis': {},
            'traces_analysis': {},
            'cross_data_analysis': {},
            'quality_analysis': {}
        }
        
        metrics_df = data.get('metrics', pd.DataFrame())
        logs_df = data.get('logs', pd.DataFrame())
        traces_df = data.get('traces', pd.DataFrame())
        
        # Metrics analysis
        if not metrics_df.empty:
            analysis['metrics_analysis'] = {
                'total_metrics': len(metrics_df),
                'unique_services': metrics_df['service'].nunique(),
                'unique_metrics': metrics_df['metric_name'].nunique(),
                'time_range': {
                    'start': metrics_df['timestamp'].min(),
                    'end': metrics_df['timestamp'].max()
                },
                'quality_score': self.quality_analyzer.calculate_quality_score(metrics_df)
            }
            
        # Logs analysis
        if not logs_df.empty:
            # Extract templates for analysis
            templates = self.logs_collector.extract_log_templates(logs_df)
            
            # Cluster logs
            all_messages = logs_df['message'].tolist()
            clusters = self.log_clusterer.cluster_logs(all_messages)
            
            # Analyze patterns
            pattern_analysis = self.log_pattern_analyzer.analyze_cluster_patterns(clusters)
            error_patterns = self.log_pattern_analyzer.detect_error_patterns(clusters)
            
            analysis['logs_analysis'] = {
                'total_logs': len(logs_df),
                'unique_services': logs_df['service'].nunique(),
                'log_levels': logs_df['level'].value_counts().to_dict(),
                'templates_count': len(templates),
                'clusters_count': len(clusters),
                'pattern_analysis': pattern_analysis,
                'error_patterns': error_patterns[:10]  # Top 10 error patterns
            }
            
        # Traces analysis
        if not traces_df.empty:
            traces_analysis = self.traces_collector.analyze_traces(traces_df)
            
            # Additional analysis
            critical_paths = self.trace_analyzer.identify_critical_path(traces_df)
            bottlenecks = self.trace_analyzer.find_bottlenecks(traces_df)
            service_health = self.trace_analyzer.analyze_service_health(traces_df)
            
            analysis['traces_analysis'] = {
                **traces_analysis,
                'critical_paths': critical_paths[:5],  # Top 5 critical paths
                'bottlenecks': bottlenecks[:5],  # Top 5 bottlenecks
                'service_health': service_health
            }
            
        # Cross-data analysis
        analysis['cross_data_analysis'] = self._perform_cross_data_analysis(
            metrics_df, logs_df, traces_df
        )
        
        # Quality analysis
        analysis['quality_analysis'] = self._perform_quality_analysis(
            metrics_df, logs_df, traces_df
        )
        
        return analysis
        
    def _perform_cross_data_analysis(self, metrics_df: pd.DataFrame,
                                   logs_df: pd.DataFrame, traces_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform cross-data analysis between different data sources."""
        cross_analysis = {}
        
        # Service coverage analysis
        services_in_metrics = set(metrics_df['service'].unique()) if not metrics_df.empty else set()
        services_in_logs = set(logs_df['service'].unique()) if not logs_df.empty else set()
        services_in_traces = set(traces_df['service'].unique()) if not traces_df.empty else set()
        
        all_services = services_in_metrics | services_in_logs | services_in_traces
        
        cross_analysis['service_coverage'] = {
            'total_services': len(all_services),
            'in_metrics': len(services_in_metrics),
            'in_logs': len(services_in_logs),
            'in_traces': len(services_in_traces),
            'in_all_sources': len(services_in_metrics & services_in_logs & services_in_traces),
            'missing_from_metrics': list(all_services - services_in_metrics),
            'missing_from_logs': list(all_services - services_in_logs),
            'missing_from_traces': list(all_services - services_in_traces)
        }
        
        # Time range analysis
        time_ranges = {}
        if not metrics_df.empty:
            time_ranges['metrics'] = {
                'start': metrics_df['timestamp'].min(),
                'end': metrics_df['timestamp'].max()
            }
        if not logs_df.empty:
            time_ranges['logs'] = {
                'start': logs_df['timestamp'].min(),
                'end': logs_df['timestamp'].max()
            }
        if not traces_df.empty:
            time_ranges['traces'] = {
                'start': traces_df['start_time'].min(),
                'end': traces_df['start_time'].max()
            }
            
        cross_analysis['time_ranges'] = time_ranges
        
        # Correlation opportunities
        if not traces_df.empty and not logs_df.empty:
            # Correlate traces with logs
            correlated_traces = self.traces_collector.correlator.correlate_with_logs(
                traces_df, logs_df
            )
            
            cross_analysis['trace_log_correlation'] = {
                'total_traces': len(traces_df),
                'traces_with_logs': len(correlated_traces[correlated_traces['correlated_log_count'] > 0]),
                'avg_logs_per_trace': correlated_traces['correlated_log_count'].mean()
            }
            
        return cross_analysis
        
    def _perform_quality_analysis(self, metrics_df: pd.DataFrame,
                                logs_df: pd.DataFrame, traces_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform data quality analysis."""
        quality_analysis = {}
        
        # Metrics quality
        if not metrics_df.empty:
            quality_analysis['metrics'] = {
                'completeness': self.quality_analyzer.analyze_completeness(metrics_df),
                'overall_quality_score': self.quality_analyzer.calculate_quality_score(metrics_df),
                'anomaly_count': len(self.quality_analyzer.detect_anomalies(metrics_df))
            }
            
        # Logs quality
        if not logs_df.empty:
            quality_analysis['logs'] = {
                'total_entries': len(logs_df),
                'missing_messages': logs_df['message'].isna().sum(),
                'missing_services': logs_df['service'].isna().sum(),
                'level_distribution': logs_df['level'].value_counts().to_dict()
            }
            
        # Traces quality
        if not traces_df.empty:
            quality_analysis['traces'] = {
                'total_spans': len(traces_df),
                'unique_traces': traces_df['trace_id'].nunique(),
                'missing_durations': traces_df['duration_ms'].isna().sum(),
                'error_spans': traces_df['error'].sum(),
                'orphaned_spans': traces_df['parent_span_id'].isna().sum()
            }
            
        return quality_analysis
        
    def generate_cluster_info(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate cluster_info.json from collected data.
        
        Args:
            data: Dict with collected data DataFrames
            
        Returns:
            Cluster info dictionary
        """
        metrics_df = data.get('metrics', pd.DataFrame())
        logs_df = data.get('logs', pd.DataFrame())
        traces_df = data.get('traces', pd.DataFrame())
        
        # Extract log templates
        log_templates = {}
        if not logs_df.empty:
            log_templates = self.logs_collector.extract_log_templates(logs_df)
            
        # Generate cluster info
        cluster_info = self.cluster_info_generator.generate_cluster_info(
            logs_df, metrics_df, traces_df, log_templates
        )
        
        return cluster_info
        
    def export_data_for_re2(self, data: Dict[str, pd.DataFrame], 
                          output_dir: Path, experiment_name: str) -> Dict[str, str]:
        """
        Export data in RE2-compatible format.
        
        Args:
            data: Dict with collected data DataFrames
            output_dir: Output directory path
            experiment_name: Name of the experiment
            
        Returns:
            Dict with exported file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        metrics_df = data.get('metrics', pd.DataFrame())
        logs_df = data.get('logs', pd.DataFrame())
        traces_df = data.get('traces', pd.DataFrame())
        
        # Export metrics
        if not metrics_df.empty:
            metrics_path = output_dir / 'metrics.csv'
            metrics_df.to_csv(metrics_path, index=False)
            exported_files['metrics'] = str(metrics_path)
            
        # Export logs
        if not logs_df.empty:
            logs_path = output_dir / 'logs.csv'
            logs_df.to_csv(logs_path, index=False)
            exported_files['logs'] = str(logs_path)
            
            # Export log timestamps separately
            logts_df = logs_df[['timestamp', 'service']].copy()
            logts_path = output_dir / 'logts.csv'
            logts_df.to_csv(logts_path, index=False)
            exported_files['logts'] = str(logts_path)
            
        # Export traces
        if not traces_df.empty:
            traces_path = output_dir / 'traces.csv'
            traces_df.to_csv(traces_path, index=False)
            exported_files['traces'] = str(traces_path)
            
            # Export trace error and latency data
            traces_main, error_traces, latency_traces = self.traces_collector.export_for_csv(traces_df)
            
            if not error_traces.empty:
                tracets_err_path = output_dir / 'tracets_err.csv'
                error_traces.to_csv(tracets_err_path, index=False)
                exported_files['tracets_err'] = str(tracets_err_path)
                
            if not latency_traces.empty:
                tracets_lat_path = output_dir / 'tracets_lat.csv'
                latency_traces.to_csv(tracets_lat_path, index=False)
                exported_files['tracets_lat'] = str(tracets_lat_path)
                
        # Generate and export cluster info
        cluster_info = self.generate_cluster_info(data)
        cluster_info_path = output_dir / 'cluster_info.json'
        self.cluster_info_generator.save_cluster_info(cluster_info, str(cluster_info_path))
        exported_files['cluster_info'] = str(cluster_info_path)
        
        logger.info(f"Data exported to {output_dir}. Files: {list(exported_files.keys())}")
        
        return exported_files