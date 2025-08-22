"""
Data export manager for coordinating all RE2-compatible data export operations.
Integrates all export modules for comprehensive data export workflow.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
import os
import json
from pathlib import Path

from .metrics_csv_exporter import MetricsCSVExporter
from .logs_csv_exporter import LogsCSVExporter
from .traces_csv_exporter import TracesCSVExporter
from .directory_organizer import DirectoryOrganizer
from .injection_timestamp_recorder import InjectionTimestampRecorder
from .cluster_info_generator import ClusterInfoGenerator

logger = logging.getLogger(__name__)


class DataExportManager:
    """Comprehensive data export manager for RE2-compatible datasets."""
    
    def __init__(self, base_output_dir: str = "data/collected"):
        """
        Initialize data export manager.
        
        Args:
            base_output_dir: Base directory for exported data
        """
        self.base_output_dir = base_output_dir
        
        # Initialize all exporters
        self.metrics_exporter = MetricsCSVExporter()
        self.logs_exporter = LogsCSVExporter()
        self.traces_exporter = TracesCSVExporter()
        self.directory_organizer = DirectoryOrganizer(base_output_dir)
        self.timestamp_recorder = InjectionTimestampRecorder(base_output_dir)
        self.cluster_info_generator = ClusterInfoGenerator()
        
    def export_complete_experiment(self, 
                                 metrics_df: pd.DataFrame,
                                 logs_df: pd.DataFrame,
                                 traces_df: pd.DataFrame,
                                 service: str,
                                 fault_type: str,
                                 experiment_number: Optional[int] = None,
                                 injection_timestamp: Optional[int] = None,
                                 metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Export complete experiment data in RE2 format.
        
        Args:
            metrics_df: DataFrame with metrics data
            logs_df: DataFrame with logs data
            traces_df: DataFrame with traces data
            service: Service name
            fault_type: Fault type
            experiment_number: Experiment number (auto-generated if None)
            injection_timestamp: Chaos injection timestamp
            metadata: Optional experiment metadata
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Get experiment number if not provided
            if experiment_number is None:
                experiment_number = self.directory_organizer.get_next_experiment_number(service, fault_type)
                
            logger.info(f"Exporting complete experiment: {service}_{fault_type}/{experiment_number}")
            
            # Create experiment directory
            experiment_dir = self.directory_organizer.create_re2_structure(
                service, fault_type, experiment_number
            )
            
            # Export all data formats
            success_flags = []
            
            # 1. Export metrics.csv
            metrics_path = os.path.join(experiment_dir, 'metrics.csv')
            metrics_success = self.metrics_exporter.export_to_csv(metrics_df, metrics_path)
            success_flags.append(metrics_success)
            
            # 2. Export logs.csv and logts.csv
            logs_path = os.path.join(experiment_dir, 'logs.csv')
            logts_path = os.path.join(experiment_dir, 'logts.csv')
            logs_success = self.logs_exporter.export_logs_csv(logs_df, logs_path)
            logts_success = self.logs_exporter.export_logts_csv(logs_df, logts_path)
            success_flags.extend([logs_success, logts_success])
            
            # 3. Export traces.csv, tracets_err.csv, tracets_lat.csv
            traces_path = os.path.join(experiment_dir, 'traces.csv')
            tracets_err_path = os.path.join(experiment_dir, 'tracets_err.csv')
            tracets_lat_path = os.path.join(experiment_dir, 'tracets_lat.csv')
            traces_success = self.traces_exporter.export_traces_csv(traces_df, traces_path)
            err_success = self.traces_exporter.export_tracets_err_csv(traces_df, tracets_err_path)
            lat_success = self.traces_exporter.export_tracets_lat_csv(traces_df, tracets_lat_path)
            success_flags.extend([traces_success, err_success, lat_success])
            
            # 4. Generate cluster_info.json
            cluster_info_path = os.path.join(experiment_dir, 'cluster_info.json')
            
            # Extract log templates first
            log_templates = {}
            if not logs_df.empty:
                try:
                    from ..logs.loki_client import LogTemplateExtractor
                    template_extractor = LogTemplateExtractor()
                    extracted_templates = template_extractor.extract_templates(logs_df['message'].tolist())
                    
                    # Ensure log_templates is a dict with service as key
                    if isinstance(extracted_templates, dict):
                        # Group by service
                        for service in logs_df['service'].unique():
                            service_messages = logs_df[logs_df['service'] == service]['message'].tolist()
                            service_templates = template_extractor.extract_templates(service_messages)
                            if service_templates:
                                log_templates[service] = service_templates
                    else:
                        log_templates = {}
                except Exception as e:
                    logger.warning(f"Failed to extract log templates: {e}")
                    log_templates = {}
            
            # Generate cluster info
            cluster_info_data = self.cluster_info_generator.generate_cluster_info(
                logs_df, metrics_df, traces_df, log_templates
            )
            
            # Write to file
            try:
                with open(cluster_info_path, 'w') as f:
                    json.dump(cluster_info_data, f, indent=2)
                cluster_info_success = True
                logger.info(f"Generated cluster_info.json: {cluster_info_path}")
            except Exception as e:
                logger.error(f"Failed to write cluster_info.json: {e}")
                cluster_info_success = False
            success_flags.append(cluster_info_success)
            
            # 5. Create inject_time.txt if injection timestamp provided
            if injection_timestamp:
                inject_time_path = os.path.join(experiment_dir, 'inject_time.txt')
                with open(inject_time_path, 'w') as f:
                    f.write(str(injection_timestamp))
                logger.info(f"Created inject_time.txt with timestamp: {injection_timestamp}")
            
            # 6. Create experiment metadata
            if metadata:
                self.directory_organizer.create_experiment_metadata(
                    service, fault_type, experiment_number, metadata
                )
                
            # Check overall success
            overall_success = any(success_flags)  # At least one export should succeed
            
            if overall_success:
                logger.info(f"Successfully exported experiment {service}_{fault_type}/{experiment_number}")
                logger.info(f"Export success rates: metrics={metrics_success}, logs={logs_success}, "
                          f"logts={logts_success}, traces={traces_success}, err={err_success}, "
                          f"lat={lat_success}, cluster_info={cluster_info_success}")
            else:
                logger.error(f"Failed to export experiment {service}_{fault_type}/{experiment_number}")
                
            return overall_success
            
        except Exception as e:
            logger.error(f"Failed to export complete experiment: {e}")
            return False
            
    def export_experiment_from_collectors(self,
                                        prometheus_url: str,
                                        loki_url: str,
                                        jaeger_url: str,
                                        service: str,
                                        fault_type: str,
                                        start_time: datetime,
                                        end_time: datetime,
                                        experiment_number: Optional[int] = None,
                                        injection_timestamp: Optional[int] = None,
                                        services_filter: Optional[List[str]] = None) -> bool:
        """
        Export experiment data by collecting from observability backends.
        
        Args:
            prometheus_url: Prometheus server URL
            loki_url: Loki server URL
            jaeger_url: Jaeger server URL
            service: Target service name
            fault_type: Fault type
            start_time: Data collection start time
            end_time: Data collection end time
            experiment_number: Experiment number (auto-generated if None)
            injection_timestamp: Chaos injection timestamp
            services_filter: Optional list of services to collect
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            from ..metrics.prometheus_client import MetricsCollector
            from ..logs.loki_client import LogsCollector
            from ..traces.jaeger_client import TracesCollector
            
            logger.info(f"Collecting data for experiment {service}_{fault_type}")
            
            # Collect data from all sources
            metrics_collector = MetricsCollector(prometheus_url)
            logs_collector = LogsCollector(loki_url)
            traces_collector = TracesCollector(jaeger_url)
            
            # Collect metrics
            logger.info("Collecting metrics data...")
            metrics_df = metrics_collector.collect_system_metrics(
                start_time, end_time, services_filter
            )
            
            # Collect logs
            logger.info("Collecting logs data...")
            logs_df = logs_collector.collect_service_logs(
                start_time, end_time, services_filter
            )
            
            # Collect traces
            logger.info("Collecting traces data...")
            traces_df = traces_collector.collect_service_traces(
                start_time, end_time, services_filter
            )
            
            # Create metadata
            metadata = {
                'collection_start_time': start_time.isoformat(),
                'collection_end_time': end_time.isoformat(),
                'prometheus_url': prometheus_url,
                'loki_url': loki_url,
                'jaeger_url': jaeger_url,
                'services_filter': services_filter,
                'data_summary': {
                    'metrics_records': len(metrics_df),
                    'logs_records': len(logs_df),
                    'traces_records': len(traces_df)
                }
            }
            
            # Export complete experiment
            return self.export_complete_experiment(
                metrics_df, logs_df, traces_df,
                service, fault_type, experiment_number,
                injection_timestamp, metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to export experiment from collectors: {e}")
            return False
            
    def batch_export_experiments(self, experiments_config: List[Dict[str, Any]]) -> Dict[str, bool]:
        """
        Export multiple experiments in batch.
        
        Args:
            experiments_config: List of experiment configurations
            
        Returns:
            Dict mapping experiment names to success status
        """
        results = {}
        
        for config in experiments_config:
            try:
                experiment_name = f"{config['service']}_{config['fault_type']}/{config.get('experiment_number', 'auto')}"
                logger.info(f"Processing batch experiment: {experiment_name}")
                
                if 'dataframes' in config:
                    # Export from provided DataFrames
                    success = self.export_complete_experiment(
                        config['dataframes']['metrics'],
                        config['dataframes']['logs'],
                        config['dataframes']['traces'],
                        config['service'],
                        config['fault_type'],
                        config.get('experiment_number'),
                        config.get('injection_timestamp'),
                        config.get('metadata')
                    )
                elif 'collectors' in config:
                    # Export from collectors
                    collectors = config['collectors']
                    success = self.export_experiment_from_collectors(
                        collectors['prometheus_url'],
                        collectors['loki_url'],
                        collectors['jaeger_url'],
                        config['service'],
                        config['fault_type'],
                        config['start_time'],
                        config['end_time'],
                        config.get('experiment_number'),
                        config.get('injection_timestamp'),
                        config.get('services_filter')
                    )
                else:
                    logger.error(f"Invalid experiment config: {experiment_name}")
                    success = False
                    
                results[experiment_name] = success
                
            except Exception as e:
                logger.error(f"Failed to process batch experiment {experiment_name}: {e}")
                results[experiment_name] = False
                
        return results
        
    def validate_exported_experiment(self, service: str, fault_type: str, 
                                   experiment_number: int) -> Dict[str, Any]:
        """
        Validate exported experiment data.
        
        Args:
            service: Service name
            fault_type: Fault type
            experiment_number: Experiment number
            
        Returns:
            Dict with validation results
        """
        try:
            # Validate directory structure
            structure_validation = self.directory_organizer.validate_experiment_structure(
                service, fault_type, experiment_number
            )
            
            # Get experiment directory
            service_fault_dir = f"{service}_{fault_type}"
            experiment_dir = Path(self.base_output_dir) / service_fault_dir / str(experiment_number)
            
            validation_results = {
                'structure_valid': structure_validation.get('valid', False),
                'directory_exists': structure_validation.get('directory_exists', False),
                'file_validations': {}
            }
            
            # Validate individual CSV files
            csv_files = ['metrics.csv', 'logs.csv', 'logts.csv', 'traces.csv', 'tracets_err.csv', 'tracets_lat.csv']
            
            for csv_file in csv_files:
                csv_path = experiment_dir / csv_file
                if csv_path.exists():
                    if 'metrics' in csv_file:
                        validation_results['file_validations'][csv_file] = self.metrics_exporter.validate_csv_format(str(csv_path))
                    elif 'log' in csv_file:
                        validation_results['file_validations'][csv_file] = self.logs_exporter.validate_logs_csv_format(str(csv_path))
                    elif 'trace' in csv_file:
                        validation_results['file_validations'][csv_file] = self.traces_exporter.validate_traces_csv_format(str(csv_path))
                else:
                    validation_results['file_validations'][csv_file] = False
                    
            # Validate inject_time.txt
            inject_time_path = experiment_dir / 'inject_time.txt'
            if inject_time_path.exists():
                validation_results['inject_time_valid'] = self.timestamp_recorder.validate_inject_time_file(str(inject_time_path))
            else:
                validation_results['inject_time_valid'] = False
                
            # Overall validation
            validation_results['overall_valid'] = (
                validation_results['structure_valid'] and
                any(validation_results['file_validations'].values())
            )
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate exported experiment: {e}")
            return {'overall_valid': False, 'error': str(e)}
            
    def get_export_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive export summary.
        
        Returns:
            Dict with export summary
        """
        try:
            # Get directory summary
            directory_summary = self.directory_organizer.get_directory_summary()
            
            # Get injection summary
            injection_summary = self.timestamp_recorder.get_injection_summary()
            
            # Combine summaries
            export_summary = {
                'base_output_directory': self.base_output_dir,
                'directory_structure': directory_summary,
                'injection_timing': injection_summary,
                'export_capabilities': {
                    'metrics_formats': ['metrics.csv'],
                    'logs_formats': ['logs.csv', 'logts.csv'],
                    'traces_formats': ['traces.csv', 'tracets_err.csv', 'tracets_lat.csv'],
                    'metadata_formats': ['cluster_info.json', 'inject_time.txt', 'experiment_metadata.json']
                }
            }
            
            return export_summary
            
        except Exception as e:
            logger.error(f"Failed to get export summary: {e}")
            return {}
            
    def cleanup_failed_exports(self) -> int:
        """
        Clean up failed or incomplete exports.
        
        Returns:
            Number of cleaned up experiments
        """
        try:
            cleaned_count = 0
            experiments = self.directory_organizer.list_experiments()
            
            for exp in experiments:
                validation = self.validate_exported_experiment(
                    exp['service'], exp['fault_type'], exp['experiment_number']
                )
                
                if not validation.get('overall_valid', False):
                    logger.info(f"Cleaning up failed export: {exp['service']}_{exp['fault_type']}/{exp['experiment_number']}")
                    
                    success = self.directory_organizer.cleanup_experiment(
                        exp['service'], exp['fault_type'], exp['experiment_number']
                    )
                    
                    if success:
                        cleaned_count += 1
                        
            logger.info(f"Cleaned up {cleaned_count} failed exports")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup failed exports: {e}")
            return 0