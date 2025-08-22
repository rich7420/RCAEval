"""
Configuration validator for experiment configurations.
Implements advanced validation logic and safety checks.
"""

import re
import requests
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
from urllib.parse import urlparse

from .experiment_config_parser import ExperimentConfig

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Advanced validator for experiment configurations."""
    
    def __init__(self):
        """Initialize configuration validator."""
        self.fault_type_requirements = {
            'cpu': ['target_services'],
            'memory': ['target_services'],
            'disk': ['target_services', 'path'],
            'network_delay': ['target_services', 'delay_ms'],
            'network_loss': ['target_services', 'loss_percent'],
            'socket': ['target_services', 'port']
        }
        
        self.intensity_parameters = {
            'cpu': {
                'low': {'cpu_percent': 25},
                'medium': {'cpu_percent': 50},
                'high': {'cpu_percent': 80}
            },
            'memory': {
                'low': {'memory_percent': 25},
                'medium': {'memory_percent': 50},
                'high': {'memory_percent': 75}
            },
            'disk': {
                'low': {'io_percent': 25},
                'medium': {'io_percent': 50},
                'high': {'io_percent': 80}
            },
            'network_delay': {
                'low': {'delay_ms': 50},
                'medium': {'delay_ms': 100},
                'high': {'delay_ms': 200}
            },
            'network_loss': {
                'low': {'loss_percent': 5},
                'medium': {'loss_percent': 10},
                'high': {'loss_percent': 20}
            }
        }
    
    def validate_comprehensive(self, config: ExperimentConfig, 
                             check_connectivity: bool = True) -> Tuple[List[str], List[str]]:
        """
        Perform comprehensive validation of experiment configuration.
        
        Args:
            config: ExperimentConfig to validate
            check_connectivity: Whether to check external service connectivity
            
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        
        # Basic validation
        basic_errors = self._validate_basic_structure(config)
        errors.extend(basic_errors)
        
        # Validate chaos configuration
        chaos_errors, chaos_warnings = self._validate_chaos_config(config)
        errors.extend(chaos_errors)
        warnings.extend(chaos_warnings)
        
        # Validate data collection configuration
        dc_errors, dc_warnings = self._validate_data_collection_config(config)
        errors.extend(dc_errors)
        warnings.extend(dc_warnings)
        
        # Validate traffic configuration
        traffic_errors, traffic_warnings = self._validate_traffic_config(config)
        errors.extend(traffic_errors)
        warnings.extend(traffic_warnings)
        
        # Validate export configuration
        export_errors, export_warnings = self._validate_export_config(config)
        errors.extend(export_errors)
        warnings.extend(export_warnings)
        
        # Validate timing consistency
        timing_errors, timing_warnings = self._validate_timing_consistency(config)
        errors.extend(timing_errors)
        warnings.extend(timing_warnings)
        
        # Check connectivity if requested
        if check_connectivity:
            conn_errors, conn_warnings = self._validate_connectivity(config)
            errors.extend(conn_errors)
            warnings.extend(conn_warnings)
        
        return errors, warnings
    
    def _validate_basic_structure(self, config: ExperimentConfig) -> List[str]:
        """Validate basic configuration structure."""
        errors = []
        
        # Required fields
        if not config.name or not config.name.strip():
            errors.append("Experiment name is required and cannot be empty")
        
        # Name format validation
        if config.name and not re.match(r'^[a-zA-Z0-9_-]+$', config.name):
            errors.append("Experiment name can only contain letters, numbers, underscores, and hyphens")
        
        if len(config.name) > 50:
            errors.append("Experiment name cannot exceed 50 characters")
        
        # Description validation
        if len(config.description) > 500:
            errors.append("Experiment description cannot exceed 500 characters")
        
        return errors
    
    def _validate_chaos_config(self, config: ExperimentConfig) -> Tuple[List[str], List[str]]:
        """Validate chaos configuration."""
        errors = []
        warnings = []
        
        chaos = config.chaos
        
        # Validate fault type requirements
        required_params = self.fault_type_requirements.get(chaos.fault_type, [])
        for param in required_params:
            if param == 'target_services':
                if not chaos.target_services:
                    errors.append(f"target_services is required for {chaos.fault_type} fault")
            elif param not in chaos.parameters:
                errors.append(f"Parameter '{param}' is required for {chaos.fault_type} fault")
        
        # Validate duration
        if chaos.duration < 30:
            warnings.append("Chaos duration less than 30 seconds may not provide meaningful results")
        elif chaos.duration > 1800:  # 30 minutes
            warnings.append("Chaos duration over 30 minutes may cause significant service disruption")
        
        # Validate intensity parameters
        if chaos.fault_type in self.intensity_parameters:
            expected_params = self.intensity_parameters[chaos.fault_type][chaos.intensity]
            for param, default_value in expected_params.items():
                if param not in chaos.parameters:
                    warnings.append(f"Using default {param}={default_value} for {chaos.intensity} intensity")
                    chaos.parameters[param] = default_value
        
        # Validate specific fault type parameters
        if chaos.fault_type == 'network_delay':
            delay_ms = chaos.parameters.get('delay_ms', 0)
            if delay_ms > 1000:
                warnings.append("Network delay over 1000ms may cause timeouts")
        
        elif chaos.fault_type == 'network_loss':
            loss_percent = chaos.parameters.get('loss_percent', 0)
            if loss_percent > 50:
                errors.append("Network loss over 50% may cause complete service failure")
        
        elif chaos.fault_type == 'disk':
            path = chaos.parameters.get('path', '/')
            if not Path(path).exists():
                warnings.append(f"Disk path '{path}' may not exist on target systems")
        
        return errors, warnings
    
    def _validate_data_collection_config(self, config: ExperimentConfig) -> Tuple[List[str], List[str]]:
        """Validate data collection configuration."""
        errors = []
        warnings = []
        
        dc = config.data_collection
        
        # Validate URLs
        for url_name, url in [
            ('prometheus_url', dc.prometheus_url),
            ('loki_url', dc.loki_url),
            ('jaeger_url', dc.jaeger_url)
        ]:
            if not self._is_valid_url(url):
                errors.append(f"Invalid {url_name}: {url}")
        
        # Validate sampling interval
        if dc.sampling_interval < 5:
            warnings.append("Sampling interval less than 5 seconds may generate excessive data")
        elif dc.sampling_interval > 60:
            warnings.append("Sampling interval over 60 seconds may miss important events")
        
        # Validate collection duration
        total_experiment_time = config.get_total_duration()
        if dc.collection_duration < total_experiment_time:
            errors.append(f"Collection duration ({dc.collection_duration}s) must be at least "
                         f"total experiment time ({total_experiment_time}s)")
        
        # Validate services filter
        if dc.services_filter:
            chaos_services = set(config.chaos.target_services)
            filter_services = set(dc.services_filter)
            if not chaos_services.intersection(filter_services):
                warnings.append("Data collection services filter doesn't include any chaos target services")
        
        return errors, warnings
    
    def _validate_traffic_config(self, config: ExperimentConfig) -> Tuple[List[str], List[str]]:
        """Validate traffic configuration."""
        errors = []
        warnings = []
        
        traffic = config.traffic
        
        if not traffic.enabled:
            warnings.append("Traffic generation is disabled - experiment may not generate realistic load")
            return errors, warnings
        
        # Validate user count and spawn rate
        if traffic.users > 1000:
            warnings.append("High user count (>1000) may overwhelm target services")
        
        if traffic.spawn_rate > 10:
            warnings.append("High spawn rate (>10 users/sec) may cause rapid load spikes")
        
        # Validate duration
        if traffic.duration < config.chaos.duration:
            warnings.append("Traffic duration is shorter than chaos duration")
        
        # Validate target services
        if traffic.target_services:
            chaos_services = set(config.chaos.target_services)
            traffic_services = set(traffic.target_services)
            
            if not chaos_services.intersection(traffic_services):
                warnings.append("Traffic and chaos target different services - may not show fault impact")
        
        return errors, warnings
    
    def _validate_export_config(self, config: ExperimentConfig) -> Tuple[List[str], List[str]]:
        """Validate export configuration."""
        errors = []
        warnings = []
        
        export = config.export
        
        # Validate output directory
        output_path = Path(export.output_directory)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            errors.append(f"No write permission for output directory: {export.output_directory}")
        except Exception as e:
            errors.append(f"Cannot create output directory: {e}")
        
        # Validate export format
        if export.export_format == 're2' and export.compress_output:
            warnings.append("RE2 format with compression may not be compatible with all tools")
        
        return errors, warnings
    
    def _validate_timing_consistency(self, config: ExperimentConfig) -> Tuple[List[str], List[str]]:
        """Validate timing consistency across all components."""
        errors = []
        warnings = []
        
        # Calculate all timing components
        pre_chaos = config.data_collection.pre_chaos_duration
        chaos_duration = config.chaos.duration
        post_chaos = config.data_collection.post_chaos_duration
        collection_duration = config.data_collection.collection_duration
        traffic_duration = config.traffic.duration if config.traffic.enabled else 0
        
        total_experiment = pre_chaos + chaos_duration + post_chaos
        
        # Validate collection duration covers experiment
        if collection_duration < total_experiment:
            errors.append(f"Collection duration ({collection_duration}s) is less than "
                         f"total experiment duration ({total_experiment}s)")
        
        # Validate pre-chaos period
        if pre_chaos < 30:
            warnings.append("Pre-chaos period less than 30 seconds may not establish baseline")
        
        # Validate post-chaos period
        if post_chaos < 30:
            warnings.append("Post-chaos period less than 30 seconds may not capture recovery")
        
        # Validate traffic timing
        if config.traffic.enabled and traffic_duration > 0:
            if traffic_duration < total_experiment:
                warnings.append("Traffic duration is shorter than total experiment duration")
        
        return errors, warnings
    
    def _validate_connectivity(self, config: ExperimentConfig) -> Tuple[List[str], List[str]]:
        """Validate connectivity to external services."""
        errors = []
        warnings = []
        
        dc = config.data_collection
        
        # Test Prometheus connectivity
        try:
            response = requests.get(f"{dc.prometheus_url}/api/v1/status/config", timeout=5)
            if response.status_code != 200:
                warnings.append(f"Prometheus at {dc.prometheus_url} returned status {response.status_code}")
        except requests.exceptions.RequestException:
            warnings.append(f"Cannot connect to Prometheus at {dc.prometheus_url}")
        
        # Test Loki connectivity
        try:
            response = requests.get(f"{dc.loki_url}/ready", timeout=5)
            if response.status_code != 200:
                warnings.append(f"Loki at {dc.loki_url} returned status {response.status_code}")
        except requests.exceptions.RequestException:
            warnings.append(f"Cannot connect to Loki at {dc.loki_url}")
        
        # Test Jaeger connectivity
        try:
            response = requests.get(f"{dc.jaeger_url}/api/services", timeout=5)
            if response.status_code != 200:
                warnings.append(f"Jaeger at {dc.jaeger_url} returned status {response.status_code}")
        except requests.exceptions.RequestException:
            warnings.append(f"Cannot connect to Jaeger at {dc.jaeger_url}")
        
        return errors, warnings
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def validate_service_names(self, service_names: List[str]) -> List[str]:
        """
        Validate service names format.
        
        Args:
            service_names: List of service names to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        for service_name in service_names:
            if not service_name or not service_name.strip():
                errors.append("Service name cannot be empty")
                continue
            
            # Check format (Kubernetes service name rules)
            if not re.match(r'^[a-z0-9]([-a-z0-9]*[a-z0-9])?$', service_name):
                errors.append(f"Invalid service name '{service_name}': must be lowercase alphanumeric with hyphens")
            
            if len(service_name) > 63:
                errors.append(f"Service name '{service_name}' exceeds 63 characters")
        
        return errors
    
    def suggest_improvements(self, config: ExperimentConfig) -> List[str]:
        """
        Suggest configuration improvements.
        
        Args:
            config: ExperimentConfig to analyze
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Timing suggestions
        if config.data_collection.pre_chaos_duration < 60:
            suggestions.append("Consider increasing pre-chaos duration to 60+ seconds for better baseline")
        
        if config.data_collection.post_chaos_duration < 60:
            suggestions.append("Consider increasing post-chaos duration to 60+ seconds to observe recovery")
        
        # Traffic suggestions
        if config.traffic.enabled and config.traffic.users < 5:
            suggestions.append("Consider increasing user count for more realistic load testing")
        
        # Chaos suggestions
        if config.chaos.intensity == 'high' and config.chaos.duration > 600:
            suggestions.append("High intensity chaos for >10 minutes may cause excessive disruption")
        
        # Data collection suggestions
        if not config.data_collection.services_filter:
            suggestions.append("Consider adding services_filter to reduce data collection overhead")
        
        # Export suggestions
        if not config.export.compress_output and config.chaos.duration > 600:
            suggestions.append("Consider enabling compression for long experiments to save disk space")
        
        return suggestions