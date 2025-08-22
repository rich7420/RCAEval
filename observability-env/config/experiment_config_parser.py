"""
Experiment configuration parser for YAML/JSON configuration files.
Implements requirements 6.1, 6.2, 6.3, 6.4, 6.5 for configuration parsing and validation.
"""

import yaml
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ChaosConfig:
    """Configuration for chaos injection parameters."""
    fault_type: str
    duration: int  # seconds
    intensity: str  # low, medium, high
    target_services: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate chaos configuration after initialization."""
        valid_fault_types = ['cpu', 'memory', 'disk', 'network_delay', 'network_loss', 'socket']
        if self.fault_type not in valid_fault_types:
            raise ValueError(f"Invalid fault_type: {self.fault_type}. Must be one of {valid_fault_types}")
        
        valid_intensities = ['low', 'medium', 'high']
        if self.intensity not in valid_intensities:
            raise ValueError(f"Invalid intensity: {self.intensity}. Must be one of {valid_intensities}")
        
        if self.duration <= 0:
            raise ValueError("Duration must be positive")
        
        if not self.target_services:
            raise ValueError("Target services cannot be empty")


@dataclass
class DataCollectionConfig:
    """Configuration for data collection parameters."""
    collection_duration: int  # seconds
    pre_chaos_duration: int = 60  # seconds
    post_chaos_duration: int = 60  # seconds
    sampling_interval: int = 15  # seconds
    
    # Data source configurations
    prometheus_url: str = "http://localhost:9090"
    loki_url: str = "http://localhost:3100"
    jaeger_url: str = "http://localhost:16686"
    
    # Collection filters
    services_filter: Optional[List[str]] = None
    metrics_filter: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate data collection configuration."""
        if self.collection_duration <= 0:
            raise ValueError("Collection duration must be positive")
        if self.pre_chaos_duration < 0:
            raise ValueError("Pre-chaos duration cannot be negative")
        if self.post_chaos_duration < 0:
            raise ValueError("Post-chaos duration cannot be negative")
        if self.sampling_interval <= 0:
            raise ValueError("Sampling interval must be positive")


@dataclass
class TrafficConfig:
    """Configuration for traffic generation."""
    enabled: bool = True
    traffic_type: str = "normal"  # normal, spike, burst
    users: int = 10
    spawn_rate: float = 1.0  # users per second
    duration: int = 300  # seconds
    target_services: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate traffic configuration."""
        valid_traffic_types = ['normal', 'spike', 'burst', 'custom']
        if self.traffic_type not in valid_traffic_types:
            raise ValueError(f"Invalid traffic_type: {self.traffic_type}. Must be one of {valid_traffic_types}")
        
        if self.users <= 0:
            raise ValueError("Number of users must be positive")
        if self.spawn_rate <= 0:
            raise ValueError("Spawn rate must be positive")
        if self.duration <= 0:
            raise ValueError("Duration must be positive")


@dataclass
class ExportConfig:
    """Configuration for data export."""
    output_directory: str = "data/collected"
    export_format: str = "re2"  # re2, csv, json
    compress_output: bool = False
    include_raw_data: bool = False
    
    def __post_init__(self):
        """Validate export configuration."""
        valid_formats = ['re2', 'csv', 'json']
        if self.export_format not in valid_formats:
            raise ValueError(f"Invalid export_format: {self.export_format}. Must be one of {valid_formats}")


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str
    description: str
    chaos: ChaosConfig
    data_collection: DataCollectionConfig
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    
    # Experiment metadata
    experiment_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Set default values and validate configuration."""
        if self.created_at is None:
            self.created_at = datetime.now()
        
        if self.experiment_id is None:
            # Generate experiment ID from name and timestamp
            timestamp = self.created_at.strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(c for c in self.name if c.isalnum() or c in "_-").lower()
            self.experiment_id = f"{safe_name}_{timestamp}"
    
    def get_total_duration(self) -> int:
        """Get total experiment duration including pre/post chaos periods."""
        return (self.data_collection.pre_chaos_duration + 
                self.chaos.duration + 
                self.data_collection.post_chaos_duration)
    
    def get_chaos_start_time(self) -> int:
        """Get relative time when chaos injection starts."""
        return self.data_collection.pre_chaos_duration
    
    def get_chaos_end_time(self) -> int:
        """Get relative time when chaos injection ends."""
        return self.data_collection.pre_chaos_duration + self.chaos.duration


class ExperimentConfigParser:
    """Parser for experiment configuration files (YAML/JSON)."""
    
    def __init__(self):
        """Initialize configuration parser."""
        self.supported_formats = ['.yaml', '.yml', '.json']
    
    def parse_file(self, config_path: Union[str, Path]) -> ExperimentConfig:
        """
        Parse experiment configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            ExperimentConfig object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config format is invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if config_path.suffix not in self.supported_formats:
            raise ValueError(f"Unsupported config format: {config_path.suffix}. "
                           f"Supported formats: {self.supported_formats}")
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix == '.json':
                    config_data = json.load(f)
                else:  # YAML
                    config_data = yaml.safe_load(f)
            
            logger.info(f"Loaded configuration from {config_path}")
            return self.parse_dict(config_data)
            
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to parse configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")
    
    def parse_dict(self, config_data: Dict[str, Any]) -> ExperimentConfig:
        """
        Parse experiment configuration from dictionary.
        
        Args:
            config_data: Configuration dictionary
            
        Returns:
            ExperimentConfig object
        """
        try:
            # Parse chaos configuration
            chaos_data = config_data.get('chaos', {})
            chaos_config = ChaosConfig(
                fault_type=chaos_data.get('fault_type'),
                duration=chaos_data.get('duration'),
                intensity=chaos_data.get('intensity'),
                target_services=chaos_data.get('target_services', []),
                parameters=chaos_data.get('parameters', {})
            )
            
            # Parse data collection configuration
            data_collection_data = config_data.get('data_collection', {})
            data_collection_config = DataCollectionConfig(
                collection_duration=data_collection_data.get('collection_duration', 300),
                pre_chaos_duration=data_collection_data.get('pre_chaos_duration', 60),
                post_chaos_duration=data_collection_data.get('post_chaos_duration', 60),
                sampling_interval=data_collection_data.get('sampling_interval', 15),
                prometheus_url=data_collection_data.get('prometheus_url', 'http://localhost:9090'),
                loki_url=data_collection_data.get('loki_url', 'http://localhost:3100'),
                jaeger_url=data_collection_data.get('jaeger_url', 'http://localhost:16686'),
                services_filter=data_collection_data.get('services_filter'),
                metrics_filter=data_collection_data.get('metrics_filter')
            )
            
            # Parse traffic configuration
            traffic_data = config_data.get('traffic', {})
            traffic_config = TrafficConfig(
                enabled=traffic_data.get('enabled', True),
                traffic_type=traffic_data.get('traffic_type', 'normal'),
                users=traffic_data.get('users', 10),
                spawn_rate=traffic_data.get('spawn_rate', 1.0),
                duration=traffic_data.get('duration', 300),
                target_services=traffic_data.get('target_services')
            )
            
            # Parse export configuration
            export_data = config_data.get('export', {})
            export_config = ExportConfig(
                output_directory=export_data.get('output_directory', 'data/collected'),
                export_format=export_data.get('export_format', 're2'),
                compress_output=export_data.get('compress_output', False),
                include_raw_data=export_data.get('include_raw_data', False)
            )
            
            # Create experiment configuration
            experiment_config = ExperimentConfig(
                name=config_data.get('name'),
                description=config_data.get('description', ''),
                chaos=chaos_config,
                data_collection=data_collection_config,
                traffic=traffic_config,
                export=export_config,
                experiment_id=config_data.get('experiment_id'),
                tags=config_data.get('tags', [])
            )
            
            logger.info(f"Parsed experiment configuration: {experiment_config.name}")
            return experiment_config
            
        except KeyError as e:
            raise ValueError(f"Missing required configuration key: {e}")
        except Exception as e:
            raise ValueError(f"Error parsing configuration: {e}")
    
    def validate_config(self, config: ExperimentConfig) -> List[str]:
        """
        Validate experiment configuration.
        
        Args:
            config: ExperimentConfig to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate required fields
        if not config.name:
            errors.append("Experiment name is required")
        
        if not config.chaos.fault_type:
            errors.append("Chaos fault_type is required")
        
        if not config.chaos.target_services:
            errors.append("Chaos target_services cannot be empty")
        
        # Validate timing consistency
        total_collection = config.data_collection.collection_duration
        total_experiment = config.get_total_duration()
        
        if total_collection < total_experiment:
            errors.append(f"Collection duration ({total_collection}s) is less than "
                         f"total experiment duration ({total_experiment}s)")
        
        # Validate traffic configuration
        if config.traffic.enabled and config.traffic.duration < config.chaos.duration:
            errors.append("Traffic duration should be at least as long as chaos duration")
        
        # Validate service consistency
        if config.traffic.target_services:
            chaos_services = set(config.chaos.target_services)
            traffic_services = set(config.traffic.target_services)
            if not chaos_services.intersection(traffic_services):
                errors.append("Chaos and traffic target services should have some overlap")
        
        return errors
    
    def save_config(self, config: ExperimentConfig, output_path: Union[str, Path], 
                   format: str = 'yaml') -> None:
        """
        Save experiment configuration to file.
        
        Args:
            config: ExperimentConfig to save
            output_path: Output file path
            format: Output format ('yaml' or 'json')
        """
        output_path = Path(output_path)
        
        # Convert config to dictionary
        config_dict = self._config_to_dict(config)
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                if format.lower() == 'json':
                    json.dump(config_dict, f, indent=2, default=str)
                else:  # YAML
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved configuration to {output_path}")
            
        except Exception as e:
            raise ValueError(f"Failed to save configuration: {e}")
    
    def _config_to_dict(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Convert ExperimentConfig to dictionary."""
        return {
            'name': config.name,
            'description': config.description,
            'experiment_id': config.experiment_id,
            'tags': config.tags,
            'created_at': config.created_at.isoformat() if config.created_at else None,
            'chaos': {
                'fault_type': config.chaos.fault_type,
                'duration': config.chaos.duration,
                'intensity': config.chaos.intensity,
                'target_services': config.chaos.target_services,
                'parameters': config.chaos.parameters
            },
            'data_collection': {
                'collection_duration': config.data_collection.collection_duration,
                'pre_chaos_duration': config.data_collection.pre_chaos_duration,
                'post_chaos_duration': config.data_collection.post_chaos_duration,
                'sampling_interval': config.data_collection.sampling_interval,
                'prometheus_url': config.data_collection.prometheus_url,
                'loki_url': config.data_collection.loki_url,
                'jaeger_url': config.data_collection.jaeger_url,
                'services_filter': config.data_collection.services_filter,
                'metrics_filter': config.data_collection.metrics_filter
            },
            'traffic': {
                'enabled': config.traffic.enabled,
                'traffic_type': config.traffic.traffic_type,
                'users': config.traffic.users,
                'spawn_rate': config.traffic.spawn_rate,
                'duration': config.traffic.duration,
                'target_services': config.traffic.target_services
            },
            'export': {
                'output_directory': config.export.output_directory,
                'export_format': config.export.export_format,
                'compress_output': config.export.compress_output,
                'include_raw_data': config.export.include_raw_data
            }
        }
    
    def create_default_config(self, name: str, fault_type: str, 
                            target_services: List[str]) -> ExperimentConfig:
        """
        Create a default experiment configuration.
        
        Args:
            name: Experiment name
            fault_type: Type of chaos fault
            target_services: List of target services
            
        Returns:
            ExperimentConfig with default values
        """
        chaos_config = ChaosConfig(
            fault_type=fault_type,
            duration=300,  # 5 minutes
            intensity='medium',
            target_services=target_services
        )
        
        data_collection_config = DataCollectionConfig(
            collection_duration=420,  # 7 minutes (60 + 300 + 60)
            pre_chaos_duration=60,
            post_chaos_duration=60,
            sampling_interval=15
        )
        
        return ExperimentConfig(
            name=name,
            description=f"Default {fault_type} chaos experiment for {', '.join(target_services)}",
            chaos=chaos_config,
            data_collection=data_collection_config
        )