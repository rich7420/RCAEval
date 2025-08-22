"""
Configuration templates manager for creating default experiment configurations.
Provides templates for common chaos engineering scenarios.
"""

from typing import Dict, List, Any
from pathlib import Path
import yaml
import json
import logging

from .experiment_config_parser import ExperimentConfig, ChaosConfig, DataCollectionConfig, TrafficConfig, ExportConfig

logger = logging.getLogger(__name__)


class ConfigTemplateManager:
    """Manager for experiment configuration templates."""
    
    def __init__(self):
        """Initialize template manager."""
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize built-in configuration templates."""
        return {
            'cpu_stress_basic': {
                'name': 'CPU Stress Test',
                'description': 'Basic CPU stress test for microservices',
                'chaos': {
                    'fault_type': 'cpu',
                    'duration': 300,
                    'intensity': 'medium',
                    'target_services': ['frontend', 'backend'],
                    'parameters': {
                        'cpu_percent': 50
                    }
                },
                'data_collection': {
                    'collection_duration': 420,
                    'pre_chaos_duration': 60,
                    'post_chaos_duration': 60,
                    'sampling_interval': 15
                },
                'traffic': {
                    'enabled': True,
                    'traffic_type': 'normal',
                    'users': 10,
                    'spawn_rate': 1.0,
                    'duration': 360
                }
            },
            
            'memory_stress_basic': {
                'name': 'Memory Stress Test',
                'description': 'Basic memory stress test for microservices',
                'chaos': {
                    'fault_type': 'memory',
                    'duration': 300,
                    'intensity': 'medium',
                    'target_services': ['backend', 'database'],
                    'parameters': {
                        'memory_percent': 50
                    }
                },
                'data_collection': {
                    'collection_duration': 420,
                    'pre_chaos_duration': 60,
                    'post_chaos_duration': 60,
                    'sampling_interval': 15
                },
                'traffic': {
                    'enabled': True,
                    'traffic_type': 'normal',
                    'users': 15,
                    'spawn_rate': 1.5,
                    'duration': 360
                }
            },
            
            'network_delay_basic': {
                'name': 'Network Delay Test',
                'description': 'Basic network latency injection test',
                'chaos': {
                    'fault_type': 'network_delay',
                    'duration': 180,
                    'intensity': 'medium',
                    'target_services': ['frontend', 'backend'],
                    'parameters': {
                        'delay_ms': 100,
                        'jitter_ms': 20
                    }
                },
                'data_collection': {
                    'collection_duration': 300,
                    'pre_chaos_duration': 60,
                    'post_chaos_duration': 60,
                    'sampling_interval': 10
                },
                'traffic': {
                    'enabled': True,
                    'traffic_type': 'normal',
                    'users': 20,
                    'spawn_rate': 2.0,
                    'duration': 240
                }
            },
            
            'network_loss_basic': {
                'name': 'Network Packet Loss Test',
                'description': 'Basic network packet loss injection test',
                'chaos': {
                    'fault_type': 'network_loss',
                    'duration': 120,
                    'intensity': 'low',
                    'target_services': ['frontend'],
                    'parameters': {
                        'loss_percent': 5
                    }
                },
                'data_collection': {
                    'collection_duration': 240,
                    'pre_chaos_duration': 60,
                    'post_chaos_duration': 60,
                    'sampling_interval': 10
                },
                'traffic': {
                    'enabled': True,
                    'traffic_type': 'normal',
                    'users': 25,
                    'spawn_rate': 2.5,
                    'duration': 180
                }
            },
            
            'disk_stress_basic': {
                'name': 'Disk I/O Stress Test',
                'description': 'Basic disk I/O stress test',
                'chaos': {
                    'fault_type': 'disk',
                    'duration': 240,
                    'intensity': 'medium',
                    'target_services': ['database', 'backend'],
                    'parameters': {
                        'io_percent': 50,
                        'path': '/tmp'
                    }
                },
                'data_collection': {
                    'collection_duration': 360,
                    'pre_chaos_duration': 60,
                    'post_chaos_duration': 60,
                    'sampling_interval': 15
                },
                'traffic': {
                    'enabled': True,
                    'traffic_type': 'normal',
                    'users': 12,
                    'spawn_rate': 1.2,
                    'duration': 300
                }
            },
            
            'socket_failure_basic': {
                'name': 'Socket Connection Failure Test',
                'description': 'Basic socket connection failure test',
                'chaos': {
                    'fault_type': 'socket',
                    'duration': 150,
                    'intensity': 'medium',
                    'target_services': ['backend'],
                    'parameters': {
                        'port': 8080,
                        'failure_rate': 50
                    }
                },
                'data_collection': {
                    'collection_duration': 270,
                    'pre_chaos_duration': 60,
                    'post_chaos_duration': 60,
                    'sampling_interval': 10
                },
                'traffic': {
                    'enabled': True,
                    'traffic_type': 'normal',
                    'users': 18,
                    'spawn_rate': 1.8,
                    'duration': 210
                }
            },
            
            'comprehensive_test': {
                'name': 'Comprehensive Resilience Test',
                'description': 'Extended test with multiple phases and high data collection',
                'chaos': {
                    'fault_type': 'cpu',
                    'duration': 600,
                    'intensity': 'high',
                    'target_services': ['frontend', 'backend', 'database'],
                    'parameters': {
                        'cpu_percent': 80
                    }
                },
                'data_collection': {
                    'collection_duration': 900,
                    'pre_chaos_duration': 120,
                    'post_chaos_duration': 180,
                    'sampling_interval': 10,
                    'services_filter': ['frontend', 'backend', 'database', 'cache'],
                    'metrics_filter': ['cpu_usage', 'memory_usage', 'request_latency', 'error_rate']
                },
                'traffic': {
                    'enabled': True,
                    'traffic_type': 'spike',
                    'users': 50,
                    'spawn_rate': 5.0,
                    'duration': 720
                },
                'export': {
                    'output_directory': 'data/comprehensive_tests',
                    'export_format': 're2',
                    'compress_output': True,
                    'include_raw_data': True
                }
            },
            
            'quick_validation': {
                'name': 'Quick Validation Test',
                'description': 'Short test for validating system setup',
                'chaos': {
                    'fault_type': 'cpu',
                    'duration': 60,
                    'intensity': 'low',
                    'target_services': ['frontend'],
                    'parameters': {
                        'cpu_percent': 25
                    }
                },
                'data_collection': {
                    'collection_duration': 180,
                    'pre_chaos_duration': 30,
                    'post_chaos_duration': 90,
                    'sampling_interval': 5
                },
                'traffic': {
                    'enabled': True,
                    'traffic_type': 'normal',
                    'users': 5,
                    'spawn_rate': 1.0,
                    'duration': 120
                }
            }
        }
    
    def get_template(self, template_name: str) -> Dict[str, Any]:
        """
        Get configuration template by name.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Template configuration dictionary
            
        Raises:
            KeyError: If template doesn't exist
        """
        if template_name not in self.templates:
            available = list(self.templates.keys())
            raise KeyError(f"Template '{template_name}' not found. Available templates: {available}")
        
        return self.templates[template_name].copy()
    
    def list_templates(self) -> List[Dict[str, str]]:
        """
        List all available templates with descriptions.
        
        Returns:
            List of template info dictionaries
        """
        template_list = []
        for name, template in self.templates.items():
            template_list.append({
                'name': name,
                'title': template.get('name', name),
                'description': template.get('description', ''),
                'fault_type': template.get('chaos', {}).get('fault_type', ''),
                'duration': template.get('chaos', {}).get('duration', 0)
            })
        
        return sorted(template_list, key=lambda x: x['name'])
    
    def create_config_from_template(self, template_name: str, 
                                  target_services: List[str],
                                  overrides: Dict[str, Any] = None) -> ExperimentConfig:
        """
        Create ExperimentConfig from template.
        
        Args:
            template_name: Name of the template to use
            target_services: List of target services
            overrides: Optional configuration overrides
            
        Returns:
            ExperimentConfig object
        """
        template = self.get_template(template_name)
        
        # Apply target services
        template['chaos']['target_services'] = target_services
        
        # Apply overrides if provided
        if overrides:
            template = self._deep_merge(template, overrides)
        
        # Parse into ExperimentConfig
        from .experiment_config_parser import ExperimentConfigParser
        parser = ExperimentConfigParser()
        return parser.parse_dict(template)
    
    def save_template(self, template_name: str, config: ExperimentConfig, 
                     output_path: Path) -> None:
        """
        Save configuration as a template file.
        
        Args:
            template_name: Name for the template
            config: ExperimentConfig to save as template
            output_path: Path to save template file
        """
        from .experiment_config_parser import ExperimentConfigParser
        parser = ExperimentConfigParser()
        
        # Convert config to dict and save
        config_dict = parser._config_to_dict(config)
        
        # Add template metadata
        template_dict = {
            'template_name': template_name,
            'created_at': config.created_at.isoformat() if config.created_at else None,
            'config': config_dict
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix.lower() == '.json':
            with open(output_path, 'w') as f:
                json.dump(template_dict, f, indent=2, default=str)
        else:
            with open(output_path, 'w') as f:
                yaml.dump(template_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved template '{template_name}' to {output_path}")
    
    def load_template_from_file(self, template_path: Path) -> Dict[str, Any]:
        """
        Load template from file.
        
        Args:
            template_path: Path to template file
            
        Returns:
            Template configuration dictionary
        """
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
        
        with open(template_path, 'r') as f:
            if template_path.suffix.lower() == '.json':
                template_data = json.load(f)
            else:
                template_data = yaml.safe_load(f)
        
        # Extract config from template structure
        if 'config' in template_data:
            return template_data['config']
        else:
            return template_data
    
    def customize_template(self, template_name: str, 
                          customizations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Customize a template with specific parameters.
        
        Args:
            template_name: Base template name
            customizations: Dictionary of customizations to apply
            
        Returns:
            Customized template configuration
        """
        template = self.get_template(template_name)
        return self._deep_merge(template, customizations)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def generate_template_for_services(self, services: List[str], 
                                     fault_type: str = 'cpu',
                                     intensity: str = 'medium',
                                     duration: int = 300) -> Dict[str, Any]:
        """
        Generate a custom template for specific services.
        
        Args:
            services: List of target services
            fault_type: Type of chaos fault
            intensity: Fault intensity level
            duration: Chaos duration in seconds
            
        Returns:
            Generated template configuration
        """
        # Start with basic template
        base_template = self.get_template(f'{fault_type}_stress_basic')
        
        # Customize for services
        customizations = {
            'name': f'{fault_type.title()} Test for {", ".join(services)}',
            'description': f'Custom {fault_type} stress test targeting {", ".join(services)}',
            'chaos': {
                'target_services': services,
                'intensity': intensity,
                'duration': duration
            },
            'data_collection': {
                'collection_duration': duration + 120,  # Add 2 minutes buffer
                'services_filter': services
            },
            'traffic': {
                'target_services': services,
                'users': min(len(services) * 5, 50)  # Scale users with service count
            }
        }
        
        return self._deep_merge(base_template, customizations)
    
    def get_recommended_template(self, service_count: int, 
                               experiment_duration: int = 300) -> str:
        """
        Get recommended template based on experiment parameters.
        
        Args:
            service_count: Number of target services
            experiment_duration: Desired experiment duration
            
        Returns:
            Recommended template name
        """
        if experiment_duration < 120:
            return 'quick_validation'
        elif experiment_duration > 600:
            return 'comprehensive_test'
        elif service_count == 1:
            return 'cpu_stress_basic'
        elif service_count <= 3:
            return 'memory_stress_basic'
        else:
            return 'comprehensive_test'