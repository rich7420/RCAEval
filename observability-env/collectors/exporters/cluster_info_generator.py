"""
Cluster info generator for creating RE2-compatible cluster_info.json.
Implements requirement 4.4 for log template extraction and container-to-service mapping.
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set
import logging
from collections import defaultdict, Counter
import re
import os

logger = logging.getLogger(__name__)


class ClusterInfoGenerator:
    """Generate cluster_info.json in RE2 format with log templates and service mappings."""
    
    def __init__(self):
        """Initialize cluster info generator."""
        self.service_mappings = {}
        self.log_templates = {}
        self.container_info = {}
        
    def generate_cluster_info(self, logs_df: pd.DataFrame, metrics_df: pd.DataFrame,
                            traces_df: pd.DataFrame, log_templates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate complete cluster_info.json structure.
        
        Args:
            logs_df: DataFrame with log data
            metrics_df: DataFrame with metrics data
            traces_df: DataFrame with traces data
            log_templates: Dict with extracted log templates
            
        Returns:
            Dict with cluster_info structure
        """
        cluster_info = {
            "services": {},
            "containers": {},
            "log_templates": {},
            "service_dependencies": {},
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "data_sources": ["logs", "metrics", "traces"],
                "total_services": 0,
                "total_containers": 0,
                "total_log_templates": 0
            }
        }
        
        # Extract service information
        services = self._extract_services_info(logs_df, metrics_df, traces_df)
        cluster_info["services"] = services
        
        # Extract container information
        containers = self._extract_containers_info(logs_df, metrics_df)
        cluster_info["containers"] = containers
        
        # Process log templates
        processed_templates = self._process_log_templates(log_templates)
        cluster_info["log_templates"] = processed_templates
        
        # Extract service dependencies
        dependencies = self._extract_service_dependencies(traces_df)
        cluster_info["service_dependencies"] = dependencies
        
        # Update metadata
        cluster_info["metadata"]["total_services"] = len(services)
        cluster_info["metadata"]["total_containers"] = len(containers)
        cluster_info["metadata"]["total_log_templates"] = len(processed_templates)
        
        return cluster_info
        
    def _extract_services_info(self, logs_df: pd.DataFrame, metrics_df: pd.DataFrame,
                             traces_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract service information from all data sources."""
        services = {}
        
        # Get unique services from all sources
        all_services = set()
        
        if not logs_df.empty:
            all_services.update(logs_df['service'].unique())
        if not metrics_df.empty:
            all_services.update(metrics_df['service'].unique())
        if not traces_df.empty:
            all_services.update(traces_df['service'].unique())
            
        for service in all_services:
            service_info = {
                "name": service,
                "type": self._infer_service_type(service),
                "containers": [],
                "endpoints": [],
                "log_sources": [],
                "metrics": {
                    "total_requests": 0,
                    "error_count": 0,
                    "avg_response_time_ms": 0
                },
                "health_status": "unknown"
            }
            
            # Extract container information for this service
            containers = self._get_service_containers(service, logs_df, metrics_df)
            service_info["containers"] = containers
            
            # Extract endpoints from traces
            if not traces_df.empty:
                service_traces = traces_df[traces_df['service'] == service]
                endpoints = service_traces['operation'].unique().tolist()
                service_info["endpoints"] = endpoints
                
                # Calculate metrics
                if not service_traces.empty:
                    service_info["metrics"]["total_requests"] = len(service_traces)
                    service_info["metrics"]["error_count"] = service_traces['error'].sum()
                    service_info["metrics"]["avg_response_time_ms"] = service_traces['duration_ms'].mean()
                    
            # Extract log sources
            if not logs_df.empty:
                service_logs = logs_df[logs_df['service'] == service]
                if not service_logs.empty:
                    # Extract log file paths from labels
                    log_sources = self._extract_log_sources(service_logs)
                    service_info["log_sources"] = log_sources
                    
            # Determine health status
            service_info["health_status"] = self._determine_health_status(service_info)
            
            services[service] = service_info
            
        return services
        
    def _extract_containers_info(self, logs_df: pd.DataFrame, 
                               metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract container information and create container-to-service mapping."""
        containers = {}
        
        # Extract from logs
        if not logs_df.empty:
            for _, log_entry in logs_df.iterrows():
                container_info = self._parse_container_from_labels(log_entry.get('labels', '{}'))
                if container_info:
                    container_id = container_info.get('container_id', container_info.get('container_name', ''))
                    if container_id:
                        containers[container_id] = {
                            "container_id": container_id,
                            "container_name": container_info.get('container_name', container_id),
                            "service": log_entry['service'],
                            "image": container_info.get('image', 'unknown'),
                            "pod_name": container_info.get('pod_name', ''),
                            "namespace": container_info.get('namespace', 'default'),
                            "node": container_info.get('node', ''),
                            "labels": container_info.get('labels', {}),
                            "first_seen": log_entry['timestamp'],
                            "last_seen": log_entry['timestamp']
                        }
                        
        # Extract from metrics
        if not metrics_df.empty:
            for _, metric_entry in metrics_df.iterrows():
                container_info = self._parse_container_from_labels(metric_entry.get('labels', '{}'))
                if container_info:
                    container_id = container_info.get('container_id', container_info.get('container_name', ''))
                    if container_id:
                        if container_id in containers:
                            # Update existing container info
                            containers[container_id]['last_seen'] = max(
                                containers[container_id]['last_seen'],
                                metric_entry['timestamp']
                            )
                        else:
                            containers[container_id] = {
                                "container_id": container_id,
                                "container_name": container_info.get('container_name', container_id),
                                "service": metric_entry['service'],
                                "image": container_info.get('image', 'unknown'),
                                "pod_name": container_info.get('pod_name', ''),
                                "namespace": container_info.get('namespace', 'default'),
                                "node": container_info.get('node', ''),
                                "labels": container_info.get('labels', {}),
                                "first_seen": metric_entry['timestamp'],
                                "last_seen": metric_entry['timestamp']
                            }
                            
        return containers
        
    def _process_log_templates(self, log_templates: Dict[str, Any]) -> Dict[str, Any]:
        """Process log templates into RE2 format."""
        processed_templates = {}
        
        template_id = 1
        for service, service_templates in log_templates.items():
            for original_id, template_info in service_templates.items():
                processed_template = {
                    "template_id": template_id,
                    "service": service,
                    "template": template_info.get('template', ''),
                    "pattern": template_info.get('pattern', []),
                    "count": template_info.get('count', 0),
                    "examples": template_info.get('messages', [])[:3],  # Keep only 3 examples
                    "regex": self._create_regex_from_template(template_info.get('template', '')),
                    "variables": self._extract_variables_from_template(template_info.get('template', ''))
                }
                
                processed_templates[str(template_id)] = processed_template
                template_id += 1
                
        return processed_templates
        
    def _extract_service_dependencies(self, traces_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract service dependencies from traces."""
        dependencies = {}
        
        if traces_df.empty:
            return dependencies
            
        # Build dependency graph
        for trace_id, trace_group in traces_df.groupby('trace_id'):
            spans = trace_group.sort_values('start_time')
            
            # Build span hierarchy
            span_to_service = dict(zip(spans['span_id'], spans['service']))
            
            for _, span in spans.iterrows():
                service = span['service']
                parent_span_id = span['parent_span_id']
                
                if service not in dependencies:
                    dependencies[service] = {
                        "depends_on": [],
                        "depended_by": [],
                        "call_counts": {},
                        "avg_latencies": {}
                    }
                    
                if parent_span_id and parent_span_id in span_to_service:
                    parent_service = span_to_service[parent_span_id]
                    
                    if parent_service != service:
                        # Add dependency
                        if parent_service not in dependencies[service]["depends_on"]:
                            dependencies[service]["depends_on"].append(parent_service)
                            
                        if parent_service not in dependencies:
                            dependencies[parent_service] = {
                                "depends_on": [],
                                "depended_by": [],
                                "call_counts": {},
                                "avg_latencies": {}
                            }
                            
                        if service not in dependencies[parent_service]["depended_by"]:
                            dependencies[parent_service]["depended_by"].append(service)
                            
                        # Update call counts and latencies
                        call_key = f"{parent_service}->{service}"
                        if call_key not in dependencies[service]["call_counts"]:
                            dependencies[service]["call_counts"][call_key] = 0
                            dependencies[service]["avg_latencies"][call_key] = []
                            
                        dependencies[service]["call_counts"][call_key] += 1
                        dependencies[service]["avg_latencies"][call_key].append(span['duration_ms'])
                        
        # Calculate average latencies
        for service, deps in dependencies.items():
            for call_key, latencies in deps["avg_latencies"].items():
                if latencies:
                    deps["avg_latencies"][call_key] = sum(latencies) / len(latencies)
                else:
                    deps["avg_latencies"][call_key] = 0
                    
        return dependencies
        
    def _infer_service_type(self, service_name: str) -> str:
        """Infer service type from service name."""
        service_lower = service_name.lower()
        
        if any(keyword in service_lower for keyword in ['frontend', 'ui', 'web', 'client']):
            return 'frontend'
        elif any(keyword in service_lower for keyword in ['api', 'gateway', 'proxy']):
            return 'api_gateway'
        elif any(keyword in service_lower for keyword in ['db', 'database', 'postgres', 'mysql', 'mongo']):
            return 'database'
        elif any(keyword in service_lower for keyword in ['cache', 'redis', 'memcache']):
            return 'cache'
        elif any(keyword in service_lower for keyword in ['queue', 'kafka', 'rabbitmq', 'pubsub']):
            return 'message_queue'
        elif any(keyword in service_lower for keyword in ['auth', 'login', 'user']):
            return 'authentication'
        elif any(keyword in service_lower for keyword in ['payment', 'billing', 'checkout']):
            return 'payment'
        elif any(keyword in service_lower for keyword in ['email', 'notification', 'sms']):
            return 'notification'
        else:
            return 'microservice'
            
    def _get_service_containers(self, service: str, logs_df: pd.DataFrame,
                              metrics_df: pd.DataFrame) -> List[str]:
        """Get container IDs/names for a service."""
        containers = set()
        
        # From logs
        if not logs_df.empty:
            service_logs = logs_df[logs_df['service'] == service]
            for _, log_entry in service_logs.iterrows():
                container_info = self._parse_container_from_labels(log_entry.get('labels', '{}'))
                if container_info:
                    container_id = container_info.get('container_id', container_info.get('container_name', ''))
                    if container_id:
                        containers.add(container_id)
                        
        # From metrics
        if not metrics_df.empty:
            service_metrics = metrics_df[metrics_df['service'] == service]
            for _, metric_entry in service_metrics.iterrows():
                container_info = self._parse_container_from_labels(metric_entry.get('labels', '{}'))
                if container_info:
                    container_id = container_info.get('container_id', container_info.get('container_name', ''))
                    if container_id:
                        containers.add(container_id)
                        
        return list(containers)
        
    def _parse_container_from_labels(self, labels_json: str) -> Optional[Dict[str, Any]]:
        """Parse container information from labels JSON."""
        try:
            labels = json.loads(labels_json) if isinstance(labels_json, str) else labels_json
            
            if not isinstance(labels, dict):
                return None
                
            container_info = {}
            
            # Extract container ID/name
            for key in ['container', 'container_name', 'container_id', 'pod']:
                if key in labels:
                    container_info['container_name'] = labels[key]
                    container_info['container_id'] = labels[key]
                    break
                    
            # Extract image
            for key in ['image', 'container_image']:
                if key in labels:
                    container_info['image'] = labels[key]
                    break
                    
            # Extract pod information
            for key in ['pod', 'pod_name']:
                if key in labels:
                    container_info['pod_name'] = labels[key]
                    break
                    
            # Extract namespace
            for key in ['namespace', 'k8s_namespace']:
                if key in labels:
                    container_info['namespace'] = labels[key]
                    break
                    
            # Extract node
            for key in ['node', 'node_name', 'instance']:
                if key in labels:
                    container_info['node'] = labels[key]
                    break
                    
            container_info['labels'] = labels
            
            return container_info if container_info else None
            
        except (json.JSONDecodeError, TypeError):
            return None
            
    def _extract_log_sources(self, service_logs: pd.DataFrame) -> List[str]:
        """Extract log source files for a service."""
        log_sources = set()
        
        for _, log_entry in service_logs.iterrows():
            try:
                labels = json.loads(log_entry.get('labels', '{}'))
                
                # Look for file paths in labels
                for key in ['filename', 'file', 'log_file', 'source']:
                    if key in labels:
                        log_sources.add(labels[key])
                        
            except (json.JSONDecodeError, TypeError):
                continue
                
        return list(log_sources)
        
    def _determine_health_status(self, service_info: Dict[str, Any]) -> str:
        """Determine health status based on service metrics."""
        metrics = service_info.get('metrics', {})
        
        error_count = metrics.get('error_count', 0)
        total_requests = metrics.get('total_requests', 0)
        
        if total_requests == 0:
            return 'unknown'
            
        error_rate = error_count / total_requests
        
        if error_rate > 0.1:  # 10% error rate
            return 'unhealthy'
        elif error_rate > 0.05:  # 5% error rate
            return 'degraded'
        else:
            return 'healthy'
            
    def _create_regex_from_template(self, template: str) -> str:
        """Create regex pattern from log template."""
        # Escape special regex characters
        escaped = re.escape(template)
        
        # Replace placeholders with regex patterns
        replacements = {
            r'\\<IP\\>': r'\\d+\\.\\d+\\.\\d+\\.\\d+',
            r'\\<UUID\\>': r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}',
            r'\\<TIMESTAMP\\>': r'\\d{4}-\\d{2}-\\d{2}[T\\s]\\d{2}:\\d{2}:\\d{2}',
            r'\\<NUMBER\\>': r'\\d+',
            r'\\<FLOAT\\>': r'\\d+\\.\\d+',
            r'\\<EMAIL\\>': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}',
            r'\\<PATH\\>': r'/[^\\s]*',
            r'\\<URL\\>': r'https?://[^\\s]+',
            r'\\<HASH\\>': r'[a-fA-F0-9]{32,}',
            r'\\<\\*\\>': r'.*'  # Wildcard
        }
        
        regex_pattern = escaped
        for placeholder, pattern in replacements.items():
            regex_pattern = re.sub(placeholder, pattern, regex_pattern)
            
        return regex_pattern
        
    def _extract_variables_from_template(self, template: str) -> List[str]:
        """Extract variable placeholders from template."""
        variables = []
        
        # Find all placeholder patterns
        placeholders = re.findall(r'<([^>]+)>', template)
        
        for placeholder in placeholders:
            if placeholder not in variables:
                variables.append(placeholder)
                
        return variables
        
    def save_cluster_info(self, cluster_info: Dict[str, Any], output_path: str) -> None:
        """
        Save cluster_info to JSON file.
        
        Args:
            cluster_info: Cluster info dictionary
            output_path: Output file path
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(cluster_info, f, indent=2, default=str)
                
            logger.info(f"Cluster info saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save cluster info: {e}")
            raise
            
    def validate_cluster_info(self, cluster_info: Dict[str, Any]) -> List[str]:
        """
        Validate cluster_info structure.
        
        Args:
            cluster_info: Cluster info dictionary
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check required top-level keys
        required_keys = ['services', 'containers', 'log_templates', 'service_dependencies', 'metadata']
        for key in required_keys:
            if key not in cluster_info:
                errors.append(f"Missing required key: {key}")
                
        # Validate services
        if 'services' in cluster_info:
            for service_name, service_info in cluster_info['services'].items():
                if not isinstance(service_info, dict):
                    errors.append(f"Service {service_name} info must be a dictionary")
                    continue
                    
                required_service_keys = ['name', 'type', 'containers', 'endpoints']
                for key in required_service_keys:
                    if key not in service_info:
                        errors.append(f"Service {service_name} missing required key: {key}")
                        
        # Validate containers
        if 'containers' in cluster_info:
            for container_id, container_info in cluster_info['containers'].items():
                if not isinstance(container_info, dict):
                    errors.append(f"Container {container_id} info must be a dictionary")
                    continue
                    
                required_container_keys = ['container_id', 'service']
                for key in required_container_keys:
                    if key not in container_info:
                        errors.append(f"Container {container_id} missing required key: {key}")
                        
        # Validate log templates
        if 'log_templates' in cluster_info:
            for template_id, template_info in cluster_info['log_templates'].items():
                if not isinstance(template_info, dict):
                    errors.append(f"Template {template_id} info must be a dictionary")
                    continue
                    
                required_template_keys = ['template_id', 'service', 'template']
                for key in required_template_keys:
                    if key not in template_info:
                        errors.append(f"Template {template_id} missing required key: {key}")
                        
        return errors