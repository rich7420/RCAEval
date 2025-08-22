"""
Service targeting system for chaos engineering experiments.
Implements service discovery, selection logic, and safety checks.
Implements requirements 6.2 and 6.5 for service targeting and safety.
"""

import requests
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceInfo:
    """Information about a discovered service."""
    name: str
    status: ServiceStatus
    endpoints: List[str]
    metrics: Dict[str, float]
    last_seen: datetime
    metadata: Dict[str, Any]
    
    def is_healthy(self) -> bool:
        """Check if service is healthy enough for chaos testing."""
        return self.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]
    
    def get_error_rate(self) -> float:
        """Get current error rate of the service."""
        return self.metrics.get('error_rate', 0.0)
    
    def get_response_time(self) -> float:
        """Get average response time of the service."""
        return self.metrics.get('avg_response_time_ms', 0.0)


class ServiceDiscovery:
    """Service discovery using observability backends."""
    
    def __init__(self, prometheus_url: str = "http://localhost:9090",
                 loki_url: str = "http://localhost:3100",
                 jaeger_url: str = "http://localhost:16686"):
        """
        Initialize service discovery.
        
        Args:
            prometheus_url: Prometheus server URL
            loki_url: Loki server URL  
            jaeger_url: Jaeger server URL
        """
        self.prometheus_url = prometheus_url.rstrip('/')
        self.loki_url = loki_url.rstrip('/')
        self.jaeger_url = jaeger_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 10
    
    def discover_services(self) -> List[ServiceInfo]:
        """
        Discover services from all observability backends.
        
        Returns:
            List of discovered services
        """
        services = {}
        
        # Discover from Prometheus
        try:
            prom_services = self._discover_from_prometheus()
            for service in prom_services:
                services[service.name] = service
        except Exception as e:
            logger.warning(f"Failed to discover services from Prometheus: {e}")
        
        # Discover from Loki
        try:
            loki_services = self._discover_from_loki()
            for service in loki_services:
                if service.name in services:
                    # Merge information
                    services[service.name] = self._merge_service_info(services[service.name], service)
                else:
                    services[service.name] = service
        except Exception as e:
            logger.warning(f"Failed to discover services from Loki: {e}")
        
        # Discover from Jaeger
        try:
            jaeger_services = self._discover_from_jaeger()
            for service in jaeger_services:
                if service.name in services:
                    services[service.name] = self._merge_service_info(services[service.name], service)
                else:
                    services[service.name] = service
        except Exception as e:
            logger.warning(f"Failed to discover services from Jaeger: {e}")
        
        return list(services.values())
    
    def _discover_from_prometheus(self) -> List[ServiceInfo]:
        """Discover services from Prometheus metrics."""
        services = []
        
        # Query for service metrics
        queries = [
            'up',  # Service up status
            'rate(http_requests_total[5m])',  # Request rate
            'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))',  # P95 latency
            'rate(http_requests_total{status=~"5.."}[5m])'  # Error rate
        ]
        
        service_metrics = {}
        
        for query in queries:
            try:
                response = self.session.get(f"{self.prometheus_url}/api/v1/query", 
                                          params={'query': query})
                if response.status_code == 200:
                    data = response.json()
                    if data['status'] == 'success':
                        for result in data['data']['result']:
                            metric = result['metric']
                            value = float(result['value'][1])
                            
                            # Extract service name
                            service_name = self._extract_service_name(metric)
                            if service_name:
                                if service_name not in service_metrics:
                                    service_metrics[service_name] = {}
                                
                                # Store metric based on query type
                                if 'up' in query:
                                    service_metrics[service_name]['up'] = value
                                elif 'http_requests_total{status=~"5.."}' in query:
                                    service_metrics[service_name]['error_rate'] = value
                                elif 'http_request_duration' in query:
                                    service_metrics[service_name]['avg_response_time_ms'] = value * 1000
                                elif 'http_requests_total' in query:
                                    service_metrics[service_name]['request_rate'] = value
            except Exception as e:
                logger.warning(f"Failed to execute Prometheus query '{query}': {e}")
        
        # Create ServiceInfo objects
        for service_name, metrics in service_metrics.items():
            status = ServiceStatus.HEALTHY
            if metrics.get('up', 1) == 0:
                status = ServiceStatus.UNHEALTHY
            elif metrics.get('error_rate', 0) > 0.1:  # 10% error rate
                status = ServiceStatus.DEGRADED
            
            services.append(ServiceInfo(
                name=service_name,
                status=status,
                endpoints=[],
                metrics=metrics,
                last_seen=datetime.now(),
                metadata={'source': 'prometheus'}
            ))
        
        return services
    
    def _discover_from_loki(self) -> List[ServiceInfo]:
        """Discover services from Loki logs."""
        services = []
        
        try:
            # Query for log labels to find services
            response = self.session.get(f"{self.loki_url}/loki/api/v1/labels")
            if response.status_code == 200:
                labels = response.json().get('data', [])
                
                # Look for service-related labels
                service_labels = [label for label in labels if 'service' in label.lower() or 'job' in label.lower()]
                
                for label in service_labels:
                    # Get label values
                    values_response = self.session.get(f"{self.loki_url}/loki/api/v1/label/{label}/values")
                    if values_response.status_code == 200:
                        values = values_response.json().get('data', [])
                        
                        for service_name in values:
                            if service_name and service_name != 'unknown':
                                services.append(ServiceInfo(
                                    name=service_name,
                                    status=ServiceStatus.UNKNOWN,
                                    endpoints=[],
                                    metrics={},
                                    last_seen=datetime.now(),
                                    metadata={'source': 'loki', 'label': label}
                                ))
        except Exception as e:
            logger.warning(f"Failed to discover services from Loki: {e}")
        
        return services
    
    def _discover_from_jaeger(self) -> List[ServiceInfo]:
        """Discover services from Jaeger traces."""
        services = []
        
        try:
            response = self.session.get(f"{self.jaeger_url}/api/services")
            if response.status_code == 200:
                data = response.json()
                service_names = data.get('data', [])
                
                for service_name in service_names:
                    if service_name:
                        # Get operations for this service
                        ops_response = self.session.get(f"{self.jaeger_url}/api/operations",
                                                      params={'service': service_name})
                        endpoints = []
                        if ops_response.status_code == 200:
                            ops_data = ops_response.json()
                            endpoints = [op.get('operationName', '') for op in ops_data.get('data', [])]
                        
                        services.append(ServiceInfo(
                            name=service_name,
                            status=ServiceStatus.UNKNOWN,
                            endpoints=endpoints,
                            metrics={},
                            last_seen=datetime.now(),
                            metadata={'source': 'jaeger'}
                        ))
        except Exception as e:
            logger.warning(f"Failed to discover services from Jaeger: {e}")
        
        return services
    
    def _extract_service_name(self, metric: Dict[str, str]) -> Optional[str]:
        """Extract service name from Prometheus metric labels."""
        # Try common service label names
        for label in ['service', 'job', 'service_name', 'app', 'application']:
            if label in metric:
                return metric[label]
        
        # Try to extract from instance
        instance = metric.get('instance', '')
        if instance:
            # Remove port if present
            return instance.split(':')[0]
        
        return None
    
    def _merge_service_info(self, existing: ServiceInfo, new: ServiceInfo) -> ServiceInfo:
        """Merge information from two ServiceInfo objects."""
        # Use the most recent status if one is more specific
        status = existing.status
        if new.status != ServiceStatus.UNKNOWN and existing.status == ServiceStatus.UNKNOWN:
            status = new.status
        elif new.status == ServiceStatus.UNHEALTHY:
            status = ServiceStatus.UNHEALTHY
        elif new.status == ServiceStatus.DEGRADED and existing.status == ServiceStatus.HEALTHY:
            status = ServiceStatus.DEGRADED
        
        # Merge endpoints
        endpoints = list(set(existing.endpoints + new.endpoints))
        
        # Merge metrics
        metrics = existing.metrics.copy()
        metrics.update(new.metrics)
        
        # Merge metadata
        metadata = existing.metadata.copy()
        metadata.update(new.metadata)
        
        return ServiceInfo(
            name=existing.name,
            status=status,
            endpoints=endpoints,
            metrics=metrics,
            last_seen=max(existing.last_seen, new.last_seen),
            metadata=metadata
        )


class ServiceTargetingSystem:
    """System for selecting and validating chaos experiment targets."""
    
    def __init__(self, prometheus_url: str = "http://localhost:9090",
                 loki_url: str = "http://localhost:3100",
                 jaeger_url: str = "http://localhost:16686"):
        """
        Initialize service targeting system.
        
        Args:
            prometheus_url: Prometheus server URL
            loki_url: Loki server URL
            jaeger_url: Jaeger server URL
        """
        self.discovery = ServiceDiscovery(prometheus_url, loki_url, jaeger_url)
        self.safety_rules = self._initialize_safety_rules()
        self.service_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def _initialize_safety_rules(self) -> Dict[str, Any]:
        """Initialize safety rules for chaos experiments."""
        return {
            'max_error_rate': 0.2,  # Don't target services with >20% error rate
            'min_health_check_success': 0.8,  # Service must be >80% healthy
            'critical_services': ['auth', 'payment', 'database'],  # Never target these
            'max_concurrent_targets': 3,  # Don't target more than 3 services at once
            'min_instances': 2,  # Don't target services with <2 instances
            'cooldown_period': 3600,  # 1 hour cooldown between experiments on same service
        }
    
    def discover_and_validate_targets(self, target_services: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        Discover and validate target services for chaos experiments.
        
        Args:
            target_services: List of requested target services
            
        Returns:
            Tuple of (valid_targets, invalid_targets, warnings)
        """
        # Discover current services
        discovered_services = self._get_cached_services()
        service_map = {svc.name: svc for svc in discovered_services}
        
        valid_targets = []
        invalid_targets = []
        warnings = []
        
        for target in target_services:
            if target not in service_map:
                invalid_targets.append(target)
                warnings.append(f"Service '{target}' not found in discovery")
                continue
            
            service_info = service_map[target]
            
            # Apply safety checks
            is_valid, safety_warnings = self._validate_target_safety(service_info)
            
            if is_valid:
                valid_targets.append(target)
            else:
                invalid_targets.append(target)
            
            warnings.extend(safety_warnings)
        
        # Check overall targeting rules
        if len(valid_targets) > self.safety_rules['max_concurrent_targets']:
            excess_targets = valid_targets[self.safety_rules['max_concurrent_targets']:]
            valid_targets = valid_targets[:self.safety_rules['max_concurrent_targets']]
            invalid_targets.extend(excess_targets)
            warnings.append(f"Reduced target count to {self.safety_rules['max_concurrent_targets']} for safety")
        
        return valid_targets, invalid_targets, warnings
    
    def _get_cached_services(self) -> List[ServiceInfo]:
        """Get services from cache or discover new ones."""
        now = datetime.now()
        
        if ('services' not in self.service_cache or 
            'timestamp' not in self.service_cache or
            (now - self.service_cache['timestamp']).seconds > self.cache_ttl):
            
            logger.info("Discovering services...")
            services = self.discovery.discover_services()
            self.service_cache = {
                'services': services,
                'timestamp': now
            }
            logger.info(f"Discovered {len(services)} services")
        
        return self.service_cache['services']
    
    def _validate_target_safety(self, service_info: ServiceInfo) -> Tuple[bool, List[str]]:
        """
        Validate if a service is safe to target for chaos experiments.
        
        Args:
            service_info: ServiceInfo to validate
            
        Returns:
            Tuple of (is_valid, warnings)
        """
        warnings = []
        
        # Check if service is in critical services list
        if service_info.name.lower() in [s.lower() for s in self.safety_rules['critical_services']]:
            warnings.append(f"Service '{service_info.name}' is marked as critical")
            return False, warnings
        
        # Check service health status
        if not service_info.is_healthy():
            warnings.append(f"Service '{service_info.name}' is not healthy (status: {service_info.status.value})")
            return False, warnings
        
        # Check error rate
        error_rate = service_info.get_error_rate()
        if error_rate > self.safety_rules['max_error_rate']:
            warnings.append(f"Service '{service_info.name}' has high error rate: {error_rate:.2%}")
            return False, warnings
        
        # Check if service is up
        if service_info.metrics.get('up', 1) == 0:
            warnings.append(f"Service '{service_info.name}' appears to be down")
            return False, warnings
        
        # Check response time (warn if very high)
        response_time = service_info.get_response_time()
        if response_time > 5000:  # 5 seconds
            warnings.append(f"Service '{service_info.name}' has high response time: {response_time:.0f}ms")
        
        return True, warnings
    
    def monitor_service_health(self, service_names: List[str], 
                             duration: int = 300) -> Dict[str, List[Dict[str, Any]]]:
        """
        Monitor service health during experiment.
        
        Args:
            service_names: List of services to monitor
            duration: Monitoring duration in seconds
            
        Returns:
            Dict mapping service names to health metrics over time
        """
        health_data = {service: [] for service in service_names}
        start_time = time.time()
        
        while time.time() - start_time < duration:
            current_services = self.discovery.discover_services()
            service_map = {svc.name: svc for svc in current_services}
            
            timestamp = datetime.now()
            
            for service_name in service_names:
                if service_name in service_map:
                    service_info = service_map[service_name]
                    
                    health_data[service_name].append({
                        'timestamp': timestamp.isoformat(),
                        'status': service_info.status.value,
                        'error_rate': service_info.get_error_rate(),
                        'response_time_ms': service_info.get_response_time(),
                        'up': service_info.metrics.get('up', 0)
                    })
                else:
                    health_data[service_name].append({
                        'timestamp': timestamp.isoformat(),
                        'status': 'not_found',
                        'error_rate': 1.0,
                        'response_time_ms': 0,
                        'up': 0
                    })
            
            time.sleep(30)  # Check every 30 seconds
        
        return health_data
    
    def get_service_dependencies(self, service_name: str) -> List[str]:
        """
        Get dependencies for a service from traces.
        
        Args:
            service_name: Name of the service
            
        Returns:
            List of dependent service names
        """
        dependencies = []
        
        try:
            # Get recent traces for the service
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)
            
            params = {
                'service': service_name,
                'start': int(start_time.timestamp() * 1000000),  # microseconds
                'end': int(end_time.timestamp() * 1000000),
                'limit': 100
            }
            
            response = self.discovery.session.get(f"{self.discovery.jaeger_url}/api/traces", params=params)
            
            if response.status_code == 200:
                data = response.json()
                traces = data.get('data', [])
                
                dependent_services = set()
                
                for trace in traces:
                    spans = trace.get('spans', [])
                    for span in spans:
                        process = span.get('process', {})
                        span_service = process.get('serviceName', '')
                        
                        if span_service and span_service != service_name:
                            dependent_services.add(span_service)
                
                dependencies = list(dependent_services)
        
        except Exception as e:
            logger.warning(f"Failed to get dependencies for {service_name}: {e}")
        
        return dependencies
    
    def suggest_targets(self, fault_type: str, max_targets: int = 3) -> List[Dict[str, Any]]:
        """
        Suggest optimal targets for a specific fault type.
        
        Args:
            fault_type: Type of chaos fault
            max_targets: Maximum number of targets to suggest
            
        Returns:
            List of suggested target information
        """
        services = self._get_cached_services()
        suggestions = []
        
        # Filter services based on fault type suitability
        suitable_services = []
        
        for service in services:
            if not service.is_healthy():
                continue
            
            # Apply fault-type specific filtering
            if fault_type == 'cpu':
                # Prefer services with moderate CPU usage
                if service.metrics.get('request_rate', 0) > 0:
                    suitable_services.append((service, service.metrics.get('request_rate', 0)))
            
            elif fault_type == 'memory':
                # Prefer services that handle data
                if 'database' in service.name.lower() or 'cache' in service.name.lower():
                    suitable_services.append((service, 10))  # High priority
                elif service.metrics.get('request_rate', 0) > 0:
                    suitable_services.append((service, service.metrics.get('request_rate', 0)))
            
            elif fault_type in ['network_delay', 'network_loss']:
                # Prefer services with network communication
                if len(service.endpoints) > 0:
                    suitable_services.append((service, len(service.endpoints)))
            
            else:
                # Default: prefer services with activity
                if service.metrics.get('request_rate', 0) > 0:
                    suitable_services.append((service, service.metrics.get('request_rate', 0)))
        
        # Sort by suitability score and take top candidates
        suitable_services.sort(key=lambda x: x[1], reverse=True)
        
        for service, score in suitable_services[:max_targets]:
            is_valid, warnings = self._validate_target_safety(service)
            
            if is_valid:
                suggestions.append({
                    'name': service.name,
                    'status': service.status.value,
                    'suitability_score': score,
                    'endpoints': service.endpoints,
                    'metrics': service.metrics,
                    'warnings': warnings
                })
        
        return suggestions
    
    def update_safety_rules(self, new_rules: Dict[str, Any]) -> None:
        """
        Update safety rules for targeting.
        
        Args:
            new_rules: Dictionary of new safety rules
        """
        self.safety_rules.update(new_rules)
        logger.info(f"Updated safety rules: {new_rules}")
    
    def get_targeting_report(self, target_services: List[str]) -> Dict[str, Any]:
        """
        Generate comprehensive targeting report.
        
        Args:
            target_services: List of target services
            
        Returns:
            Comprehensive targeting report
        """
        valid_targets, invalid_targets, warnings = self.discover_and_validate_targets(target_services)
        
        services = self._get_cached_services()
        service_map = {svc.name: svc for svc in services}
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'requested_targets': target_services,
            'valid_targets': valid_targets,
            'invalid_targets': invalid_targets,
            'warnings': warnings,
            'target_details': {},
            'safety_rules': self.safety_rules,
            'total_discovered_services': len(services)
        }
        
        # Add details for valid targets
        for target in valid_targets:
            if target in service_map:
                service_info = service_map[target]
                dependencies = self.get_service_dependencies(target)
                
                report['target_details'][target] = {
                    'status': service_info.status.value,
                    'endpoints': service_info.endpoints,
                    'metrics': service_info.metrics,
                    'dependencies': dependencies,
                    'last_seen': service_info.last_seen.isoformat(),
                    'metadata': service_info.metadata
                }
        
        return report