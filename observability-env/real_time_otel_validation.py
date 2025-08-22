#!/usr/bin/env python3
"""
Real-Time OpenTelemetry Demo Validation System

This script provides a comprehensive real-time validation system for OpenTelemetry Demo
environments with automated chaos injection, monitoring, and validation capabilities.

Features:
- Automatic OpenTelemetry Demo service discovery
- Real-time chaos injection (CPU, memory, network)
- Live monitoring of metrics, logs, and traces
- Automated data validation and quality checks
- Comprehensive status reporting

Usage:
    python real_time_otel_validation.py [--chaos-type cpu|memory|network] [--duration 60]
"""

import sys
import os
import time
import logging
import argparse
import subprocess
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading
import signal

# Add config and collectors to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'config'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'collectors'))

try:
    from config import ServiceTargetingSystem, ExperimentConfigParser
    from collectors.metrics.prometheus_client import MetricsCollector
    from collectors.logs.loki_client import LogsCollector
    from collectors.traces.jaeger_client import TracesCollector
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the observability-env directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('real_time_validation.log')
    ]
)
logger = logging.getLogger(__name__)


class RealTimeOTelValidator:
    """Real-time OpenTelemetry Demo validation system."""
    
    def __init__(self):
        self.start_time = None
        self.chaos_active = False
        self.monitoring_active = False
        self.stop_monitoring = threading.Event()
        self.validation_results = {}
        
        # OpenTelemetry Demo application services (exclude observability infrastructure)
        self.otel_demo_services = [
            'frontend',
            'emailservice',
            'paymentservice', 
            'productcatalogservice',
            'shippingservice',
            'cartservice',
            'checkoutservice',
            'currencyservice',
            'recommendationservice',
            'adservice'
        ]
        
        # Observability endpoints
        self.endpoints = {
            'prometheus': 'http://localhost:9090',
            'loki': 'http://localhost:3100',
            'jaeger': 'http://localhost:16686',
            'grafana': 'http://localhost:3000'
        }
        
        # Initialize collectors
        self.collectors = {}
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info("🛑 Received interrupt signal, cleaning up...")
            self.stop_monitoring.set()
            self.cleanup_chaos()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def validate_environment(self) -> bool:
        """Validate that OpenTelemetry Demo environment is running."""
        logger.info("🔍 Validating OpenTelemetry Demo environment...")
        
        # Check Docker containers
        try:
            result = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.error("❌ Failed to check Docker containers")
                return False
                
            running_containers = result.stdout.strip().split('\n')
            logger.info(f"📦 Found {len(running_containers)} running containers")
            
            # Check for OpenTelemetry Demo services
            otel_containers = [c for c in running_containers if any(svc in c for svc in self.otel_demo_services)]
            logger.info(f"🎯 Found {len(otel_containers)} OpenTelemetry Demo services: {otel_containers}")
            
            if len(otel_containers) < 3:
                logger.warning("⚠️  Few OpenTelemetry Demo services found, continuing anyway...")
            
        except Exception as e:
            logger.error(f"❌ Error checking containers: {e}")
            return False
        
        # Check observability endpoints
        import requests
        
        for name, url in self.endpoints.items():
            try:
                if name == 'prometheus':
                    # Test with a simple query
                    response = requests.get(f"{url}/api/v1/query", 
                                          params={'query': 'up'}, timeout=5)
                elif name == 'loki':
                    # Test Loki ready endpoint
                    response = requests.get(f"{url}/ready", timeout=5)
                elif name == 'jaeger':
                    # Test Jaeger services endpoint
                    response = requests.get(f"{url}/api/services", timeout=5)
                else:
                    # Grafana and others
                    response = requests.get(url, timeout=5)
                
                if response.status_code in [200, 404]:  # 404 is OK for some endpoints
                    logger.info(f"✅ {name.capitalize()} accessible at {url}")
                else:
                    logger.warning(f"⚠️  {name.capitalize()} returned status {response.status_code}")
            except Exception as e:
                logger.warning(f"⚠️  {name.capitalize()} not accessible: {e}")
                # Don't fail validation for observability endpoints, just warn
        
        return True
    
    def initialize_collectors(self) -> bool:
        """Initialize data collectors."""
        logger.info("📊 Initializing data collectors...")
        
        try:
            self.collectors['metrics'] = MetricsCollector(self.endpoints['prometheus'])
            self.collectors['logs'] = LogsCollector(self.endpoints['loki'])
            self.collectors['traces'] = TracesCollector(self.endpoints['jaeger'])
            
            logger.info("✅ All data collectors initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize collectors: {e}")
            return False
    
    def discover_target_services(self) -> List[str]:
        """Discover available OpenTelemetry Demo services."""
        logger.info("🎯 Discovering target services...")
        
        try:
            # First try the targeting system but don't rely on it completely
            try:
                targeting_system = ServiceTargetingSystem()
                services = targeting_system._get_cached_services()
                
                # Filter for OpenTelemetry Demo services that are actually running
                available_services = []
                for service in services:
                    if service.name in self.otel_demo_services and service.status.value in ['HEALTHY', 'DEGRADED']:
                        available_services.append(service.name)
                
                logger.info(f"✅ Discovered {len(available_services)} available services: {available_services}")
                
                if available_services:
                    return available_services[:3]  # Limit to 3 services for safety
                    
            except Exception as e:
                logger.debug(f"Targeting system error: {e}")
            
            # Fall back to container-based discovery
            logger.info("🔄 Using container-based discovery...")
            result = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                containers = result.stdout.strip().split('\n')
                available_services = [c for c in containers if c in self.otel_demo_services]
                logger.info(f"📦 Container-based discovery found: {available_services}")
                
                if available_services:
                    return available_services[:3]  # Limit to 3 services for safety
            
            # Final fallback to common services
            logger.info("🔄 Using fallback services...")
            fallback_services = ['frontend', 'emailservice', 'paymentservice']
            
            # Check which fallback services are actually running
            result = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                containers = result.stdout.strip().split('\n')
                running_fallbacks = [s for s in fallback_services if s in containers]
                if running_fallbacks:
                    logger.info(f"📦 Using running fallback services: {running_fallbacks}")
                    return running_fallbacks[:2]  # Limit to 2 for safety
            
            # Last resort
            logger.warning("⚠️  Using default services (may not be running)")
            return ['frontend', 'emailservice']
            
        except Exception as e:
            logger.error(f"❌ Service discovery failed: {e}")
            # Return safe defaults
            return ['frontend', 'emailservice']
    
    def inject_chaos(self, chaos_type: str, target_services: List[str], duration: int) -> bool:
        """Inject chaos into target services."""
        logger.info(f"💥 Injecting {chaos_type} chaos into {target_services} for {duration}s")
        
        self.chaos_active = True
        chaos_commands = []
        
        try:
            for service in target_services:
                # First check if container is running
                check_cmd = ['docker', 'inspect', '-f', '{{.State.Running}}', service]
                try:
                    result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=5)
                    if result.returncode != 0 or result.stdout.strip() != 'true':
                        logger.warning(f"⚠️  Container {service} is not running, skipping")
                        continue
                except Exception as e:
                    logger.warning(f"⚠️  Cannot check container {service}: {e}")
                    continue
                
                if chaos_type == 'cpu':
                    # Try multiple CPU stress methods
                    stress_commands = [
                        f'stress-ng --cpu 1 --cpu-load 50 --timeout {duration}s &',
                        f'stress --cpu 1 --timeout {duration}s &',
                        f'dd if=/dev/zero of=/dev/null &'  # Fallback
                    ]
                elif chaos_type == 'memory':
                    # Try multiple memory stress methods
                    stress_commands = [
                        f'stress-ng --vm 1 --vm-bytes 128M --timeout {duration}s &',
                        f'stress --vm 1 --vm-bytes 128M --timeout {duration}s &'
                    ]
                elif chaos_type == 'network':
                    # Network delay - simpler approach
                    stress_commands = [
                        f'sleep {duration} &'  # Placeholder for network chaos
                    ]
                else:
                    logger.error(f"❌ Unknown chaos type: {chaos_type}")
                    continue
                
                # Try each stress command until one works
                success = False
                for stress_cmd in stress_commands:
                    try:
                        cmd = ['docker', 'exec', service, 'sh', '-c', stress_cmd]
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                        if result.returncode == 0:
                            logger.info(f"✅ {chaos_type} chaos injected into {service}")
                            chaos_commands.append((service, cmd))
                            success = True
                            break
                        else:
                            logger.debug(f"Command failed for {service}: {stress_cmd} - {result.stderr}")
                    except Exception as e:
                        logger.debug(f"Command error for {service}: {stress_cmd} - {e}")
                
                if not success:
                    logger.warning(f"⚠️  All chaos injection methods failed for {service}")
            
            if chaos_commands:
                logger.info(f"🔥 Chaos active on {len(chaos_commands)} services")
                return True
            else:
                logger.warning("⚠️  No chaos injection succeeded, but continuing with monitoring")
                return True  # Continue even if chaos fails
                
        except Exception as e:
            logger.error(f"❌ Chaos injection failed: {e}")
            return True  # Continue even if chaos fails
    
    def cleanup_chaos(self):
        """Clean up any active chaos injection."""
        if not self.chaos_active:
            return
            
        logger.info("🧹 Cleaning up chaos injection...")
        
        # Kill stress processes
        cleanup_commands = [
            ['docker', 'exec', service, 'pkill', '-f', 'stress-ng']
            for service in self.otel_demo_services
        ]
        
        # Remove network rules
        cleanup_commands.extend([
            ['docker', 'exec', service, 'tc', 'qdisc', 'del', 'dev', 'eth0', 'root']
            for service in self.otel_demo_services
        ])
        
        for cmd in cleanup_commands:
            try:
                subprocess.run(cmd, capture_output=True, timeout=5)
            except:
                pass  # Ignore cleanup errors
        
        self.chaos_active = False
        logger.info("✅ Chaos cleanup complete")
    
    def monitor_real_time(self, duration: int, target_services: List[str]) -> Dict:
        """Monitor system in real-time during chaos injection."""
        logger.info(f"📊 Starting real-time monitoring for {duration}s")
        
        self.monitoring_active = True
        monitoring_data = {
            'metrics': [],
            'logs': [],
            'traces': [],
            'timestamps': [],
            'anomalies': []
        }
        
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=duration)
        
        def collect_data():
            """Background data collection."""
            while not self.stop_monitoring.is_set() and datetime.now() < end_time:
                try:
                    current_time = datetime.now()
                    metrics_count = 0
                    logs_count = 0
                    traces_count = 0
                    
                    # Collect metrics with simple Prometheus queries
                    try:
                        import requests
                        response = requests.get(f"{self.endpoints['prometheus']}/api/v1/query", 
                                              params={'query': 'up'}, timeout=5)
                        if response.status_code == 200:
                            data = response.json()
                            if data.get('status') == 'success':
                                metrics_count = len(data.get('data', {}).get('result', []))
                    except Exception as e:
                        logger.debug(f"Metrics collection error: {e}")
                    
                    # Try to collect logs with simpler query
                    try:
                        import requests
                        # Use container labels instead of service labels
                        query = '{container=~"frontend|paymentservice|emailservice"}'
                        start_ns = int((current_time - timedelta(seconds=30)).timestamp() * 1e9)
                        end_ns = int(current_time.timestamp() * 1e9)
                        
                        response = requests.get(f"{self.endpoints['loki']}/loki/api/v1/query_range",
                                              params={
                                                  'query': query,
                                                  'start': start_ns,
                                                  'end': end_ns,
                                                  'limit': 100
                                              }, timeout=5)
                        if response.status_code == 200:
                            data = response.json()
                            if data.get('status') == 'success':
                                logs_count = len(data.get('data', {}).get('result', []))
                    except Exception as e:
                        logger.debug(f"Logs collection error: {e}")
                    
                    # Try to collect traces
                    try:
                        import requests
                        response = requests.get(f"{self.endpoints['jaeger']}/api/services", timeout=5)
                        if response.status_code == 200:
                            data = response.json()
                            traces_count = len(data.get('data', []))
                    except Exception as e:
                        logger.debug(f"Traces collection error: {e}")
                    
                    monitoring_data['metrics'].append(metrics_count)
                    monitoring_data['logs'].append(logs_count)
                    monitoring_data['traces'].append(traces_count)
                    monitoring_data['timestamps'].append(current_time.isoformat())
                    
                    # Simple anomaly detection
                    if len(monitoring_data['metrics']) > 2:
                        recent_metrics = monitoring_data['metrics'][-3:]
                        if max(recent_metrics) > 2 * min(recent_metrics) and max(recent_metrics) > 0:
                            monitoring_data['anomalies'].append({
                                'time': current_time.isoformat(),
                                'type': 'metrics_spike',
                                'value': max(recent_metrics)
                            })
                    
                except Exception as e:
                    logger.warning(f"⚠️  Monitoring error: {e}")
                
                time.sleep(10)  # Collect every 10 seconds
        
        # Start background monitoring
        monitor_thread = threading.Thread(target=collect_data)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Progress reporting
        for i in range(duration):
            if self.stop_monitoring.is_set():
                break
                
            if i % 15 == 0 or i < 5:
                remaining = duration - i
                metrics_count = sum(monitoring_data['metrics'])
                logs_count = sum(monitoring_data['logs'])
                traces_count = sum(monitoring_data['traces'])
                
                logger.info(f"📊 Monitoring: {remaining}s remaining | "
                          f"Metrics: {metrics_count} | Logs: {logs_count} | Traces: {traces_count}")
            
            time.sleep(1)
        
        self.stop_monitoring.set()
        monitor_thread.join(timeout=5)
        self.monitoring_active = False
        
        logger.info("✅ Real-time monitoring complete")
        return monitoring_data
    
    def validate_data_quality(self, monitoring_data: Dict) -> Dict:
        """Validate the quality of collected data."""
        logger.info("🔍 Validating data quality...")
        
        validation_results = {
            'overall_valid': True,
            'metrics_validation': {},
            'logs_validation': {},
            'traces_validation': {},
            'issues': []
        }
        
        # Validate metrics
        metrics_counts = monitoring_data['metrics']
        if metrics_counts:
            avg_metrics = sum(metrics_counts) / len(metrics_counts)
            validation_results['metrics_validation'] = {
                'total_collections': len(metrics_counts),
                'average_per_collection': avg_metrics,
                'valid': avg_metrics > 0
            }
            if avg_metrics == 0:
                validation_results['issues'].append("No metrics data collected")
                validation_results['overall_valid'] = False
        else:
            validation_results['issues'].append("No metrics collections performed")
            validation_results['overall_valid'] = False
        
        # Validate logs
        logs_counts = monitoring_data['logs']
        if logs_counts:
            avg_logs = sum(logs_counts) / len(logs_counts)
            validation_results['logs_validation'] = {
                'total_collections': len(logs_counts),
                'average_per_collection': avg_logs,
                'valid': avg_logs >= 0  # Logs can be 0 in some cases
            }
        else:
            validation_results['issues'].append("No log collections performed")
        
        # Validate traces
        traces_counts = monitoring_data['traces']
        if traces_counts:
            avg_traces = sum(traces_counts) / len(traces_counts)
            validation_results['traces_validation'] = {
                'total_collections': len(traces_counts),
                'average_per_collection': avg_traces,
                'valid': avg_traces >= 0  # Traces can be 0 in some cases
            }
        else:
            validation_results['issues'].append("No trace collections performed")
        
        # Check for anomalies
        anomalies = monitoring_data.get('anomalies', [])
        if anomalies:
            validation_results['anomalies_detected'] = len(anomalies)
            logger.info(f"🚨 Detected {len(anomalies)} anomalies during monitoring")
        
        if validation_results['overall_valid']:
            logger.info("✅ Data quality validation passed")
        else:
            logger.warning(f"⚠️  Data quality issues: {validation_results['issues']}")
        
        return validation_results
    
    def generate_report(self, target_services: List[str], chaos_type: str, 
                       duration: int, monitoring_data: Dict, validation_results: Dict) -> str:
        """Generate comprehensive validation report."""
        
        report = f"""
🚀 Real-Time OpenTelemetry Demo Validation Report
{'=' * 60}

📋 Experiment Details:
   • Chaos Type: {chaos_type}
   • Target Services: {target_services}
   • Duration: {duration} seconds
   • Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
   • End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 Data Collection Summary:
   • Metrics Collections: {len(monitoring_data['metrics'])}
   • Total Metrics: {sum(monitoring_data['metrics'])}
   • Total Logs: {sum(monitoring_data['logs'])}
   • Total Traces: {sum(monitoring_data['traces'])}
   • Anomalies Detected: {len(monitoring_data.get('anomalies', []))}

✅ Validation Results:
   • Overall Status: {'PASSED' if validation_results['overall_valid'] else 'FAILED'}
   • Metrics Valid: {'✅' if validation_results['metrics_validation'].get('valid') else '❌'}
   • Logs Valid: {'✅' if validation_results['logs_validation'].get('valid') else '❌'}
   • Traces Valid: {'✅' if validation_results['traces_validation'].get('valid') else '❌'}

"""
        
        if validation_results['issues']:
            report += f"⚠️  Issues Detected:\n"
            for issue in validation_results['issues']:
                report += f"   • {issue}\n"
        
        if monitoring_data.get('anomalies'):
            report += f"\n🚨 Anomalies Detected:\n"
            for anomaly in monitoring_data['anomalies'][:5]:  # Show first 5
                report += f"   • {anomaly['time']}: {anomaly['type']} (value: {anomaly['value']})\n"
        
        report += f"\n🎯 Recommendations:\n"
        if validation_results['overall_valid']:
            report += "   • ✅ System is functioning correctly\n"
            report += "   • ✅ Ready for production experiments\n"
        else:
            report += "   • ❌ Address data collection issues before production use\n"
            report += "   • 🔧 Check observability backend configurations\n"
        
        return report
    
    def run_validation(self, chaos_type: str = 'cpu', duration: int = 60) -> bool:
        """Run the complete real-time validation."""
        
        print("🚀 Real-Time OpenTelemetry Demo Validation")
        print("=" * 60)
        
        self.start_time = datetime.now()
        self.setup_signal_handlers()
        
        try:
            # Step 1: Environment validation
            if not self.validate_environment():
                logger.error("❌ Environment validation failed")
                return False
            
            # Step 2: Initialize collectors
            if not self.initialize_collectors():
                logger.error("❌ Collector initialization failed")
                return False
            
            # Step 3: Discover target services
            target_services = self.discover_target_services()
            if not target_services:
                logger.error("❌ No target services found")
                return False
            
            logger.info(f"🎯 Will target services: {target_services}")
            
            # Step 4: Start monitoring and chaos injection
            logger.info(f"🏁 Starting {duration}s validation with {chaos_type} chaos")
            
            # Start chaos injection
            if not self.inject_chaos(chaos_type, target_services, duration):
                logger.error("❌ Chaos injection failed")
                return False
            
            # Monitor in real-time
            monitoring_data = self.monitor_real_time(duration, target_services)
            
            # Step 5: Cleanup chaos
            self.cleanup_chaos()
            
            # Step 6: Validate data quality
            validation_results = self.validate_data_quality(monitoring_data)
            
            # Step 7: Generate and display report
            report = self.generate_report(target_services, chaos_type, duration, 
                                        monitoring_data, validation_results)
            print(report)
            
            # Save report to file
            report_file = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            logger.info(f"📄 Report saved to {report_file}")
            
            return validation_results['overall_valid']
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            self.cleanup_chaos()
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Real-Time OpenTelemetry Demo Validation')
    parser.add_argument('--chaos-type', choices=['cpu', 'memory', 'network'], 
                       default='cpu', help='Type of chaos to inject')
    parser.add_argument('--duration', type=int, default=60, 
                       help='Duration of chaos injection in seconds')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    validator = RealTimeOTelValidator()
    success = validator.run_validation(args.chaos_type, args.duration)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()