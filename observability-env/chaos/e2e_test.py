#!/usr/bin/env python3
"""
End-to-End Chaos Engineering Test
This demonstrates how to perform chaos engineering experiments with the current setup.
"""

import subprocess
import time
import json
import requests
import sys
from datetime import datetime
from pathlib import Path

class ChaosExperiment:
    """Manages a complete chaos engineering experiment."""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.start_time = None
        self.end_time = None
        self.baseline_metrics = {}
        self.chaos_metrics = {}
        self.recovery_metrics = {}
        
    def collect_baseline_metrics(self):
        """Collect baseline metrics before chaos injection."""
        print("📊 Collecting baseline metrics...")
        
        metrics = {}
        
        # Collect Prometheus metrics
        try:
            # CPU usage
            response = requests.get(
                'http://localhost:9090/api/v1/query?query=rate(container_cpu_usage_seconds_total[5m])',
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                metrics['cpu_usage'] = len(data.get('data', {}).get('result', []))
            
            # Memory usage
            response = requests.get(
                'http://localhost:9090/api/v1/query?query=container_memory_usage_bytes',
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                metrics['memory_usage'] = len(data.get('data', {}).get('result', []))
            
            # HTTP requests
            response = requests.get(
                'http://localhost:9090/api/v1/query?query=rate(http_requests_total[5m])',
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                metrics['http_requests'] = len(data.get('data', {}).get('result', []))
                
        except requests.RequestException as e:
            print(f"⚠️  Could not collect Prometheus metrics: {e}")
        
        # Collect Jaeger services
        try:
            response = requests.get('http://localhost:16686/api/services', timeout=5)
            if response.status_code == 200:
                services = response.json()
                metrics['jaeger_services'] = len(services.get('data', []))
        except requests.RequestException as e:
            print(f"⚠️  Could not collect Jaeger metrics: {e}")
        
        self.baseline_metrics = metrics
        print(f"✅ Baseline metrics collected: {metrics}")
        
    def inject_chaos(self, chaos_type: str, target_container: str = None, duration: int = 30):
        """Inject chaos into the system."""
        print(f"💥 Injecting {chaos_type} chaos...")
        
        chaos_dir = Path(__file__).parent
        
        if chaos_type == 'cpu':
            from cpu.cpu_stress import CPUStressInjector
            
            config = {
                "intensity": "50%",
                "duration": duration,
                "target_container": target_container,
                "monitoring_interval": 1
            }
            
            injector = CPUStressInjector(config)
            
            if injector.start_injection():
                print(f"✅ CPU stress injection started on {target_container or 'host'}")
                
                # Wait for injection to complete
                while injector.get_status()['running']:
                    status = injector.get_status()
                    remaining = status.get('remaining_seconds', 0)
                    print(f"⏱️  Chaos running... {remaining:.0f}s remaining")
                    time.sleep(5)
                
                # Save injection log
                log_file = f"logs/{self.experiment_name}_cpu_injection.json"
                Path("logs").mkdir(exist_ok=True)
                injector.save_injection_log(log_file)
                print(f"📝 Injection log saved: {log_file}")
                
                return True
            else:
                print("❌ Failed to start CPU stress injection")
                return False
                
        elif chaos_type == 'memory':
            from memory.memory_stress import MemoryStressInjector
            
            config = {
                "size": "256M",
                "duration": duration,
                "target_container": target_container,
                "monitoring_interval": 1
            }
            
            injector = MemoryStressInjector(config)
            
            if injector.start_injection():
                print(f"✅ Memory stress injection started on {target_container or 'host'}")
                
                # Wait for injection to complete
                while injector.get_status()['running']:
                    status = injector.get_status()
                    remaining = status.get('remaining_seconds', 0)
                    print(f"⏱️  Chaos running... {remaining:.0f}s remaining")
                    time.sleep(5)
                
                # Save injection log
                log_file = f"logs/{self.experiment_name}_memory_injection.json"
                Path("logs").mkdir(exist_ok=True)
                injector.save_injection_log(log_file)
                print(f"📝 Injection log saved: {log_file}")
                
                return True
            else:
                print("❌ Failed to start memory stress injection")
                return False
        
        else:
            print(f"❌ Unsupported chaos type: {chaos_type}")
            return False
    
    def collect_recovery_metrics(self):
        """Collect metrics after chaos injection to verify recovery."""
        print("🔄 Collecting recovery metrics...")
        
        # Wait a bit for system to stabilize
        time.sleep(10)
        
        # Collect same metrics as baseline
        self.collect_baseline_metrics()  # Reuse the same collection logic
        self.recovery_metrics = self.baseline_metrics.copy()
        
        print(f"✅ Recovery metrics collected: {self.recovery_metrics}")
    
    def generate_experiment_report(self):
        """Generate a comprehensive experiment report."""
        report = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else None,
            "baseline_metrics": self.baseline_metrics,
            "recovery_metrics": self.recovery_metrics,
            "observations": []
        }
        
        # Add observations
        if self.baseline_metrics and self.recovery_metrics:
            for metric, baseline_value in self.baseline_metrics.items():
                recovery_value = self.recovery_metrics.get(metric, 0)
                if baseline_value != recovery_value:
                    report["observations"].append({
                        "metric": metric,
                        "baseline": baseline_value,
                        "recovery": recovery_value,
                        "change": recovery_value - baseline_value
                    })
        
        # Save report
        report_file = f"logs/{self.experiment_name}_report.json"
        Path("logs").mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Experiment report saved: {report_file}")
        return report

def run_cpu_stress_experiment():
    """Run a complete CPU stress experiment."""
    print("🧪 Starting CPU Stress Experiment")
    print("="*50)
    
    experiment = ChaosExperiment("cpu_stress_experiment")
    experiment.start_time = datetime.now()
    
    # Step 1: Collect baseline
    experiment.collect_baseline_metrics()
    
    # Step 2: Get target container
    try:
        result = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], 
                              capture_output=True, text=True, check=True)
        containers = [name.strip() for name in result.stdout.split('\n') if name.strip()]
        
        # Look for a microservice container
        target_container = None
        for container in containers:
            if any(service in container.lower() for service in ['frontend', 'checkout', 'payment', 'product']):
                target_container = container
                break
        
        if not target_container and containers:
            target_container = containers[0]  # Use first available container
            
    except subprocess.CalledProcessError:
        target_container = None
    
    print(f"🎯 Target container: {target_container or 'host system'}")
    
    # Step 3: Inject chaos
    if experiment.inject_chaos('cpu', target_container, duration=20):
        print("✅ Chaos injection completed")
    else:
        print("❌ Chaos injection failed")
        return False
    
    # Step 4: Collect recovery metrics
    experiment.collect_recovery_metrics()
    
    # Step 5: Generate report
    experiment.end_time = datetime.now()
    report = experiment.generate_experiment_report()
    
    # Step 6: Show results
    print("\n📊 EXPERIMENT RESULTS")
    print("="*30)
    print(f"Experiment: {report['experiment_name']}")
    print(f"Duration: {report['duration_seconds']:.1f} seconds")
    print(f"Target: {target_container or 'host system'}")
    
    if report['observations']:
        print("\nObserved Changes:")
        for obs in report['observations']:
            print(f"  {obs['metric']}: {obs['baseline']} → {obs['recovery']} (Δ{obs['change']:+})")
    else:
        print("\nNo significant metric changes observed")
    
    print(f"\n📝 Detailed logs available in: logs/")
    print("💡 Check Grafana (http://localhost:3000) for visual confirmation")
    
    return True

def run_memory_stress_experiment():
    """Run a complete memory stress experiment."""
    print("🧪 Starting Memory Stress Experiment")
    print("="*50)
    
    experiment = ChaosExperiment("memory_stress_experiment")
    experiment.start_time = datetime.now()
    
    # Similar to CPU experiment but with memory stress
    experiment.collect_baseline_metrics()
    
    # Get target container
    try:
        result = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], 
                              capture_output=True, text=True, check=True)
        containers = [name.strip() for name in result.stdout.split('\n') if name.strip()]
        target_container = containers[0] if containers else None
    except subprocess.CalledProcessError:
        target_container = None
    
    print(f"🎯 Target container: {target_container or 'host system'}")
    
    if experiment.inject_chaos('memory', target_container, duration=20):
        print("✅ Chaos injection completed")
    else:
        print("❌ Chaos injection failed")
        return False
    
    experiment.collect_recovery_metrics()
    experiment.end_time = datetime.now()
    report = experiment.generate_experiment_report()
    
    print("\n📊 EXPERIMENT RESULTS")
    print("="*30)
    print(f"Experiment: {report['experiment_name']}")
    print(f"Duration: {report['duration_seconds']:.1f} seconds")
    print(f"Target: {target_container or 'host system'}")
    
    return True

def main():
    """Run end-to-end chaos engineering tests."""
    print("🔬 Chaos Engineering End-to-End Test")
    print("="*50)
    print("This test demonstrates complete chaos engineering workflows")
    print("with the current observability stack (before Task 6 data collection)")
    print()
    
    # Check prerequisites
    print("🔍 Checking prerequisites...")
    
    # Check if observability stack is running
    try:
        response = requests.get('http://localhost:9090/api/v1/query?query=up', timeout=5)
        if response.status_code != 200:
            print("❌ Prometheus not accessible at localhost:9090")
            print("Please start the observability stack first:")
            print("  ./scripts/start-otel-demo.sh")
            return 1
        print("✅ Prometheus accessible")
    except requests.RequestException:
        print("❌ Prometheus not accessible")
        print("Please start the observability stack first")
        return 1
    
    # Run experiments
    experiments = [
        ("CPU Stress", run_cpu_stress_experiment),
        ("Memory Stress", run_memory_stress_experiment)
    ]
    
    passed = 0
    for exp_name, exp_func in experiments:
        print(f"\n{'='*60}")
        try:
            if exp_func():
                passed += 1
                print(f"✅ {exp_name} experiment completed successfully")
            else:
                print(f"❌ {exp_name} experiment failed")
        except Exception as e:
            print(f"❌ {exp_name} experiment failed with exception: {e}")
    
    print(f"\n{'='*60}")
    print("END-TO-END TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Completed experiments: {passed}/{len(experiments)}")
    
    if passed == len(experiments):
        print("🎉 All chaos engineering experiments completed successfully!")
        print("\n📋 Next Steps:")
        print("1. Review experiment logs in logs/ directory")
        print("2. Check Grafana dashboards for visual confirmation")
        print("3. Implement Task 6 for automated data collection")
        return 0
    else:
        print("❌ Some experiments failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())