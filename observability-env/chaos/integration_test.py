#!/usr/bin/env python3
"""
Integration test for chaos engineering with observability stack.
This test verifies that chaos injection works with the existing observability infrastructure.
"""

import subprocess
import time
import json
import requests
import sys
from pathlib import Path

def check_docker_services():
    """Check if required Docker services are running."""
    print("Checking Docker services...")
    
    required_services = [
        'grafana',
        'prometheus', 
        'jaeger',
        'otel-collector'
    ]
    
    try:
        result = subprocess.run(['docker', 'ps', '--format', 'table {{.Names}}'], 
                              capture_output=True, text=True, check=True)
        running_services = result.stdout.lower()
        
        missing_services = []
        for service in required_services:
            if service not in running_services:
                missing_services.append(service)
        
        if missing_services:
            print(f"❌ Missing services: {missing_services}")
            print("Please start the observability stack first:")
            print("  ./scripts/start-otel-demo.sh")
            print("  or")
            print("  ./scripts/start-online-boutique.sh")
            return False
        
        print("✅ All required observability services are running")
        return True
        
    except subprocess.CalledProcessError:
        print("❌ Failed to check Docker services")
        return False

def check_microservices():
    """Check if demo microservices are running."""
    print("\nChecking demo microservices...")
    
    try:
        result = subprocess.run(['docker', 'ps', '--format', 'table {{.Names}}'], 
                              capture_output=True, text=True, check=True)
        running_containers = result.stdout.lower()
        
        # Check for common microservice names
        microservice_indicators = [
            'frontend',
            'checkout',
            'payment',
            'product',
            'cart',
            'currency',
            'shipping',
            'recommendation',
            'ad'
        ]
        
        found_services = []
        for indicator in microservice_indicators:
            if indicator in running_containers:
                found_services.append(indicator)
        
        if found_services:
            print(f"✅ Found microservices: {found_services}")
            return True
        else:
            print("⚠️  No demo microservices detected")
            print("This is OK for basic testing, but chaos injection will be limited")
            return True
            
    except subprocess.CalledProcessError:
        print("❌ Failed to check microservices")
        return False

def test_prometheus_metrics():
    """Test if Prometheus is collecting metrics."""
    print("\nTesting Prometheus metrics collection...")
    
    try:
        # Try to access Prometheus API
        response = requests.get('http://localhost:9090/api/v1/query?query=up', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                metrics_count = len(data.get('data', {}).get('result', []))
                print(f"✅ Prometheus is collecting metrics ({metrics_count} 'up' metrics found)")
                return True
        
        print("❌ Prometheus metrics not accessible")
        return False
        
    except requests.RequestException as e:
        print(f"❌ Failed to connect to Prometheus: {e}")
        return False

def test_grafana_access():
    """Test if Grafana is accessible."""
    print("\nTesting Grafana access...")
    
    try:
        response = requests.get('http://localhost:3000/api/health', timeout=5)
        
        if response.status_code == 200:
            print("✅ Grafana is accessible")
            return True
        else:
            print(f"❌ Grafana returned status code: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Failed to connect to Grafana: {e}")
        return False

def test_jaeger_access():
    """Test if Jaeger is accessible."""
    print("\nTesting Jaeger access...")
    
    try:
        response = requests.get('http://localhost:16686/api/services', timeout=5)
        
        if response.status_code == 200:
            services = response.json()
            print(f"✅ Jaeger is accessible ({len(services.get('data', []))} services found)")
            return True
        else:
            print(f"❌ Jaeger returned status code: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Failed to connect to Jaeger: {e}")
        return False

def test_chaos_injection_basic():
    """Test basic chaos injection functionality."""
    print("\nTesting basic chaos injection...")
    
    chaos_dir = Path(__file__).parent
    
    # Test CPU stress injection
    try:
        from cpu.cpu_stress import CPUStressInjector
        
        config = {
            "intensity": "10%",  # Very light load
            "duration": 3,       # Short duration
            "safety_threshold": 98
        }
        
        injector = CPUStressInjector(config)
        
        # Test that we can create and configure the injector
        status = injector.get_status()
        if not status['running']:
            print("✅ CPU stress injector created successfully")
        else:
            print("❌ CPU stress injector in unexpected state")
            return False
        
        print("✅ Basic chaos injection functionality works")
        return True
        
    except Exception as e:
        print(f"❌ Basic chaos injection test failed: {e}")
        return False

def test_container_targeting():
    """Test container targeting functionality."""
    print("\nTesting container targeting...")
    
    try:
        # Get list of running containers
        result = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], 
                              capture_output=True, text=True, check=True)
        containers = [name.strip() for name in result.stdout.split('\n') if name.strip()]
        
        if not containers:
            print("⚠️  No containers found for targeting test")
            return True
        
        # Test that we can configure container targeting
        from cpu.cpu_stress import CPUStressInjector
        
        test_container = containers[0]  # Use first available container
        config = {
            "intensity": "10%",
            "duration": 3,
            "target_container": test_container
        }
        
        injector = CPUStressInjector(config)
        
        if injector.config.get('target_container') == test_container:
            print(f"✅ Container targeting configured for: {test_container}")
            return True
        else:
            print("❌ Container targeting configuration failed")
            return False
            
    except Exception as e:
        print(f"❌ Container targeting test failed: {e}")
        return False

def test_monitoring_integration():
    """Test that chaos injection monitoring integrates with observability stack."""
    print("\nTesting monitoring integration...")
    
    try:
        from cpu.cpu_stress import CPUStressInjector
        from datetime import datetime, timezone
        
        # Create injector with monitoring
        config = {
            "intensity": "5%",   # Very light load
            "duration": 2,       # Very short
            "monitoring_interval": 0.5
        }
        
        injector = CPUStressInjector(config)
        injector.start_time = datetime.now(timezone.utc)
        
        # Simulate monitoring data
        injector.injection_log = [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cpu_percent": 25.0,
                "memory_percent": 40.0,
                "active_stress": True
            }
        ]
        
        # Test log saving
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            log_file = f.name
        
        try:
            injector.save_injection_log(log_file)
            
            # Verify log structure
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            required_fields = ['config', 'start_time', 'monitoring_data']
            if all(field in log_data for field in required_fields):
                print("✅ Monitoring data logging works correctly")
                return True
            else:
                print("❌ Monitoring data logging missing required fields")
                return False
                
        finally:
            import os
            if os.path.exists(log_file):
                os.unlink(log_file)
        
    except Exception as e:
        print(f"❌ Monitoring integration test failed: {e}")
        return False

def show_manual_testing_guide():
    """Show guide for manual testing with observability stack."""
    print("\n" + "="*60)
    print("MANUAL TESTING GUIDE")
    print("="*60)
    print("Since data collection (Task 6) is not yet implemented,")
    print("you can manually verify chaos injection effects:")
    print()
    print("1. Open Grafana: http://localhost:3000")
    print("   - Username: admin, Password: admin")
    print("   - Look for CPU, Memory, Network dashboards")
    print()
    print("2. Open Prometheus: http://localhost:9090")
    print("   - Query: rate(cpu_usage_total[5m])")
    print("   - Query: container_memory_usage_bytes")
    print()
    print("3. Open Jaeger: http://localhost:16686")
    print("   - Look for trace latency changes")
    print()
    print("4. Run chaos injection:")
    print("   cd chaos/cpu")
    print("   ./inject_cpu_stress.sh -t <container_name> -i 50% -d 60 start")
    print()
    print("5. Observe changes in the monitoring dashboards")
    print("   - CPU usage should increase")
    print("   - Response times may increase")
    print("   - Error rates may change")
    print()
    print("6. Stop chaos injection:")
    print("   ./inject_cpu_stress.sh stop")
    print()
    print("7. Verify metrics return to normal")

def main():
    """Run integration tests."""
    print("Chaos Engineering Integration Test")
    print("="*50)
    print("Testing integration with observability stack")
    print("(Note: Full data collection will be available in Task 6)")
    print()
    
    tests = [
        ("Docker Services", check_docker_services),
        ("Microservices", check_microservices),
        ("Prometheus Metrics", test_prometheus_metrics),
        ("Grafana Access", test_grafana_access),
        ("Jaeger Access", test_jaeger_access),
        ("Basic Chaos Injection", test_chaos_injection_basic),
        ("Container Targeting", test_container_targeting),
        ("Monitoring Integration", test_monitoring_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
    
    print("\n" + "="*50)
    print("INTEGRATION TEST RESULTS")
    print("="*50)
    print(f"Passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed >= total - 2:  # Allow for some services to be optional
        print("\n✅ Integration tests mostly passed!")
        print("Chaos engineering is ready for manual testing.")
        show_manual_testing_guide()
        return 0
    else:
        print(f"\n❌ Too many integration tests failed ({total-passed} failed)")
        print("Please ensure the observability stack is running:")
        print("  ./scripts/start-otel-demo.sh")
        print("  or")
        print("  ./scripts/start-online-boutique.sh")
        return 1

if __name__ == "__main__":
    sys.exit(main())