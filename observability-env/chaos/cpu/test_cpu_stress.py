#!/usr/bin/env python3
"""
Test script for CPU stress injection module.
"""

import json
import time
import tempfile
import os
import sys
from cpu_stress import CPUStressInjector

def test_config_validation():
    """Test configuration validation."""
    print("Testing configuration validation...")
    
    # Valid configuration
    valid_config = {
        "intensity": "50%",
        "duration": 10
    }
    
    try:
        injector = CPUStressInjector(valid_config)
        print("✓ Valid configuration accepted")
    except Exception as e:
        print(f"✗ Valid configuration rejected: {e}")
        return False
    
    # Invalid intensity
    invalid_config = {
        "intensity": "150%",  # Invalid percentage
        "duration": 10
    }
    
    try:
        injector = CPUStressInjector(invalid_config)
        print("✗ Invalid intensity accepted (should be rejected)")
        return False
    except ValueError:
        print("✓ Invalid intensity properly rejected")
    
    # Missing required field
    incomplete_config = {
        "intensity": "50%"
        # Missing duration
    }
    
    try:
        injector = CPUStressInjector(incomplete_config)
        print("✗ Incomplete configuration accepted (should be rejected)")
        return False
    except ValueError:
        print("✓ Incomplete configuration properly rejected")
    
    return True

def test_worker_calculation():
    """Test worker calculation logic."""
    print("\nTesting worker calculation...")
    
    # Test percentage-based calculation
    config = {"intensity": "50%", "duration": 10}
    injector = CPUStressInjector(config)
    workers = injector._calculate_workers()
    
    import psutil
    expected_workers = max(1, int(psutil.cpu_count() * 50 / 100))
    
    if workers == expected_workers:
        print(f"✓ Percentage calculation correct: {workers} workers for 50%")
    else:
        print(f"✗ Percentage calculation incorrect: got {workers}, expected {expected_workers}")
        return False
    
    # Test direct worker count
    config = {"intensity": 2, "duration": 10}
    injector = CPUStressInjector(config)
    workers = injector._calculate_workers()
    
    if workers == 2:
        print("✓ Direct worker count correct")
    else:
        print(f"✗ Direct worker count incorrect: got {workers}, expected 2")
        return False
    
    return True

def test_command_building():
    """Test stress-ng command building."""
    print("\nTesting command building...")
    
    config = {
        "intensity": "25%",
        "duration": 30,
        "method": "ackermann"
    }
    
    injector = CPUStressInjector(config)
    cmd = injector._build_stress_command()
    
    # Check basic command structure
    if cmd[0] != 'stress-ng':
        print(f"✗ Command doesn't start with stress-ng: {cmd}")
        return False
    
    if '--cpu' not in cmd:
        print(f"✗ Command missing --cpu parameter: {cmd}")
        return False
    
    if '--timeout' not in cmd:
        print(f"✗ Command missing --timeout parameter: {cmd}")
        return False
    
    if '--cpu-method' not in cmd or 'ackermann' not in cmd:
        print(f"✗ Command missing CPU method: {cmd}")
        return False
    
    print(f"✓ Command built correctly: {' '.join(cmd)}")
    return True

def test_short_injection():
    """Test a very short CPU stress injection."""
    print("\nTesting short CPU stress injection...")
    
    config = {
        "intensity": "25%",  # Light load for testing
        "duration": 5,       # Very short duration
        "safety_threshold": 98,  # High threshold for testing
        "monitoring_interval": 0.5  # Fast monitoring
    }
    
    injector = CPUStressInjector(config)
    
    # Test status before starting
    status = injector.get_status()
    if status['running']:
        print("✗ Injector reports running before start")
        return False
    
    # Start injection
    if not injector.start_injection():
        print("✗ Failed to start injection")
        return False
    
    print("✓ Injection started successfully")
    
    # Wait a moment and check status
    time.sleep(1)
    status = injector.get_status()
    
    if not status['running']:
        print("✗ Injector not running after start")
        return False
    
    print(f"✓ Injection running: {status['elapsed_seconds']:.1f}s elapsed")
    
    # Wait for completion
    while injector.get_status()['running']:
        time.sleep(0.5)
    
    print("✓ Injection completed successfully")
    
    # Check monitoring data
    monitoring_data = injector.get_monitoring_data()
    if len(monitoring_data) == 0:
        print("✗ No monitoring data collected")
        return False
    
    print(f"✓ Collected {len(monitoring_data)} monitoring data points")
    
    # Test log saving
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        log_file = f.name
    
    try:
        injector.save_injection_log(log_file)
        
        # Verify log file
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        
        if 'config' not in log_data or 'monitoring_data' not in log_data:
            print("✗ Log file missing required fields")
            return False
        
        print("✓ Log file saved and validated")
        
    finally:
        if os.path.exists(log_file):
            os.unlink(log_file)
    
    return True

def test_safety_mechanisms():
    """Test safety threshold mechanism (simulation)."""
    print("\nTesting safety mechanisms...")
    
    # Create injector with very low safety threshold for testing
    config = {
        "intensity": "10%",  # Very light load
        "duration": 60,      # Long duration
        "safety_threshold": 1,  # Impossibly low threshold
        "monitoring_interval": 0.1  # Fast monitoring
    }
    
    injector = CPUStressInjector(config)
    
    # Start injection
    if not injector.start_injection():
        print("✗ Failed to start injection for safety test")
        return False
    
    # Wait for safety mechanism to trigger
    max_wait = 10  # Maximum 10 seconds
    start_time = time.time()
    
    while injector.get_status()['running'] and (time.time() - start_time) < max_wait:
        time.sleep(0.1)
    
    if injector.get_status()['running']:
        print("✗ Safety mechanism did not trigger within timeout")
        injector.stop_injection()  # Clean up
        return False
    
    print("✓ Safety mechanism triggered correctly")
    return True

def main():
    """Run all tests."""
    print("CPU Stress Injection Module Tests")
    print("=" * 40)
    
    tests = [
        test_config_validation,
        test_worker_calculation,
        test_command_building,
        test_short_injection,
        test_safety_mechanisms
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"Test {test.__name__} failed")
        except Exception as e:
            print(f"Test {test.__name__} failed with exception: {e}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())