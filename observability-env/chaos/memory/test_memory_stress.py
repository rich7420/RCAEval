#!/usr/bin/env python3
"""
Test script for Memory stress injection module.
"""

import json
import time
import tempfile
import os
import sys
import shutil
from memory_stress import MemoryStressInjector

def test_config_validation():
    """Test configuration validation."""
    print("Testing memory configuration validation...")
    
    # Valid configuration
    valid_config = {
        "size": "50%",
        "duration": 10
    }
    
    try:
        injector = MemoryStressInjector(valid_config)
        print("✓ Valid configuration accepted")
    except Exception as e:
        print(f"✗ Valid configuration rejected: {e}")
        return False
    
    # Invalid size percentage
    invalid_config = {
        "size": "150%",  # Invalid percentage
        "duration": 10
    }
    
    try:
        injector = MemoryStressInjector(invalid_config)
        print("✗ Invalid size accepted (should be rejected)")
        return False
    except ValueError:
        print("✓ Invalid size properly rejected")
    
    # Missing required field
    incomplete_config = {
        "size": "50%"
        # Missing duration
    }
    
    try:
        injector = MemoryStressInjector(incomplete_config)
        print("✗ Incomplete configuration accepted (should be rejected)")
        return False
    except ValueError:
        print("✓ Incomplete configuration properly rejected")
    
    return True

def test_memory_size_calculation():
    """Test memory size calculation logic."""
    print("\nTesting memory size calculation...")
    
    # Test percentage-based calculation
    config = {"size": "50%", "duration": 10}
    injector = MemoryStressInjector(config)
    size = injector._calculate_memory_size()
    
    # Should return a size with unit
    if size.endswith(('K', 'M', 'G')):
        print(f"✓ Percentage calculation correct: {size}")
    else:
        print(f"✗ Percentage calculation incorrect: {size}")
        return False
    
    # Test absolute size
    config = {"size": "1G", "duration": 10}
    injector = MemoryStressInjector(config)
    size = injector._calculate_memory_size()
    
    if size == "1G":
        print("✓ Absolute size calculation correct")
    else:
        print(f"✗ Absolute size calculation incorrect: got {size}, expected 1G")
        return False
    
    return True

def test_command_building():
    """Test stress-ng command building."""
    print("\nTesting memory command building...")
    
    config = {
        "size": "512M",
        "duration": 30,
        "workers": 2,
        "method": "all"
    }
    
    injector = MemoryStressInjector(config)
    cmd = injector._build_stress_command()
    
    # Check basic command structure
    if cmd[0] != 'stress-ng':
        print(f"✗ Command doesn't start with stress-ng: {cmd}")
        return False
    
    if '--vm' not in cmd:
        print(f"✗ Command missing --vm parameter: {cmd}")
        return False
    
    if '--vm-bytes' not in cmd:
        print(f"✗ Command missing --vm-bytes parameter: {cmd}")
        return False
    
    if '--timeout' not in cmd:
        print(f"✗ Command missing --timeout parameter: {cmd}")
        return False
    
    print(f"✓ Command built correctly: {' '.join(cmd)}")
    return True

def test_short_injection():
    """Test a very short memory stress injection."""
    print("\nTesting short memory stress injection...")
    
    config = {
        "size": "100M",  # Small allocation for testing
        "duration": 5,   # Very short duration
        "workers": 1,
        "safety_threshold": 98,  # High threshold for testing
        "monitoring_interval": 0.5  # Fast monitoring
    }
    
    injector = MemoryStressInjector(config)
    
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
        "size": "50M",   # Small allocation
        "duration": 60,  # Long duration
        "workers": 1,
        "safety_threshold": 1,  # Impossibly low threshold
        "monitoring_interval": 0.1  # Fast monitoring
    }
    
    injector = MemoryStressInjector(config)
    
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

def test_configuration_file():
    """Test configuration file loading."""
    print("\nTesting configuration file loading...")
    
    # Create temporary config file
    config_data = {
        "size": "256M",
        "duration": 5,
        "workers": 1,
        "safety_threshold": 95
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        config_file = f.name
    
    try:
        from memory_stress import create_memory_injector
        injector = create_memory_injector(config_file)
        
        if injector.config['size'] != '256M':
            print("✗ Configuration not loaded correctly")
            return False
        
        print("✓ Configuration file loaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Failed to load configuration file: {e}")
        return False
    finally:
        if os.path.exists(config_file):
            os.unlink(config_file)

def main():
    """Run all tests."""
    print("Memory Stress Injection Module Tests")
    print("=" * 40)
    
    tests = [
        test_config_validation,
        test_memory_size_calculation,
        test_command_building,
        test_short_injection,
        test_safety_mechanisms,
        test_configuration_file
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