#!/usr/bin/env python3
"""
Test script for Disk I/O stress injection module.
"""

import json
import time
import tempfile
import os
import sys
import shutil
from disk_stress import DiskStressInjector

def test_config_validation():
    """Test configuration validation."""
    print("Testing disk configuration validation...")
    
    # Valid configuration
    valid_config = {
        "duration": 10,
        "temp_path": "/tmp"
    }
    
    try:
        injector = DiskStressInjector(valid_config)
        print("✓ Valid configuration accepted")
    except Exception as e:
        print(f"✗ Valid configuration rejected: {e}")
        return False
    
    # Invalid duration
    invalid_config = {
        "duration": -5,  # Invalid duration
        "temp_path": "/tmp"
    }
    
    try:
        injector = DiskStressInjector(invalid_config)
        print("✗ Invalid duration accepted (should be rejected)")
        return False
    except ValueError:
        print("✓ Invalid duration properly rejected")
    
    # Invalid temp path
    invalid_path_config = {
        "duration": 10,
        "temp_path": "/nonexistent/path"
    }
    
    try:
        injector = DiskStressInjector(invalid_path_config)
        print("✗ Invalid temp path accepted (should be rejected)")
        return False
    except ValueError:
        print("✓ Invalid temp path properly rejected")
    
    return True

def test_disk_usage_monitoring():
    """Test disk usage calculation."""
    print("\nTesting disk usage monitoring...")
    
    config = {"duration": 10, "temp_path": "/tmp"}
    injector = DiskStressInjector(config)
    
    # Test disk usage calculation
    usage = injector._get_disk_usage("/tmp")
    
    required_fields = ['total_gb', 'used_gb', 'free_gb', 'used_percent']
    for field in required_fields:
        if field not in usage:
            print(f"✗ Missing field in disk usage: {field}")
            return False
    
    if usage['total_gb'] <= 0:
        print("✗ Invalid total disk size")
        return False
    
    if usage['used_percent'] < 0 or usage['used_percent'] > 100:
        print("✗ Invalid disk usage percentage")
        return False
    
    print(f"✓ Disk usage calculation correct: {usage['used_percent']:.1f}% used")
    return True

def test_command_building():
    """Test stress-ng command building."""
    print("\nTesting disk command building...")
    
    config = {
        "duration": 30,
        "workers": 2,
        "file_size": "512M",
        "temp_path": "/tmp",
        "method": "sync"
    }
    
    injector = DiskStressInjector(config)
    cmd = injector._build_stress_command()
    
    # Check basic command structure
    if cmd[0] != 'stress-ng':
        print(f"✗ Command doesn't start with stress-ng: {cmd}")
        return False
    
    if '--hdd' not in cmd:
        print(f"✗ Command missing --hdd parameter: {cmd}")
        return False
    
    if '--hdd-bytes' not in cmd:
        print(f"✗ Command missing --hdd-bytes parameter: {cmd}")
        return False
    
    if '--temp-path' not in cmd:
        print(f"✗ Command missing --temp-path parameter: {cmd}")
        return False
    
    if '--timeout' not in cmd:
        print(f"✗ Command missing --timeout parameter: {cmd}")
        return False
    
    print(f"✓ Command built correctly: {' '.join(cmd)}")
    return True

def test_short_injection():
    """Test a very short disk stress injection."""
    print("\nTesting short disk stress injection...")
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    
    try:
        config = {
            "duration": 5,       # Very short duration
            "workers": 1,
            "file_size": "50M",  # Small file for testing
            "temp_path": temp_dir,
            "safety_threshold": 98,  # High threshold for testing
            "monitoring_interval": 0.5  # Fast monitoring
        }
        
        injector = DiskStressInjector(config)
        
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
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_safety_mechanisms():
    """Test safety threshold mechanism (simulation)."""
    print("\nTesting safety mechanisms...")
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create injector with very low safety threshold for testing
        config = {
            "duration": 60,      # Long duration
            "workers": 1,
            "file_size": "10M",  # Small file
            "temp_path": temp_dir,
            "safety_threshold": 1,  # Impossibly low threshold
            "monitoring_interval": 0.1  # Fast monitoring
        }
        
        injector = DiskStressInjector(config)
        
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
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_cleanup_functionality():
    """Test temporary file cleanup."""
    print("\nTesting cleanup functionality...")
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    
    try:
        config = {
            "duration": 3,       # Short duration
            "workers": 1,
            "file_size": "10M",
            "temp_path": temp_dir
        }
        
        injector = DiskStressInjector(config)
        
        # Create some fake stress-ng files
        fake_files = [
            os.path.join(temp_dir, "stress-ng-hdd-12345"),
            os.path.join(temp_dir, "stress-ng-hdd-67890")
        ]
        
        for fake_file in fake_files:
            with open(fake_file, 'w') as f:
                f.write("fake stress file")
        
        # Test cleanup
        injector._cleanup_temp_files()
        
        # Check if files were cleaned up
        remaining_files = [f for f in fake_files if os.path.exists(f)]
        if remaining_files:
            print(f"✗ Cleanup failed, remaining files: {remaining_files}")
            return False
        
        print("✓ Cleanup functionality works correctly")
        return True
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_configuration_file():
    """Test configuration file loading."""
    print("\nTesting configuration file loading...")
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create temporary config file
        config_data = {
            "duration": 5,
            "workers": 1,
            "file_size": "100M",
            "temp_path": temp_dir,
            "safety_threshold": 95
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            from disk_stress import create_disk_injector
            injector = create_disk_injector(config_file)
            
            if injector.config['file_size'] != '100M':
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
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    """Run all tests."""
    print("Disk I/O Stress Injection Module Tests")
    print("=" * 40)
    
    tests = [
        test_config_validation,
        test_disk_usage_monitoring,
        test_command_building,
        test_short_injection,
        test_safety_mechanisms,
        test_cleanup_functionality,
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