#!/usr/bin/env python3
"""
Test script for Network chaos injection modules (delay and packet loss).
"""

import json
import time
import tempfile
import os
import sys
import subprocess
from network_delay import NetworkDelayInjector
from packet_loss import PacketLossInjector

def test_delay_config_validation():
    """Test network delay configuration validation."""
    print("Testing network delay configuration validation...")
    
    # Valid configuration
    valid_config = {
        "delay": "100ms",
        "duration": 10
    }
    
    try:
        injector = NetworkDelayInjector(valid_config)
        print("✓ Valid delay configuration accepted")
    except Exception as e:
        print(f"✗ Valid delay configuration rejected: {e}")
        return False
    
    # Invalid delay format
    invalid_config = {
        "delay": "invalid",  # Invalid format
        "duration": 10
    }
    
    try:
        injector = NetworkDelayInjector(invalid_config)
        print("✗ Invalid delay format accepted (should be rejected)")
        return False
    except ValueError:
        print("✓ Invalid delay format properly rejected")
    
    # Missing required field
    incomplete_config = {
        "delay": "50ms"
        # Missing duration
    }
    
    try:
        injector = NetworkDelayInjector(incomplete_config)
        print("✗ Incomplete delay configuration accepted (should be rejected)")
        return False
    except ValueError:
        print("✓ Incomplete delay configuration properly rejected")
    
    return True

def test_loss_config_validation():
    """Test packet loss configuration validation."""
    print("\nTesting packet loss configuration validation...")
    
    # Valid configuration
    valid_config = {
        "loss_percent": 5.0,
        "duration": 10
    }
    
    try:
        injector = PacketLossInjector(valid_config)
        print("✓ Valid loss configuration accepted")
    except Exception as e:
        print(f"✗ Valid loss configuration rejected: {e}")
        return False
    
    # Invalid loss percentage
    invalid_config = {
        "loss_percent": 150.0,  # Invalid percentage
        "duration": 10
    }
    
    try:
        injector = PacketLossInjector(invalid_config)
        print("✗ Invalid loss percentage accepted (should be rejected)")
        return False
    except ValueError:
        print("✓ Invalid loss percentage properly rejected")
    
    return True

def test_delay_command_building():
    """Test network delay command building."""
    print("\nTesting delay command building...")
    
    config = {
        "delay": "50ms",
        "jitter": "5ms",
        "duration": 30,
        "interface": "lo"  # Use loopback for testing
    }
    
    injector = NetworkDelayInjector(config)
    
    try:
        commands = injector._build_tc_commands()
        
        if not commands:
            print("✗ No commands generated")
            return False
        
        # Check first command structure
        cmd = commands[0]
        if cmd[0] != 'tc':
            print(f"✗ Command doesn't start with tc: {cmd}")
            return False
        
        if 'netem' not in cmd:
            print(f"✗ Command missing netem: {cmd}")
            return False
        
        if 'delay' not in cmd:
            print(f"✗ Command missing delay parameter: {cmd}")
            return False
        
        print(f"✓ Delay command built correctly: {' '.join(cmd)}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to build delay command: {e}")
        return False

def test_loss_command_building():
    """Test packet loss command building."""
    print("\nTesting loss command building...")
    
    config = {
        "loss_percent": 10.0,
        "correlation": 25,
        "duration": 30,
        "interface": "lo",  # Use loopback for testing
        "pattern": "random"
    }
    
    injector = PacketLossInjector(config)
    
    try:
        commands = injector._build_tc_commands()
        
        if not commands:
            print("✗ No commands generated")
            return False
        
        # Check first command structure
        cmd = commands[0]
        if cmd[0] != 'tc':
            print(f"✗ Command doesn't start with tc: {cmd}")
            return False
        
        if 'netem' not in cmd:
            print(f"✗ Command missing netem: {cmd}")
            return False
        
        if 'loss' not in cmd:
            print(f"✗ Command missing loss parameter: {cmd}")
            return False
        
        print(f"✓ Loss command built correctly: {' '.join(cmd)}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to build loss command: {e}")
        return False

def test_interface_detection():
    """Test network interface detection."""
    print("\nTesting network interface detection...")
    
    config = {
        "delay": "10ms",
        "duration": 5,
        "interface": "nonexistent_interface"
    }
    
    injector = NetworkDelayInjector(config)
    
    try:
        # This should try to find a fallback interface
        interface = injector._get_network_interface()
        
        # Should find some interface (like lo)
        if interface:
            print(f"✓ Interface detection works: {interface}")
            return True
        else:
            print("✗ No interface detected")
            return False
            
    except Exception as e:
        print(f"✗ Interface detection failed: {e}")
        return False

def test_delay_status_monitoring():
    """Test delay injection status and monitoring."""
    print("\nTesting delay status and monitoring...")
    
    config = {
        "delay": "1ms",      # Very small delay for testing
        "duration": 3,       # Short duration
        "interface": "lo",   # Use loopback
        "monitoring_interval": 0.5
    }
    
    injector = NetworkDelayInjector(config)
    
    # Test status before starting
    status = injector.get_status()
    if status['running']:
        print("✗ Injector reports running before start")
        return False
    
    print("✓ Initial status correct")
    
    # Test monitoring data collection (without actually starting injection)
    # This tests the monitoring thread functionality
    injector.start_time = injector.start_time or time.time()
    injector.stop_monitoring.clear()
    
    # Simulate some monitoring
    import threading
    monitor_thread = threading.Thread(target=injector._monitor_network_latency)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    time.sleep(1)  # Let it collect some data
    injector.stop_monitoring.set()
    monitor_thread.join(timeout=2)
    
    monitoring_data = injector.get_monitoring_data()
    if len(monitoring_data) == 0:
        print("✗ No monitoring data collected")
        return False
    
    print(f"✓ Monitoring data collected: {len(monitoring_data)} points")
    return True

def test_loss_status_monitoring():
    """Test packet loss injection status and monitoring."""
    print("\nTesting loss status and monitoring...")
    
    config = {
        "loss_percent": 1.0,  # Very small loss for testing
        "duration": 3,        # Short duration
        "interface": "lo",    # Use loopback
        "monitoring_interval": 0.5
    }
    
    injector = PacketLossInjector(config)
    
    # Test status before starting
    status = injector.get_status()
    if status['running']:
        print("✗ Injector reports running before start")
        return False
    
    print("✓ Initial status correct")
    
    # Test monitoring data collection (without actually starting injection)
    injector.start_time = injector.start_time or time.time()
    injector.stop_monitoring.clear()
    
    # Simulate some monitoring
    import threading
    monitor_thread = threading.Thread(target=injector._monitor_packet_loss)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    time.sleep(1)  # Let it collect some data
    injector.stop_monitoring.set()
    monitor_thread.join(timeout=2)
    
    monitoring_data = injector.get_monitoring_data()
    if len(monitoring_data) == 0:
        print("✗ No monitoring data collected")
        return False
    
    print(f"✓ Monitoring data collected: {len(monitoring_data)} points")
    return True

def test_configuration_files():
    """Test configuration file loading."""
    print("\nTesting configuration file loading...")
    
    # Test delay configuration
    delay_config_data = {
        "delay": "25ms",
        "jitter": "5ms",
        "duration": 5,
        "interface": "lo"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(delay_config_data, f)
        delay_config_file = f.name
    
    try:
        from network_delay import create_delay_injector
        delay_injector = create_delay_injector(delay_config_file)
        
        if delay_injector.config['delay'] != '25ms':
            print("✗ Delay configuration not loaded correctly")
            return False
        
        print("✓ Delay configuration file loaded successfully")
        
    except Exception as e:
        print(f"✗ Failed to load delay configuration file: {e}")
        return False
    finally:
        if os.path.exists(delay_config_file):
            os.unlink(delay_config_file)
    
    # Test loss configuration
    loss_config_data = {
        "loss_percent": 2.5,
        "correlation": 30,
        "duration": 5,
        "interface": "lo"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(loss_config_data, f)
        loss_config_file = f.name
    
    try:
        from packet_loss import create_loss_injector
        loss_injector = create_loss_injector(loss_config_file)
        
        if loss_injector.config['loss_percent'] != 2.5:
            print("✗ Loss configuration not loaded correctly")
            return False
        
        print("✓ Loss configuration file loaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Failed to load loss configuration file: {e}")
        return False
    finally:
        if os.path.exists(loss_config_file):
            os.unlink(loss_config_file)

def test_log_saving():
    """Test log saving functionality."""
    print("\nTesting log saving functionality...")
    
    # Test delay log saving
    config = {
        "delay": "10ms",
        "duration": 1,
        "interface": "lo"
    }
    
    injector = NetworkDelayInjector(config)
    injector.start_time = time.time()
    
    # Add some fake monitoring data
    injector.injection_log = [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "bytes_sent": 1000,
            "bytes_recv": 2000,
            "active_delay": True
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        log_file = f.name
    
    try:
        injector.save_injection_log(log_file)
        
        # Verify log file
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        
        required_fields = ['config', 'start_time', 'monitoring_data']
        for field in required_fields:
            if field not in log_data:
                print(f"✗ Log file missing required field: {field}")
                return False
        
        print("✓ Log saving functionality works correctly")
        return True
        
    finally:
        if os.path.exists(log_file):
            os.unlink(log_file)

def check_tc_availability():
    """Check if tc (traffic control) is available."""
    try:
        result = subprocess.run(['tc', '--help'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def main():
    """Run all tests."""
    print("Network Chaos Injection Module Tests")
    print("=" * 40)
    
    # Check if tc is available
    if not check_tc_availability():
        print("Warning: tc (traffic control) not available. Some tests may be limited.")
    
    tests = [
        test_delay_config_validation,
        test_loss_config_validation,
        test_delay_command_building,
        test_loss_command_building,
        test_interface_detection,
        test_delay_status_monitoring,
        test_loss_status_monitoring,
        test_configuration_files,
        test_log_saving
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