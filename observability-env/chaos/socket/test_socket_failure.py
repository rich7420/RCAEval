#!/usr/bin/env python3
"""
Test script for Socket/Connection failure injection module.
"""

import json
import time
import tempfile
import os
import sys
import subprocess
from socket_failure import SocketFailureInjector

def test_config_validation():
    """Test configuration validation."""
    print("Testing socket failure configuration validation...")
    
    # Valid configuration
    valid_config = {
        "duration": 10,
        "failure_pattern": "selective",
        "target_ports": [80, 443]
    }
    
    try:
        injector = SocketFailureInjector(valid_config)
        print("✓ Valid configuration accepted")
    except Exception as e:
        print(f"✗ Valid configuration rejected: {e}")
        return False
    
    # Invalid failure pattern
    invalid_config = {
        "duration": 10,
        "failure_pattern": "invalid_pattern"
    }
    
    try:
        injector = SocketFailureInjector(invalid_config)
        print("✗ Invalid failure pattern accepted (should be rejected)")
        return False
    except ValueError:
        print("✓ Invalid failure pattern properly rejected")
    
    # Invalid protocol
    invalid_protocol_config = {
        "duration": 10,
        "protocols": ["invalid_protocol"]
    }
    
    try:
        injector = SocketFailureInjector(invalid_protocol_config)
        print("✗ Invalid protocol accepted (should be rejected)")
        return False
    except ValueError:
        print("✓ Invalid protocol properly rejected")
    
    return True

def test_iptables_command_building():
    """Test iptables command building."""
    print("\nTesting iptables command building...")
    
    # Test selective blocking with ports
    config = {
        "duration": 30,
        "failure_pattern": "selective",
        "target_ports": [80, 443],
        "protocols": ["tcp"],
        "block_incoming": True,
        "block_outgoing": False
    }
    
    injector = SocketFailureInjector(config)
    commands = injector._build_iptables_commands()
    
    if not commands:
        print("✗ No commands generated")
        return False
    
    # Check command structure
    for cmd in commands:
        if cmd[0] != 'iptables':
            print(f"✗ Command doesn't start with iptables: {cmd}")
            return False
        
        if '-A' not in cmd:
            print(f"✗ Command missing -A parameter: {cmd}")
            return False
        
        if 'DROP' not in cmd:
            print(f"✗ Command missing DROP action: {cmd}")
            return False
    
    print(f"✓ Commands built correctly: {len(commands)} commands")
    return True

def test_complete_failure_commands():
    """Test complete failure command building."""
    print("\nTesting complete failure commands...")
    
    config = {
        "duration": 30,
        "failure_pattern": "complete",
        "protocols": ["tcp"],
        "block_incoming": True,
        "block_outgoing": True
    }
    
    injector = SocketFailureInjector(config)
    commands = injector._build_iptables_commands()
    
    if not commands:
        print("✗ No commands generated for complete failure")
        return False
    
    # Should have commands for both INPUT and OUTPUT chains
    input_commands = [cmd for cmd in commands if 'INPUT' in cmd]
    output_commands = [cmd for cmd in commands if 'OUTPUT' in cmd]
    
    if not input_commands:
        print("✗ No INPUT chain commands generated")
        return False
    
    if not output_commands:
        print("✗ No OUTPUT chain commands generated")
        return False
    
    print("✓ Complete failure commands built correctly")
    return True

def test_status_monitoring():
    """Test status and monitoring functionality."""
    print("\nTesting status and monitoring...")
    
    config = {
        "duration": 5,
        "failure_pattern": "selective",
        "target_ports": [8080],
        "monitoring_interval": 0.5
    }
    
    injector = SocketFailureInjector(config)
    
    # Test status before starting
    status = injector.get_status()
    if status['running']:
        print("✗ Injector reports running before start")
        return False
    
    print("✓ Initial status correct")
    
    # Test monitoring data collection (without actually starting injection)
    injector.start_time = time.time()
    injector.stop_monitoring.clear()
    
    # Simulate some monitoring
    import threading
    monitor_thread = threading.Thread(target=injector._monitor_connections)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    time.sleep(1)  # Let it collect some data
    injector.stop_monitoring.set()
    monitor_thread.join(timeout=2)
    
    monitoring_data = injector.get_monitoring_data()
    if len(monitoring_data) == 0:
        print("✗ No monitoring data collected")
        return False
    
    # Check monitoring data structure
    data_point = monitoring_data[0]
    required_fields = ['timestamp', 'connection_stats', 'bytes_sent', 'bytes_recv']
    for field in required_fields:
        if field not in data_point:
            print(f"✗ Monitoring data missing field: {field}")
            return False
    
    print(f"✓ Monitoring data collected: {len(monitoring_data)} points")
    return True

def test_rule_cleanup_simulation():
    """Test rule cleanup functionality (simulation)."""
    print("\nTesting rule cleanup simulation...")
    
    config = {
        "duration": 5,
        "failure_pattern": "selective",
        "target_ports": [9999]  # Use uncommon port
    }
    
    injector = SocketFailureInjector(config)
    
    # Simulate some active rules
    fake_rules = [
        ['iptables', '-A', 'INPUT', '-p', 'tcp', '--dport', '9999', '-j', 'DROP'],
        ['iptables', '-A', 'OUTPUT', '-p', 'tcp', '--dport', '9999', '-j', 'DROP']
    ]
    
    injector.active_rules = fake_rules
    
    # Test cleanup (this will try to run iptables -D commands)
    # We expect this to fail gracefully since the rules don't actually exist
    injector._cleanup_rules()
    
    # Rules should be cleared even if cleanup commands fail
    if injector.active_rules:
        print("✗ Rules not cleared after cleanup")
        return False
    
    print("✓ Rule cleanup simulation works correctly")
    return True

def test_configuration_file():
    """Test configuration file loading."""
    print("\nTesting configuration file loading...")
    
    # Create temporary config file
    config_data = {
        "duration": 10,
        "failure_pattern": "selective",
        "target_ports": [80, 443, 8080],
        "protocols": ["tcp"],
        "block_incoming": True,
        "block_outgoing": False
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        config_file = f.name
    
    try:
        from socket_failure import create_socket_injector
        injector = create_socket_injector(config_file)
        
        if injector.config['failure_pattern'] != 'selective':
            print("✗ Configuration not loaded correctly")
            return False
        
        if injector.config['target_ports'] != [80, 443, 8080]:
            print("✗ Target ports not loaded correctly")
            return False
        
        print("✓ Configuration file loaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Failed to load configuration file: {e}")
        return False
    finally:
        if os.path.exists(config_file):
            os.unlink(config_file)

def test_log_saving():
    """Test log saving functionality."""
    print("\nTesting log saving functionality...")
    
    config = {
        "duration": 5,
        "failure_pattern": "selective",
        "target_ports": [8080]
    }
    
    injector = SocketFailureInjector(config)
    injector.start_time = time.time()
    
    # Add some fake monitoring data
    injector.injection_log = [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "connection_stats": {"total": 10, "established": 5},
            "bytes_sent": 1000,
            "bytes_recv": 2000,
            "active_blocking": True
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

def test_container_command_building():
    """Test container-specific command building."""
    print("\nTesting container command building...")
    
    config = {
        "duration": 10,
        "failure_pattern": "selective",
        "target_ports": [80],
        "target_container": "test_container"
    }
    
    injector = SocketFailureInjector(config)
    
    # Test that container targeting is properly configured
    if injector.config['target_container'] != 'test_container':
        print("✗ Container targeting not configured correctly")
        return False
    
    # Build commands (these would be executed in container)
    commands = injector._build_iptables_commands()
    
    if not commands:
        print("✗ No commands generated for container targeting")
        return False
    
    print("✓ Container command building works correctly")
    return True

def check_iptables_availability():
    """Check if iptables is available."""
    try:
        result = subprocess.run(['iptables', '--help'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def main():
    """Run all tests."""
    print("Socket/Connection Failure Injection Module Tests")
    print("=" * 50)
    
    # Check if iptables is available
    if not check_iptables_availability():
        print("Warning: iptables not available. Some tests may be limited.")
    
    tests = [
        test_config_validation,
        test_iptables_command_building,
        test_complete_failure_commands,
        test_status_monitoring,
        test_rule_cleanup_simulation,
        test_configuration_file,
        test_log_saving,
        test_container_command_building
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