#!/usr/bin/env python3
"""
Quick test script for chaos engineering modules.
This script runs basic functionality tests without requiring root privileges or system tools.
"""

import sys
import os
import json
import tempfile
import time
from pathlib import Path

# Add chaos modules to path
chaos_dir = Path(__file__).parent
sys.path.insert(0, str(chaos_dir / 'cpu'))
sys.path.insert(0, str(chaos_dir / 'memory'))
sys.path.insert(0, str(chaos_dir / 'disk'))
sys.path.insert(0, str(chaos_dir / 'network'))
sys.path.insert(0, str(chaos_dir / 'socket'))

def quick_test_cpu():
    """Quick test for CPU stress module."""
    print("Testing CPU stress module...")
    
    try:
        from cpu_stress import CPUStressInjector
        
        # Test configuration validation
        config = {"intensity": "50%", "duration": 5}
        injector = CPUStressInjector(config)
        
        # Test worker calculation
        workers = injector._calculate_workers()
        assert workers > 0, "Worker calculation failed"
        
        # Test command building
        cmd = injector._build_stress_command()
        assert cmd[0] == 'stress-ng', "Command building failed"
        assert '--cpu' in cmd, "CPU parameter missing"
        
        # Test status
        status = injector.get_status()
        assert 'running' in status, "Status missing running field"
        assert 'intensity' in status, "Status missing intensity field"
        
        print("✓ CPU stress module basic tests passed")
        return True
        
    except Exception as e:
        print(f"✗ CPU stress module test failed: {e}")
        return False

def quick_test_memory():
    """Quick test for memory stress module."""
    print("Testing memory stress module...")
    
    try:
        from memory_stress import MemoryStressInjector
        
        # Test configuration validation
        config = {"size": "256M", "duration": 5}
        injector = MemoryStressInjector(config)
        
        # Test memory size calculation
        size = injector._calculate_memory_size()
        assert size == "256M", f"Size calculation failed: {size}"
        
        # Test command building
        cmd = injector._build_stress_command()
        assert cmd[0] == 'stress-ng', "Command building failed"
        assert '--vm' in cmd, "VM parameter missing"
        
        # Test status
        status = injector.get_status()
        assert 'running' in status, "Status missing running field"
        assert 'memory_size' in status, "Status missing memory_size field"
        
        print("✓ Memory stress module basic tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Memory stress module test failed: {e}")
        return False

def quick_test_disk():
    """Quick test for disk stress module."""
    print("Testing disk stress module...")
    
    try:
        from disk_stress import DiskStressInjector
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Test configuration validation
            config = {"duration": 5, "temp_path": temp_dir, "file_size": "100M"}
            injector = DiskStressInjector(config)
            
            # Test disk usage monitoring
            usage = injector._get_disk_usage(temp_dir)
            assert 'used_percent' in usage, "Disk usage monitoring failed"
            assert usage['used_percent'] >= 0, "Invalid disk usage percentage"
            
            # Test command building
            cmd = injector._build_stress_command()
            assert cmd[0] == 'stress-ng', "Command building failed"
            assert '--hdd' in cmd, "HDD parameter missing"
            
            # Test status
            status = injector.get_status()
            assert 'running' in status, "Status missing running field"
            assert 'file_size' in status, "Status missing file_size field"
            
            print("✓ Disk stress module basic tests passed")
            return True
            
        finally:
            # Clean up
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        
    except Exception as e:
        print(f"✗ Disk stress module test failed: {e}")
        return False

def quick_test_network_delay():
    """Quick test for network delay module."""
    print("Testing network delay module...")
    
    try:
        from network_delay import NetworkDelayInjector
        
        # Test configuration validation (use generic interface for testing)
        config = {"delay": "50ms", "duration": 5, "interface": "eth0"}
        injector = NetworkDelayInjector(config)
        
        # Test command building
        commands = injector._build_tc_commands()
        assert len(commands) > 0, "No commands generated"
        assert commands[0][0] == 'tc', "Command building failed"
        assert 'netem' in commands[0], "Netem parameter missing"
        
        # Test status
        status = injector.get_status()
        assert 'running' in status, "Status missing running field"
        assert 'delay' in status, "Status missing delay field"
        
        print("✓ Network delay module basic tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Network delay module test failed: {e}")
        return False

def quick_test_packet_loss():
    """Quick test for packet loss module."""
    print("Testing packet loss module...")
    
    try:
        from packet_loss import PacketLossInjector
        
        # Test configuration validation (use generic interface for testing)
        config = {"loss_percent": 5.0, "duration": 5, "interface": "eth0"}
        injector = PacketLossInjector(config)
        
        # Test command building
        commands = injector._build_tc_commands()
        assert len(commands) > 0, "No commands generated"
        assert commands[0][0] == 'tc', "Command building failed"
        assert 'loss' in commands[0], "Loss parameter missing"
        
        # Test status
        status = injector.get_status()
        assert 'running' in status, "Status missing running field"
        assert 'loss_percent' in status, "Status missing loss_percent field"
        
        print("✓ Packet loss module basic tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Packet loss module test failed: {e}")
        return False

def quick_test_socket_failure():
    """Quick test for socket failure module."""
    print("Testing socket failure module...")
    
    try:
        from socket_failure import SocketFailureInjector
        
        # Test configuration validation
        config = {
            "duration": 5,
            "failure_pattern": "selective",
            "target_ports": [8080, 9090]
        }
        injector = SocketFailureInjector(config)
        
        # Test command building
        commands = injector._build_iptables_commands()
        assert len(commands) > 0, "No commands generated"
        assert commands[0][0] == 'iptables', "Command building failed"
        assert 'DROP' in commands[0], "DROP action missing"
        
        # Test status
        status = injector.get_status()
        assert 'running' in status, "Status missing running field"
        assert 'failure_pattern' in status, "Status missing failure_pattern field"
        
        print("✓ Socket failure module basic tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Socket failure module test failed: {e}")
        return False

def test_configuration_files():
    """Test configuration file loading for all modules."""
    print("Testing configuration file loading...")
    
    try:
        # Test CPU config
        cpu_config = {"intensity": "75%", "duration": 10}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cpu_config, f)
            cpu_config_file = f.name
        
        try:
            from cpu_stress import create_cpu_injector
            injector = create_cpu_injector(cpu_config_file)
            assert injector.config['intensity'] == '75%', "CPU config loading failed"
        finally:
            os.unlink(cpu_config_file)
        
        # Test memory config
        memory_config = {"size": "512M", "duration": 10}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(memory_config, f)
            memory_config_file = f.name
        
        try:
            from memory_stress import create_memory_injector
            injector = create_memory_injector(memory_config_file)
            assert injector.config['size'] == '512M', "Memory config loading failed"
        finally:
            os.unlink(memory_config_file)
        
        print("✓ Configuration file loading tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Configuration file loading test failed: {e}")
        return False

def test_monitoring_and_logging():
    """Test monitoring and logging functionality."""
    print("Testing monitoring and logging...")
    
    try:
        from cpu_stress import CPUStressInjector
        
        # Create injector
        config = {"intensity": "25%", "duration": 5}
        injector = CPUStressInjector(config)
        
        # Test monitoring data structure
        from datetime import datetime, timezone
        injector.start_time = datetime.now(timezone.utc)
        injector.injection_log = [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "cpu_percent": 50.0,
                "memory_percent": 30.0,
                "active_stress": True
            }
        ]
        
        # Test monitoring data retrieval
        data = injector.get_monitoring_data()
        assert len(data) == 1, "Monitoring data retrieval failed"
        assert 'timestamp' in data[0], "Monitoring data missing timestamp"
        assert 'cpu_percent' in data[0], "Monitoring data missing cpu_percent"
        
        # Test log saving
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            log_file = f.name
        
        try:
            injector.save_injection_log(log_file)
            
            # Verify log file
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            assert 'config' in log_data, "Log missing config"
            assert 'start_time' in log_data, "Log missing start_time"
            assert 'monitoring_data' in log_data, "Log missing monitoring_data"
            
        finally:
            os.unlink(log_file)
        
        print("✓ Monitoring and logging tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Monitoring and logging test failed: {e}")
        return False

def test_container_targeting():
    """Test container targeting functionality."""
    print("Testing container targeting...")
    
    try:
        # Test all modules support container targeting
        modules = [
            ('cpu_stress', 'CPUStressInjector', {"intensity": "50%", "duration": 5}),
            ('memory_stress', 'MemoryStressInjector', {"size": "100M", "duration": 5}),
            ('disk_stress', 'DiskStressInjector', {"duration": 5, "temp_path": "/tmp"}),
            ('network_delay', 'NetworkDelayInjector', {"delay": "10ms", "duration": 5}),
            ('packet_loss', 'PacketLossInjector', {"loss_percent": 1.0, "duration": 5}),
            ('socket_failure', 'SocketFailureInjector', {"duration": 5, "failure_pattern": "selective", "target_ports": [8080]})
        ]
        
        for module_name, class_name, base_config in modules:
            module = __import__(module_name)
            injector_class = getattr(module, class_name)
            
            # Test with container targeting
            config = base_config.copy()
            config['target_container'] = 'test_container'
            
            injector = injector_class(config)
            assert injector.config.get('target_container') == 'test_container', f"{module_name} container targeting failed"
        
        print("✓ Container targeting tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Container targeting test failed: {e}")
        return False

def main():
    """Run quick tests for all chaos engineering modules."""
    print("Chaos Engineering Quick Test Suite")
    print("=" * 50)
    print("Running basic functionality tests (no root required)")
    print("=" * 50)
    
    tests = [
        ("CPU Stress", quick_test_cpu),
        ("Memory Stress", quick_test_memory),
        ("Disk Stress", quick_test_disk),
        ("Network Delay", quick_test_network_delay),
        ("Packet Loss", quick_test_packet_loss),
        ("Socket Failure", quick_test_socket_failure),
        ("Configuration Files", test_configuration_files),
        ("Monitoring & Logging", test_monitoring_and_logging),
        ("Container Targeting", test_container_targeting)
    ]
    
    passed = 0
    total = len(tests)
    failed_tests = []
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed_tests.append(test_name)
        except Exception as e:
            failed_tests.append(test_name)
            print(f"✗ {test_name} test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print("QUICK TEST RESULTS")
    print("=" * 50)
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if failed_tests:
        print(f"\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test}")
    
    if passed == total:
        print("\n✅ All quick tests passed!")
        print("Basic functionality is working correctly.")
        print("\nTo run full system tests (requires root/sudo):")
        print("  python3 test_chaos_suite.py")
        return 0
    else:
        print(f"\n❌ {total - passed} tests failed.")
        print("Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())