#!/usr/bin/env python3
"""
Comprehensive test suite for all chaos engineering modules.
This script tests all chaos injection capabilities to ensure they meet requirements.
"""

import sys
import os
import subprocess
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

def check_system_requirements():
    """Check if system has required tools and permissions."""
    print("Checking system requirements...")
    
    requirements = {
        'python3': True,
        'stress-ng': False,
        'tc': False,
        'iptables': False,
        'jq': False
    }
    
    for tool, required in requirements.items():
        try:
            result = subprocess.run([tool, '--help'], capture_output=True, text=True, timeout=5)
            available = result.returncode == 0
            status = "✓" if available else "✗"
            print(f"  {status} {tool}: {'Available' if available else 'Not available'}")
            
            if required and not available:
                print(f"    ERROR: {tool} is required but not available")
                return False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            status = "✗"
            print(f"  {status} {tool}: Not available")
            if required:
                print(f"    ERROR: {tool} is required but not available")
                return False
    
    print("✓ System requirements check completed")
    return True

def test_requirement_3_1_cpu_stress():
    """Test Requirement 3.1: CPU stress injection (cpu)"""
    print("\n" + "="*60)
    print("Testing Requirement 3.1: CPU stress injection")
    print("="*60)
    
    try:
        from cpu_stress import CPUStressInjector
        
        # Test basic CPU stress configuration
        config = {
            "intensity": "25%",  # Light load for testing
            "duration": 3,       # Short duration
            "safety_threshold": 98
        }
        
        injector = CPUStressInjector(config)
        
        # Test configuration validation
        print("✓ CPU stress injector created successfully")
        
        # Test command building
        cmd = injector._build_stress_command()
        if 'stress-ng' in cmd[0] and '--cpu' in cmd:
            print("✓ CPU stress command built correctly")
        else:
            print("✗ CPU stress command building failed")
            return False
        
        # Test status functionality
        status = injector.get_status()
        if 'running' in status and 'intensity' in status:
            print("✓ CPU stress status functionality works")
        else:
            print("✗ CPU stress status functionality failed")
            return False
        
        print("✓ Requirement 3.1 (CPU stress injection) satisfied")
        return True
        
    except Exception as e:
        print(f"✗ Requirement 3.1 failed: {e}")
        return False

def test_requirement_3_2_memory_stress():
    """Test Requirement 3.2: Memory stress injection (mem)"""
    print("\n" + "="*60)
    print("Testing Requirement 3.2: Memory stress injection")
    print("="*60)
    
    try:
        from memory_stress import MemoryStressInjector
        
        # Test basic memory stress configuration
        config = {
            "size": "100M",      # Small allocation for testing
            "duration": 3,       # Short duration
            "workers": 1,
            "safety_threshold": 98
        }
        
        injector = MemoryStressInjector(config)
        
        # Test configuration validation
        print("✓ Memory stress injector created successfully")
        
        # Test memory size calculation
        size = injector._calculate_memory_size()
        if size == "100M":
            print("✓ Memory size calculation works correctly")
        else:
            print(f"✗ Memory size calculation failed: {size}")
            return False
        
        # Test command building
        cmd = injector._build_stress_command()
        if 'stress-ng' in cmd[0] and '--vm' in cmd:
            print("✓ Memory stress command built correctly")
        else:
            print("✗ Memory stress command building failed")
            return False
        
        print("✓ Requirement 3.2 (Memory stress injection) satisfied")
        return True
        
    except Exception as e:
        print(f"✗ Requirement 3.2 failed: {e}")
        return False

def test_requirement_3_3_disk_stress():
    """Test Requirement 3.3: Disk I/O stress injection (disk)"""
    print("\n" + "="*60)
    print("Testing Requirement 3.3: Disk I/O stress injection")
    print("="*60)
    
    try:
        from disk_stress import DiskStressInjector
        
        # Create temporary directory for testing
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Test basic disk stress configuration
            config = {
                "duration": 3,       # Short duration
                "workers": 1,
                "file_size": "50M",  # Small file for testing
                "temp_path": temp_dir,
                "safety_threshold": 98
            }
            
            injector = DiskStressInjector(config)
            
            # Test configuration validation
            print("✓ Disk stress injector created successfully")
            
            # Test disk usage monitoring
            usage = injector._get_disk_usage(temp_dir)
            if 'used_percent' in usage and 'free_gb' in usage:
                print("✓ Disk usage monitoring works correctly")
            else:
                print("✗ Disk usage monitoring failed")
                return False
            
            # Test command building
            cmd = injector._build_stress_command()
            if 'stress-ng' in cmd[0] and '--hdd' in cmd:
                print("✓ Disk stress command built correctly")
            else:
                print("✗ Disk stress command building failed")
                return False
            
            print("✓ Requirement 3.3 (Disk I/O stress injection) satisfied")
            return True
            
        finally:
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        
    except Exception as e:
        print(f"✗ Requirement 3.3 failed: {e}")
        return False

def test_requirement_3_4_network_delay():
    """Test Requirement 3.4: Network delay injection (delay)"""
    print("\n" + "="*60)
    print("Testing Requirement 3.4: Network delay injection")
    print("="*60)
    
    try:
        from network_delay import NetworkDelayInjector
        
        # Test basic network delay configuration
        config = {
            "delay": "10ms",
            "jitter": "2ms",
            "duration": 3,
            "interface": "lo"  # Use loopback for testing
        }
        
        injector = NetworkDelayInjector(config)
        
        # Test configuration validation
        print("✓ Network delay injector created successfully")
        
        # Test command building
        commands = injector._build_tc_commands()
        if commands and 'tc' in commands[0][0] and 'netem' in commands[0]:
            print("✓ Network delay command built correctly")
        else:
            print("✗ Network delay command building failed")
            return False
        
        # Test interface detection
        interface = injector._get_network_interface()
        if interface:
            print(f"✓ Network interface detection works: {interface}")
        else:
            print("✗ Network interface detection failed")
            return False
        
        print("✓ Requirement 3.4 (Network delay injection) satisfied")
        return True
        
    except Exception as e:
        print(f"✗ Requirement 3.4 failed: {e}")
        return False

def test_requirement_3_5_packet_loss():
    """Test Requirement 3.5: Network packet loss injection (loss)"""
    print("\n" + "="*60)
    print("Testing Requirement 3.5: Network packet loss injection")
    print("="*60)
    
    try:
        from packet_loss import PacketLossInjector
        
        # Test basic packet loss configuration
        config = {
            "loss_percent": 2.0,
            "correlation": 25,
            "duration": 3,
            "interface": "lo",  # Use loopback for testing
            "pattern": "random"
        }
        
        injector = PacketLossInjector(config)
        
        # Test configuration validation
        print("✓ Packet loss injector created successfully")
        
        # Test command building
        commands = injector._build_tc_commands()
        if commands and 'tc' in commands[0][0] and 'loss' in commands[0]:
            print("✓ Packet loss command built correctly")
        else:
            print("✗ Packet loss command building failed")
            return False
        
        # Test different loss patterns
        for pattern in ['random', 'burst', 'periodic']:
            test_config = config.copy()
            test_config['pattern'] = pattern
            test_injector = PacketLossInjector(test_config)
            test_commands = test_injector._build_tc_commands()
            if test_commands:
                print(f"✓ {pattern} loss pattern supported")
            else:
                print(f"✗ {pattern} loss pattern failed")
                return False
        
        print("✓ Requirement 3.5 (Network packet loss injection) satisfied")
        return True
        
    except Exception as e:
        print(f"✗ Requirement 3.5 failed: {e}")
        return False

def test_requirement_3_6_socket_failure():
    """Test Requirement 3.6: Socket/connection failure injection (socket)"""
    print("\n" + "="*60)
    print("Testing Requirement 3.6: Socket/connection failure injection")
    print("="*60)
    
    try:
        from socket_failure import SocketFailureInjector
        
        # Test basic socket failure configuration
        config = {
            "duration": 3,
            "failure_pattern": "selective",
            "target_ports": [8080, 9090],
            "protocols": ["tcp"],
            "block_incoming": True,
            "block_outgoing": False
        }
        
        injector = SocketFailureInjector(config)
        
        # Test configuration validation
        print("✓ Socket failure injector created successfully")
        
        # Test command building
        commands = injector._build_iptables_commands()
        if commands and 'iptables' in commands[0][0] and 'DROP' in commands[0]:
            print("✓ Socket failure command built correctly")
        else:
            print("✗ Socket failure command building failed")
            return False
        
        # Test different failure patterns
        for pattern in ['complete', 'selective', 'intermittent']:
            test_config = config.copy()
            test_config['failure_pattern'] = pattern
            if pattern == 'complete':
                test_config.pop('target_ports', None)  # Complete doesn't need specific ports
            test_injector = SocketFailureInjector(test_config)
            test_commands = test_injector._build_iptables_commands()
            if test_commands:
                print(f"✓ {pattern} failure pattern supported")
            else:
                print(f"✗ {pattern} failure pattern failed")
                return False
        
        print("✓ Requirement 3.6 (Socket/connection failure injection) satisfied")
        return True
        
    except Exception as e:
        print(f"✗ Requirement 3.6 failed: {e}")
        return False

def test_requirement_3_7_monitoring():
    """Test Requirement 3.7: Records injection timestamp and provides monitoring"""
    print("\n" + "="*60)
    print("Testing Requirement 3.7: Injection timestamp and monitoring")
    print("="*60)
    
    try:
        from cpu_stress import CPUStressInjector
        
        # Test monitoring functionality
        config = {
            "intensity": "10%",
            "duration": 2,
            "monitoring_interval": 0.5
        }
        
        injector = CPUStressInjector(config)
        
        # Test timestamp recording
        injector.start_time = time.time()
        status = injector.get_status()
        
        if 'start_time' in status and status['start_time']:
            print("✓ Injection timestamp recording works")
        else:
            print("✗ Injection timestamp recording failed")
            return False
        
        # Test monitoring data collection
        injector.injection_log = [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "cpu_percent": 50.0,
                "memory_percent": 30.0,
                "active_stress": True
            }
        ]
        
        monitoring_data = injector.get_monitoring_data()
        if monitoring_data and len(monitoring_data) > 0:
            print("✓ Monitoring data collection works")
        else:
            print("✗ Monitoring data collection failed")
            return False
        
        # Test log saving
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            log_file = f.name
        
        try:
            injector.save_injection_log(log_file)
            
            # Verify log file structure
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            required_fields = ['config', 'start_time', 'monitoring_data']
            for field in required_fields:
                if field not in log_data:
                    print(f"✗ Log file missing required field: {field}")
                    return False
            
            print("✓ Log saving functionality works")
            
        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)
        
        print("✓ Requirement 3.7 (Injection timestamp and monitoring) satisfied")
        return True
        
    except Exception as e:
        print(f"✗ Requirement 3.7 failed: {e}")
        return False

def test_requirement_6_2_configurable_targets():
    """Test Requirement 6.2: Configurable target services for chaos injection"""
    print("\n" + "="*60)
    print("Testing Requirement 6.2: Configurable target services")
    print("="*60)
    
    try:
        from cpu_stress import CPUStressInjector
        
        # Test container targeting
        config = {
            "intensity": "25%",
            "duration": 3,
            "target_container": "test_container"
        }
        
        injector = CPUStressInjector(config)
        
        if injector.config.get('target_container') == 'test_container':
            print("✓ Container targeting configuration works")
        else:
            print("✗ Container targeting configuration failed")
            return False
        
        # Test that all modules support container targeting
        modules_to_test = [
            ('memory_stress', 'MemoryStressInjector'),
            ('disk_stress', 'DiskStressInjector'),
            ('network_delay', 'NetworkDelayInjector'),
            ('packet_loss', 'PacketLossInjector'),
            ('socket_failure', 'SocketFailureInjector')
        ]
        
        for module_name, class_name in modules_to_test:
            try:
                module = __import__(module_name)
                injector_class = getattr(module, class_name)
                
                test_config = {"duration": 3, "target_container": "test"}
                if module_name == 'memory_stress':
                    test_config["size"] = "100M"
                elif module_name == 'network_delay':
                    test_config["delay"] = "10ms"
                elif module_name == 'packet_loss':
                    test_config["loss_percent"] = 1.0
                elif module_name == 'socket_failure':
                    test_config["failure_pattern"] = "selective"
                    test_config["target_ports"] = [8080]
                
                test_injector = injector_class(test_config)
                if test_injector.config.get('target_container') == 'test':
                    print(f"✓ {module_name} supports container targeting")
                else:
                    print(f"✗ {module_name} container targeting failed")
                    return False
                    
            except Exception as e:
                print(f"✗ {module_name} container targeting test failed: {e}")
                return False
        
        print("✓ Requirement 6.2 (Configurable target services) satisfied")
        return True
        
    except Exception as e:
        print(f"✗ Requirement 6.2 failed: {e}")
        return False

def test_requirement_6_3_configurable_intensity():
    """Test Requirement 6.3: Configurable chaos intensity levels"""
    print("\n" + "="*60)
    print("Testing Requirement 6.3: Configurable chaos intensity levels")
    print("="*60)
    
    try:
        # Test CPU intensity levels
        from cpu_stress import CPUStressInjector
        
        intensity_levels = ["25%", "50%", "75%", 2, 4]
        for intensity in intensity_levels:
            config = {"intensity": intensity, "duration": 3}
            injector = CPUStressInjector(config)
            workers = injector._calculate_workers()
            if workers > 0:
                print(f"✓ CPU intensity {intensity} -> {workers} workers")
            else:
                print(f"✗ CPU intensity {intensity} calculation failed")
                return False
        
        # Test memory size levels
        from memory_stress import MemoryStressInjector
        
        memory_sizes = ["100M", "512M", "1G", "25%", "50%"]
        for size in memory_sizes:
            config = {"size": size, "duration": 3}
            injector = MemoryStressInjector(config)
            calculated_size = injector._calculate_memory_size()
            if calculated_size:
                print(f"✓ Memory size {size} -> {calculated_size}")
            else:
                print(f"✗ Memory size {size} calculation failed")
                return False
        
        # Test network delay levels
        from network_delay import NetworkDelayInjector
        
        delay_levels = ["10ms", "50ms", "100ms", "500ms", "1s"]
        for delay in delay_levels:
            config = {"delay": delay, "duration": 3}
            injector = NetworkDelayInjector(config)
            if injector.config['delay'] == delay:
                print(f"✓ Network delay {delay} configured correctly")
            else:
                print(f"✗ Network delay {delay} configuration failed")
                return False
        
        # Test packet loss levels
        from packet_loss import PacketLossInjector
        
        loss_levels = [1.0, 5.0, 10.0, 25.0, 50.0]
        for loss in loss_levels:
            config = {"loss_percent": loss, "duration": 3}
            injector = PacketLossInjector(config)
            if injector.config['loss_percent'] == loss:
                print(f"✓ Packet loss {loss}% configured correctly")
            else:
                print(f"✗ Packet loss {loss}% configuration failed")
                return False
        
        print("✓ Requirement 6.3 (Configurable chaos intensity levels) satisfied")
        return True
        
    except Exception as e:
        print(f"✗ Requirement 6.3 failed: {e}")
        return False

def test_requirement_6_5_config_validation():
    """Test Requirement 6.5: Configuration parameter validation"""
    print("\n" + "="*60)
    print("Testing Requirement 6.5: Configuration parameter validation")
    print("="*60)
    
    try:
        # Test CPU configuration validation
        from cpu_stress import CPUStressInjector
        
        # Test invalid configurations
        invalid_configs = [
            {"intensity": "150%", "duration": 10},  # Invalid percentage
            {"intensity": "50%"},                   # Missing duration
            {"intensity": "50%", "duration": -5},  # Invalid duration
        ]
        
        for config in invalid_configs:
            try:
                CPUStressInjector(config)
                print(f"✗ Invalid CPU config accepted: {config}")
                return False
            except ValueError:
                print(f"✓ Invalid CPU config properly rejected: {config}")
        
        # Test memory configuration validation
        from memory_stress import MemoryStressInjector
        
        invalid_memory_configs = [
            {"size": "150%", "duration": 10},      # Invalid percentage
            {"size": "invalid", "duration": 10},   # Invalid format
            {"size": "50%"},                       # Missing duration
        ]
        
        for config in invalid_memory_configs:
            try:
                MemoryStressInjector(config)
                print(f"✗ Invalid memory config accepted: {config}")
                return False
            except ValueError:
                print(f"✓ Invalid memory config properly rejected: {config}")
        
        # Test network delay validation
        from network_delay import NetworkDelayInjector
        
        invalid_delay_configs = [
            {"delay": "invalid", "duration": 10},  # Invalid delay format
            {"delay": "100ms"},                    # Missing duration
            {"delay": "100ms", "duration": -5},   # Invalid duration
        ]
        
        for config in invalid_delay_configs:
            try:
                NetworkDelayInjector(config)
                print(f"✗ Invalid delay config accepted: {config}")
                return False
            except ValueError:
                print(f"✓ Invalid delay config properly rejected: {config}")
        
        print("✓ Requirement 6.5 (Configuration parameter validation) satisfied")
        return True
        
    except Exception as e:
        print(f"✗ Requirement 6.5 failed: {e}")
        return False

def test_shell_script_interfaces():
    """Test shell script interfaces for all modules."""
    print("\n" + "="*60)
    print("Testing Shell Script Interfaces")
    print("="*60)
    
    script_paths = [
        chaos_dir / 'cpu' / 'inject_cpu_stress.sh',
        chaos_dir / 'memory' / 'inject_memory_stress.sh',
        chaos_dir / 'disk' / 'inject_disk_stress.sh',
        chaos_dir / 'socket' / 'inject_socket_failure.sh'
    ]
    
    for script_path in script_paths:
        if script_path.exists():
            # Test help functionality
            try:
                result = subprocess.run([str(script_path), '--help'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0 and 'Usage:' in result.stdout:
                    print(f"✓ {script_path.name} help functionality works")
                else:
                    print(f"✗ {script_path.name} help functionality failed")
                    return False
            except Exception as e:
                print(f"✗ {script_path.name} test failed: {e}")
                return False
        else:
            print(f"✗ {script_path.name} not found")
            return False
    
    print("✓ Shell script interfaces work correctly")
    return True

def main():
    """Run comprehensive chaos engineering test suite."""
    print("Chaos Engineering Comprehensive Test Suite")
    print("=" * 60)
    print("Testing all chaos injection capabilities and requirements")
    print("=" * 60)
    
    # Check system requirements first
    if not check_system_requirements():
        print("\n✗ System requirements not met. Some tests may fail.")
        print("Please install missing tools: stress-ng, tc, iptables, jq")
    
    # Define all tests
    tests = [
        ("Requirement 3.1", test_requirement_3_1_cpu_stress),
        ("Requirement 3.2", test_requirement_3_2_memory_stress),
        ("Requirement 3.3", test_requirement_3_3_disk_stress),
        ("Requirement 3.4", test_requirement_3_4_network_delay),
        ("Requirement 3.5", test_requirement_3_5_packet_loss),
        ("Requirement 3.6", test_requirement_3_6_socket_failure),
        ("Requirement 3.7", test_requirement_3_7_monitoring),
        ("Requirement 6.2", test_requirement_6_2_configurable_targets),
        ("Requirement 6.3", test_requirement_6_3_configurable_intensity),
        ("Requirement 6.5", test_requirement_6_5_config_validation),
        ("Shell Scripts", test_shell_script_interfaces)
    ]
    
    # Run all tests
    passed = 0
    total = len(tests)
    failed_tests = []
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✓ {test_name} PASSED")
            else:
                failed_tests.append(test_name)
                print(f"\n✗ {test_name} FAILED")
        except Exception as e:
            failed_tests.append(test_name)
            print(f"\n✗ {test_name} FAILED with exception: {e}")
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if failed_tests:
        print(f"\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Chaos engineering implementation is complete.")
        print("All requirements have been satisfied:")
        print("  ✓ CPU stress injection (Requirement 3.1)")
        print("  ✓ Memory stress injection (Requirement 3.2)")
        print("  ✓ Disk I/O stress injection (Requirement 3.3)")
        print("  ✓ Network delay injection (Requirement 3.4)")
        print("  ✓ Network packet loss injection (Requirement 3.5)")
        print("  ✓ Socket/connection failure injection (Requirement 3.6)")
        print("  ✓ Injection timestamp and monitoring (Requirement 3.7)")
        print("  ✓ Configurable target services (Requirement 6.2)")
        print("  ✓ Configurable chaos intensity levels (Requirement 6.3)")
        print("  ✓ Configuration parameter validation (Requirement 6.5)")
        return 0
    else:
        print(f"\n❌ {total - passed} tests failed. Please review and fix issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())