#!/usr/bin/env python3
"""
Network Delay Injection Module

This module implements network delay injection using traffic control (tc) with configurable
delay patterns, jitter settings, and monitoring for chaos engineering experiments.
"""

import subprocess
import time
import logging
import json
import threading
from typing import Dict, Optional, Any, List
from datetime import datetime, timezone
import psutil
import signal
import os
import re

class NetworkDelayInjector:
    """Network delay injection with monitoring and safety mechanisms."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize network delay injector.
        
        Args:
            config: Configuration dictionary with delay parameters
        """
        self.config = config
        self.active_rules = []
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.start_time = None
        self.injection_log = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        required_fields = ['delay', 'duration']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")
        
        # Validate delay format (e.g., "100ms", "1s")
        delay = self.config['delay']
        if not re.match(r'^\d+(\.\d+)?(ms|s)$', delay):
            raise ValueError("Delay must be in format like '100ms' or '1s'")
        
        # Validate duration
        if not isinstance(self.config['duration'], (int, float)) or self.config['duration'] <= 0:
            raise ValueError("Duration must be a positive number")
        
        # Set defaults
        self.config.setdefault('jitter', '0ms')  # Delay variation
        self.config.setdefault('interface', 'eth0')  # Network interface
        self.config.setdefault('target_ports', [])  # Specific ports to target
        self.config.setdefault('target_ips', [])  # Specific IPs to target
        self.config.setdefault('monitoring_interval', 1)  # seconds
        self.config.setdefault('target_container', None)  # specific container targeting
        self.config.setdefault('direction', 'both')  # ingress, egress, or both
        
        # Validate jitter format
        jitter = self.config['jitter']
        if jitter != '0ms' and not re.match(r'^\d+(\.\d+)?(ms|s)$', jitter):
            raise ValueError("Jitter must be in format like '10ms' or '0.1s'")
        
        # Validate direction
        if self.config['direction'] not in ['ingress', 'egress', 'both']:
            raise ValueError("Direction must be 'ingress', 'egress', or 'both'")
    
    def _get_network_interface(self) -> str:
        """Get the appropriate network interface."""
        interface = self.config['interface']
        
        # Check if interface exists
        try:
            result = subprocess.run(['ip', 'link', 'show', interface], 
                                  capture_output=True, text=True, check=True)
            return interface
        except subprocess.CalledProcessError:
            # Try to find a suitable interface
            try:
                result = subprocess.run(['ip', 'route', 'show', 'default'], 
                                      capture_output=True, text=True, check=True)
                # Extract interface from default route
                match = re.search(r'dev\s+(\w+)', result.stdout)
                if match:
                    return match.group(1)
            except subprocess.CalledProcessError:
                pass
            
            # Fallback to common interface names
            for fallback in ['eth0', 'ens33', 'enp0s3', 'docker0']:
                try:
                    subprocess.run(['ip', 'link', 'show', fallback], 
                                 capture_output=True, text=True, check=True)
                    self.logger.warning(f"Using fallback interface: {fallback}")
                    return fallback
                except subprocess.CalledProcessError:
                    continue
            
            raise ValueError(f"Network interface {interface} not found and no suitable fallback available")
    
    def _build_tc_commands(self) -> List[List[str]]:
        """Build traffic control commands for delay injection."""
        interface = self._get_network_interface()
        delay = self.config['delay']
        jitter = self.config['jitter']
        direction = self.config['direction']
        
        commands = []
        
        # Build netem command
        netem_params = ['delay', delay]
        if jitter != '0ms':
            netem_params.extend([jitter, '25%'])  # 25% correlation for jitter
        
        if direction in ['egress', 'both']:
            # Egress (outgoing) traffic
            cmd = [
                'tc', 'qdisc', 'add', 'dev', interface, 'root', 'handle', '1:', 'netem'
            ] + netem_params
            commands.append(cmd)
        
        if direction in ['ingress', 'both']:
            # Ingress (incoming) traffic - requires ifb (intermediate functional block)
            # This is more complex and may require additional setup
            self.logger.warning("Ingress delay injection requires advanced setup and may not work in all environments")
        
        # Add port/IP specific rules if specified
        target_ports = self.config.get('target_ports', [])
        target_ips = self.config.get('target_ips', [])
        
        if target_ports or target_ips:
            # Use tc filters for specific targeting
            filter_commands = self._build_filter_commands(interface, target_ports, target_ips)
            commands.extend(filter_commands)
        
        return commands
    
    def _build_filter_commands(self, interface: str, target_ports: List[int], target_ips: List[str]) -> List[List[str]]:
        """Build tc filter commands for specific ports/IPs."""
        commands = []
        
        # Add class for delayed traffic
        class_cmd = ['tc', 'class', 'add', 'dev', interface, 'parent', '1:', 'classid', '1:1', 'htb', 'rate', '1000mbit']
        commands.append(class_cmd)
        
        # Add filters for specific ports
        for port in target_ports:
            # Filter for destination port
            filter_cmd = [
                'tc', 'filter', 'add', 'dev', interface, 'protocol', 'ip', 'parent', '1:',
                'prio', '1', 'u32', 'match', 'ip', 'dport', str(port), '0xffff',
                'flowid', '1:1'
            ]
            commands.append(filter_cmd)
        
        # Add filters for specific IPs
        for ip in target_ips:
            # Filter for destination IP
            filter_cmd = [
                'tc', 'filter', 'add', 'dev', interface, 'protocol', 'ip', 'parent', '1:',
                'prio', '1', 'u32', 'match', 'ip', 'dst', ip,
                'flowid', '1:1'
            ]
            commands.append(filter_cmd)
        
        return commands
    
    def _monitor_network_latency(self):
        """Monitor network latency during delay injection."""
        while not self.stop_monitoring.is_set():
            try:
                # Get network statistics
                net_stats = psutil.net_io_counters()
                
                timestamp = datetime.now(timezone.utc).isoformat()
                
                # Log monitoring data
                monitor_data = {
                    'timestamp': timestamp,
                    'bytes_sent': net_stats.bytes_sent,
                    'bytes_recv': net_stats.bytes_recv,
                    'packets_sent': net_stats.packets_sent,
                    'packets_recv': net_stats.packets_recv,
                    'errin': net_stats.errin,
                    'errout': net_stats.errout,
                    'dropin': net_stats.dropin,
                    'dropout': net_stats.dropout,
                    'active_delay': len(self.active_rules) > 0
                }
                
                # Try to measure actual latency with ping (if possible)
                try:
                    ping_result = subprocess.run(
                        ['ping', '-c', '1', '-W', '1', '8.8.8.8'],
                        capture_output=True, text=True, timeout=2
                    )
                    if ping_result.returncode == 0:
                        # Extract latency from ping output
                        match = re.search(r'time=(\d+\.?\d*)', ping_result.stdout)
                        if match:
                            monitor_data['ping_latency_ms'] = float(match.group(1))
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                    # Ping failed or not available
                    pass
                
                self.injection_log.append(monitor_data)
                
                # Log current status
                if len(self.injection_log) % 10 == 0:  # Log every 10 seconds
                    self.logger.info(
                        f"Network: {net_stats.packets_sent} sent, {net_stats.packets_recv} recv, "
                        f"Errors: {net_stats.errin + net_stats.errout}"
                    )
                
            except Exception as e:
                self.logger.error(f"Error during monitoring: {e}")
            
            time.sleep(self.config['monitoring_interval'])
    
    def start_injection(self) -> bool:
        """
        Start network delay injection.
        
        Returns:
            bool: True if injection started successfully, False otherwise
        """
        if self.active_rules:
            self.logger.warning("Network delay injection already running")
            return False
        
        try:
            self.logger.info(
                f"Starting network delay injection: {self.config['delay']} delay "
                f"with {self.config['jitter']} jitter on {self.config['interface']}"
            )
            
            # Check if running in container
            if self.config.get('target_container'):
                return self._start_container_injection()
            else:
                return self._start_host_injection()
            
        except Exception as e:
            self.logger.error(f"Failed to start network delay injection: {e}")
            return False
    
    def _start_host_injection(self) -> bool:
        """Start delay injection on host."""
        try:
            # Build tc commands
            commands = self._build_tc_commands()
            
            # Execute commands
            for cmd in commands:
                self.logger.info(f"Executing: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                self.active_rules.append(cmd)
            
            # Record start time
            self.start_time = datetime.now(timezone.utc)
            
            # Start monitoring thread
            self.stop_monitoring.clear()
            self.monitoring_thread = threading.Thread(target=self._monitor_network_latency)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            self.logger.info("Network delay injection started successfully")
            
            # Schedule automatic cleanup
            cleanup_thread = threading.Thread(target=self._auto_cleanup)
            cleanup_thread.daemon = True
            cleanup_thread.start()
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to execute tc command: {e.stderr}")
            # Clean up any partial rules
            self._cleanup_rules()
            return False
    
    def _start_container_injection(self) -> bool:
        """Start delay injection in container using docker exec."""
        try:
            container = self.config['target_container']
            
            # Build tc commands for container
            commands = self._build_tc_commands()
            
            # Execute commands in container
            for cmd in commands:
                docker_cmd = ['docker', 'exec', container] + cmd
                self.logger.info(f"Executing in container: {' '.join(docker_cmd)}")
                result = subprocess.run(docker_cmd, capture_output=True, text=True, check=True)
                self.active_rules.append(('container', container, cmd))
            
            # Record start time
            self.start_time = datetime.now(timezone.utc)
            
            # Start monitoring thread
            self.stop_monitoring.clear()
            self.monitoring_thread = threading.Thread(target=self._monitor_network_latency)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            self.logger.info(f"Network delay injection started in container: {container}")
            
            # Schedule automatic cleanup
            cleanup_thread = threading.Thread(target=self._auto_cleanup)
            cleanup_thread.daemon = True
            cleanup_thread.start()
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to execute tc command in container: {e.stderr}")
            # Clean up any partial rules
            self._cleanup_rules()
            return False
    
    def _auto_cleanup(self):
        """Automatically clean up after duration expires."""
        time.sleep(self.config['duration'])
        if self.active_rules:
            self.logger.info("Duration expired, automatically stopping delay injection")
            self.stop_injection()
    
    def stop_injection(self) -> bool:
        """
        Stop network delay injection.
        
        Returns:
            bool: True if injection stopped successfully, False otherwise
        """
        if not self.active_rules:
            self.logger.warning("No network delay injection running")
            return False
        
        try:
            # Stop monitoring
            self.stop_monitoring.set()
            
            # Clean up tc rules
            self._cleanup_rules()
            
            # Wait for monitoring thread to finish
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            self.logger.info("Network delay injection stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping network delay injection: {e}")
            return False
    
    def _cleanup_rules(self):
        """Clean up all active tc rules."""
        interface = self._get_network_interface()
        
        for rule in self.active_rules:
            try:
                if isinstance(rule, tuple) and rule[0] == 'container':
                    # Container rule
                    _, container, _ = rule
                    cleanup_cmd = ['docker', 'exec', container, 'tc', 'qdisc', 'del', 'dev', interface, 'root']
                    subprocess.run(cleanup_cmd, capture_output=True, text=True)
                else:
                    # Host rule - remove root qdisc (removes all rules)
                    cleanup_cmd = ['tc', 'qdisc', 'del', 'dev', interface, 'root']
                    subprocess.run(cleanup_cmd, capture_output=True, text=True)
                    break  # Only need to run once for host rules
            except Exception as e:
                self.logger.warning(f"Error cleaning up rule: {e}")
        
        self.active_rules.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current injection status.
        
        Returns:
            Dict containing status information
        """
        is_running = len(self.active_rules) > 0
        
        status = {
            'running': is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'duration': self.config['duration'],
            'delay': self.config['delay'],
            'jitter': self.config['jitter'],
            'interface': self.config['interface'],
            'direction': self.config['direction'],
            'target_container': self.config.get('target_container'),
            'active_rules': len(self.active_rules),
            'monitoring_data_points': len(self.injection_log)
        }
        
        if is_running and self.start_time:
            elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            status['elapsed_seconds'] = elapsed
            status['remaining_seconds'] = max(0, self.config['duration'] - elapsed)
        
        # Add current network stats
        try:
            net_stats = psutil.net_io_counters()
            status.update({
                'current_packets_sent': net_stats.packets_sent,
                'current_packets_recv': net_stats.packets_recv,
                'current_errors': net_stats.errin + net_stats.errout
            })
        except Exception:
            pass
        
        return status
    
    def get_monitoring_data(self) -> list:
        """
        Get collected monitoring data.
        
        Returns:
            List of monitoring data points
        """
        return self.injection_log.copy()
    
    def save_injection_log(self, filepath: str):
        """
        Save injection log to file.
        
        Args:
            filepath: Path to save the log file
        """
        log_data = {
            'config': self.config,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'monitoring_data': self.injection_log
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        self.logger.info(f"Injection log saved to {filepath}")


def create_delay_injector(config_file: str) -> NetworkDelayInjector:
    """
    Create network delay injector from configuration file.
    
    Args:
        config_file: Path to JSON configuration file
        
    Returns:
        NetworkDelayInjector instance
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return NetworkDelayInjector(config)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Network Delay Injection Tool")
    parser.add_argument('--config', required=True, help='Configuration file path')
    parser.add_argument('--action', choices=['start', 'stop', 'status'], 
                       default='start', help='Action to perform')
    parser.add_argument('--log-file', help='Path to save injection log')
    
    args = parser.parse_args()
    
    try:
        injector = create_delay_injector(args.config)
        
        if args.action == 'start':
            if injector.start_injection():
                print("Network delay injection started successfully")
                # Wait for completion
                while injector.get_status()['running']:
                    time.sleep(1)
                print("Network delay injection completed")
            else:
                print("Failed to start network delay injection")
                exit(1)
        
        elif args.action == 'stop':
            if injector.stop_injection():
                print("Network delay injection stopped successfully")
            else:
                print("Failed to stop network delay injection")
                exit(1)
        
        elif args.action == 'status':
            status = injector.get_status()
            print(json.dumps(status, indent=2))
        
        # Save log if requested
        if args.log_file:
            injector.save_injection_log(args.log_file)
            
    except Exception as e:
        print(f"Error: {e}")
        exit(1)