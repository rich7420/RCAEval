#!/usr/bin/env python3
"""
Socket/Connection Failure Injection Module

This module implements socket and connection failure injection using iptables with configurable
connection blocking patterns and monitoring for chaos engineering experiments.
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

class SocketFailureInjector:
    """Socket/connection failure injection with monitoring and safety mechanisms."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize socket failure injector.
        
        Args:
            config: Configuration dictionary with failure parameters
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
        required_fields = ['duration']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")
        
        # Validate duration
        if not isinstance(self.config['duration'], (int, float)) or self.config['duration'] <= 0:
            raise ValueError("Duration must be a positive number")
        
        # Set defaults
        self.config.setdefault('target_ports', [])  # Specific ports to block
        self.config.setdefault('target_ips', [])  # Specific IPs to block
        self.config.setdefault('block_incoming', True)  # Block incoming connections
        self.config.setdefault('block_outgoing', True)  # Block outgoing connections
        self.config.setdefault('monitoring_interval', 1)  # seconds
        self.config.setdefault('target_container', None)  # specific container targeting
        self.config.setdefault('failure_pattern', 'complete')  # complete, intermittent, or selective
        self.config.setdefault('protocols', ['tcp'])  # tcp, udp, or both
        
        # Validate failure pattern
        if self.config['failure_pattern'] not in ['complete', 'intermittent', 'selective']:
            raise ValueError("Failure pattern must be 'complete', 'intermittent', or 'selective'")
        
        # Validate protocols
        valid_protocols = ['tcp', 'udp', 'icmp']
        protocols = self.config['protocols']
        if not isinstance(protocols, list):
            protocols = [protocols]
        for protocol in protocols:
            if protocol not in valid_protocols:
                raise ValueError(f"Protocol must be one of: {valid_protocols}")
        
        # Ensure we have something to block
        if not self.config['target_ports'] and not self.config['target_ips']:
            if self.config['failure_pattern'] != 'complete':
                raise ValueError("Must specify target_ports or target_ips for selective blocking")
    
    def _build_iptables_commands(self) -> List[List[str]]:
        """Build iptables commands for connection blocking."""
        commands = []
        target_ports = self.config.get('target_ports', [])
        target_ips = self.config.get('target_ips', [])
        block_incoming = self.config.get('block_incoming', True)
        block_outgoing = self.config.get('block_outgoing', True)
        protocols = self.config.get('protocols', ['tcp'])
        failure_pattern = self.config.get('failure_pattern', 'complete')
        
        # Complete failure - block all traffic
        if failure_pattern == 'complete':
            if block_incoming:
                for protocol in protocols:
                    cmd = ['iptables', '-A', 'INPUT', '-p', protocol, '-j', 'DROP']
                    commands.append(cmd)
            
            if block_outgoing:
                for protocol in protocols:
                    cmd = ['iptables', '-A', 'OUTPUT', '-p', protocol, '-j', 'DROP']
                    commands.append(cmd)
        
        # Selective failure - block specific ports/IPs
        else:
            # Block specific ports
            for port in target_ports:
                for protocol in protocols:
                    if block_incoming:
                        cmd = ['iptables', '-A', 'INPUT', '-p', protocol, '--dport', str(port), '-j', 'DROP']
                        commands.append(cmd)
                    
                    if block_outgoing:
                        cmd = ['iptables', '-A', 'OUTPUT', '-p', protocol, '--dport', str(port), '-j', 'DROP']
                        commands.append(cmd)
            
            # Block specific IPs
            for ip in target_ips:
                if block_incoming:
                    cmd = ['iptables', '-A', 'INPUT', '-s', ip, '-j', 'DROP']
                    commands.append(cmd)
                
                if block_outgoing:
                    cmd = ['iptables', '-A', 'OUTPUT', '-d', ip, '-j', 'DROP']
                    commands.append(cmd)
        
        return commands
    
    def _monitor_connections(self):
        """Monitor network connections during failure injection."""
        while not self.stop_monitoring.is_set():
            try:
                # Get network connections
                connections = psutil.net_connections()
                
                # Count connections by status
                connection_stats = {
                    'established': 0,
                    'listen': 0,
                    'time_wait': 0,
                    'close_wait': 0,
                    'syn_sent': 0,
                    'syn_recv': 0,
                    'fin_wait1': 0,
                    'fin_wait2': 0,
                    'closing': 0,
                    'last_ack': 0,
                    'total': len(connections)
                }
                
                for conn in connections:
                    status = conn.status.lower() if conn.status else 'unknown'
                    if status in connection_stats:
                        connection_stats[status] += 1
                
                # Get network I/O statistics
                net_stats = psutil.net_io_counters()
                
                timestamp = datetime.now(timezone.utc).isoformat()
                
                # Log monitoring data
                monitor_data = {
                    'timestamp': timestamp,
                    'connection_stats': connection_stats,
                    'bytes_sent': net_stats.bytes_sent,
                    'bytes_recv': net_stats.bytes_recv,
                    'packets_sent': net_stats.packets_sent,
                    'packets_recv': net_stats.packets_recv,
                    'errin': net_stats.errin,
                    'errout': net_stats.errout,
                    'dropin': net_stats.dropin,
                    'dropout': net_stats.dropout,
                    'active_blocking': len(self.active_rules) > 0
                }
                
                # Try to get iptables statistics
                try:
                    result = subprocess.run(['iptables', '-L', '-n', '-v'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        # Parse iptables output for DROP rules
                        drop_count = 0
                        for line in result.stdout.split('\n'):
                            if 'DROP' in line and 'pkts' not in line:  # Skip header
                                match = re.search(r'^\s*(\d+)', line)
                                if match:
                                    drop_count += int(match.group(1))
                        monitor_data['iptables_drops'] = drop_count
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                    pass
                
                self.injection_log.append(monitor_data)
                
                # Log current status
                if len(self.injection_log) % 10 == 0:  # Log every 10 seconds
                    self.logger.info(
                        f"Connections: {connection_stats['total']} total, "
                        f"{connection_stats['established']} established, "
                        f"Drops: {net_stats.dropin + net_stats.dropout}"
                    )
                
            except Exception as e:
                self.logger.error(f"Error during monitoring: {e}")
            
            time.sleep(self.config['monitoring_interval'])
    
    def start_injection(self) -> bool:
        """
        Start socket/connection failure injection.
        
        Returns:
            bool: True if injection started successfully, False otherwise
        """
        if self.active_rules:
            self.logger.warning("Socket failure injection already running")
            return False
        
        try:
            failure_pattern = self.config.get('failure_pattern', 'complete')
            target_ports = self.config.get('target_ports', [])
            target_ips = self.config.get('target_ips', [])
            
            self.logger.info(
                f"Starting socket failure injection: {failure_pattern} pattern"
            )
            
            if target_ports:
                self.logger.info(f"Targeting ports: {target_ports}")
            if target_ips:
                self.logger.info(f"Targeting IPs: {target_ips}")
            
            # Check if running in container
            if self.config.get('target_container'):
                return self._start_container_injection()
            else:
                return self._start_host_injection()
            
        except Exception as e:
            self.logger.error(f"Failed to start socket failure injection: {e}")
            return False
    
    def _start_host_injection(self) -> bool:
        """Start socket failure injection on host."""
        try:
            # Check if we have iptables permissions
            try:
                subprocess.run(['iptables', '-L'], capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError:
                self.logger.error("No permission to run iptables. Run as root or with sudo.")
                return False
            
            # Build iptables commands
            commands = self._build_iptables_commands()
            
            if not commands:
                self.logger.error("No iptables commands generated")
                return False
            
            # Execute commands
            for cmd in commands:
                self.logger.info(f"Executing: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                self.active_rules.append(cmd)
            
            # Record start time
            self.start_time = datetime.now(timezone.utc)
            
            # Start monitoring thread
            self.stop_monitoring.clear()
            self.monitoring_thread = threading.Thread(target=self._monitor_connections)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            self.logger.info("Socket failure injection started successfully")
            
            # Schedule automatic cleanup
            cleanup_thread = threading.Thread(target=self._auto_cleanup)
            cleanup_thread.daemon = True
            cleanup_thread.start()
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to execute iptables command: {e.stderr}")
            # Clean up any partial rules
            self._cleanup_rules()
            return False
    
    def _start_container_injection(self) -> bool:
        """Start socket failure injection in container using docker exec."""
        try:
            container = self.config['target_container']
            
            # Build iptables commands for container
            commands = self._build_iptables_commands()
            
            if not commands:
                self.logger.error("No iptables commands generated")
                return False
            
            # Execute commands in container
            for cmd in commands:
                docker_cmd = ['docker', 'exec', '--privileged', container] + cmd
                self.logger.info(f"Executing in container: {' '.join(docker_cmd)}")
                result = subprocess.run(docker_cmd, capture_output=True, text=True, check=True)
                self.active_rules.append(('container', container, cmd))
            
            # Record start time
            self.start_time = datetime.now(timezone.utc)
            
            # Start monitoring thread
            self.stop_monitoring.clear()
            self.monitoring_thread = threading.Thread(target=self._monitor_connections)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            self.logger.info(f"Socket failure injection started in container: {container}")
            
            # Schedule automatic cleanup
            cleanup_thread = threading.Thread(target=self._auto_cleanup)
            cleanup_thread.daemon = True
            cleanup_thread.start()
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to execute iptables command in container: {e.stderr}")
            # Clean up any partial rules
            self._cleanup_rules()
            return False
    
    def _auto_cleanup(self):
        """Automatically clean up after duration expires."""
        time.sleep(self.config['duration'])
        if self.active_rules:
            self.logger.info("Duration expired, automatically stopping socket failure injection")
            self.stop_injection()
    
    def stop_injection(self) -> bool:
        """
        Stop socket/connection failure injection.
        
        Returns:
            bool: True if injection stopped successfully, False otherwise
        """
        if not self.active_rules:
            self.logger.warning("No socket failure injection running")
            return False
        
        try:
            # Stop monitoring
            self.stop_monitoring.set()
            
            # Clean up iptables rules
            self._cleanup_rules()
            
            # Wait for monitoring thread to finish
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            self.logger.info("Socket failure injection stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping socket failure injection: {e}")
            return False
    
    def _cleanup_rules(self):
        """Clean up all active iptables rules."""
        for rule in self.active_rules:
            try:
                if isinstance(rule, tuple) and rule[0] == 'container':
                    # Container rule
                    _, container, original_cmd = rule
                    # Convert ADD to DELETE
                    cleanup_cmd = original_cmd.copy()
                    cleanup_cmd[1] = '-D'  # Change -A to -D
                    docker_cmd = ['docker', 'exec', '--privileged', container] + cleanup_cmd
                    subprocess.run(docker_cmd, capture_output=True, text=True)
                else:
                    # Host rule - convert ADD to DELETE
                    cleanup_cmd = rule.copy()
                    cleanup_cmd[1] = '-D'  # Change -A to -D
                    subprocess.run(cleanup_cmd, capture_output=True, text=True)
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
            'failure_pattern': self.config.get('failure_pattern', 'complete'),
            'target_ports': self.config.get('target_ports', []),
            'target_ips': self.config.get('target_ips', []),
            'protocols': self.config.get('protocols', ['tcp']),
            'block_incoming': self.config.get('block_incoming', True),
            'block_outgoing': self.config.get('block_outgoing', True),
            'target_container': self.config.get('target_container'),
            'active_rules': len(self.active_rules),
            'monitoring_data_points': len(self.injection_log)
        }
        
        if is_running and self.start_time:
            elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            status['elapsed_seconds'] = elapsed
            status['remaining_seconds'] = max(0, self.config['duration'] - elapsed)
        
        # Add current connection stats
        try:
            connections = psutil.net_connections()
            status['current_connections'] = len(connections)
            
            # Count by status
            established = sum(1 for conn in connections if conn.status == 'ESTABLISHED')
            status['current_established'] = established
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


def create_socket_injector(config_file: str) -> SocketFailureInjector:
    """
    Create socket failure injector from configuration file.
    
    Args:
        config_file: Path to JSON configuration file
        
    Returns:
        SocketFailureInjector instance
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return SocketFailureInjector(config)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Socket/Connection Failure Injection Tool")
    parser.add_argument('--config', required=True, help='Configuration file path')
    parser.add_argument('--action', choices=['start', 'stop', 'status'], 
                       default='start', help='Action to perform')
    parser.add_argument('--log-file', help='Path to save injection log')
    
    args = parser.parse_args()
    
    try:
        injector = create_socket_injector(args.config)
        
        if args.action == 'start':
            if injector.start_injection():
                print("Socket failure injection started successfully")
                # Wait for completion
                while injector.get_status()['running']:
                    time.sleep(1)
                print("Socket failure injection completed")
            else:
                print("Failed to start socket failure injection")
                exit(1)
        
        elif args.action == 'stop':
            if injector.stop_injection():
                print("Socket failure injection stopped successfully")
            else:
                print("Failed to stop socket failure injection")
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