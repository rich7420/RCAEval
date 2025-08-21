#!/usr/bin/env python3
"""
Packet Loss Injection Module

This module implements packet loss injection using traffic control (tc) with configurable
loss percentages, patterns, and monitoring for chaos engineering experiments.
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

class PacketLossInjector:
    """Packet loss injection with monitoring and safety mechanisms."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize packet loss injector.
        
        Args:
            config: Configuration dictionary with loss parameters
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
        required_fields = ['loss_percent', 'duration']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")
        
        # Validate loss percentage
        loss_percent = self.config['loss_percent']
        if not isinstance(loss_percent, (int, float)) or not 0 <= loss_percent <= 100:
            raise ValueError("Loss percentage must be between 0 and 100")
        
        # Validate duration
        if not isinstance(self.config['duration'], (int, float)) or self.config['duration'] <= 0:
            raise ValueError("Duration must be a positive number")
        
        # Set defaults
        self.config.setdefault('correlation', 25)  # Correlation percentage for burst losses
        self.config.setdefault('interface', 'eth0')  # Network interface
        self.config.setdefault('target_ports', [])  # Specific ports to target
        self.config.setdefault('target_ips', [])  # Specific IPs to target
        self.config.setdefault('monitoring_interval', 1)  # seconds
        self.config.setdefault('target_container', None)  # specific container targeting
        self.config.setdefault('direction', 'both')  # ingress, egress, or both
        self.config.setdefault('pattern', 'random')  # random, burst, or periodic
        
        # Validate correlation
        correlation = self.config['correlation']
        if not isinstance(correlation, (int, float)) or not 0 <= correlation <= 100:
            raise ValueError("Correlation must be between 0 and 100")
        
        # Validate direction
        if self.config['direction'] not in ['ingress', 'egress', 'both']:
            raise ValueError("Direction must be 'ingress', 'egress', or 'both'")
        
        # Validate pattern
        if self.config['pattern'] not in ['random', 'burst', 'periodic']:
            raise ValueError("Pattern must be 'random', 'burst', or 'periodic'")
    
    def _get_network_interface(self) -> str:
        """Get the appropriate network interface for container or host environment."""
        interface = self.config['interface']
        
        # If targeting a container, we'll use the container's network interface
        if self.config.get('target_container'):
            # For container targeting, we typically use eth0 (standard container interface)
            return 'eth0'
        
        # For host-level chaos, try to detect the appropriate interface
        # First try the configured interface
        if self._interface_exists(interface):
            return interface
        
        # Try to find Docker bridge interface (common for container networking)
        docker_interfaces = ['docker0', 'br-docker0']
        for docker_if in docker_interfaces:
            if self._interface_exists(docker_if):
                self.logger.info(f"Using Docker bridge interface: {docker_if}")
                return docker_if
        
        # Fallback to common interfaces (prioritize container-friendly ones)
        fallback_interfaces = ['eth0', 'ens33', 'enp0s3', 'lo']
        for fallback in fallback_interfaces:
            if self._interface_exists(fallback):
                self.logger.warning(f"Using fallback interface: {fallback}")
                return fallback
        
        # If nothing works, return the original interface (let tc handle the error)
        self.logger.warning(f"Could not verify interface {interface}, proceeding anyway")
        return interface
    
    def _interface_exists(self, interface: str) -> bool:
        """Check if a network interface exists (cross-platform)."""
        try:
            # Try using ip command (Linux)
            result = subprocess.run(['ip', 'link', 'show', interface], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        try:
            # Try using ifconfig (Unix-like systems)
            result = subprocess.run(['ifconfig', interface], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Try checking /sys/class/net (Linux)
        try:
            import os
            return os.path.exists(f'/sys/class/net/{interface}')
        except:
            pass
        
        return False
    
    def _build_tc_commands(self) -> List[List[str]]:
        """Build traffic control commands for packet loss injection."""
        interface = self._get_network_interface()
        loss_percent = self.config['loss_percent']
        correlation = self.config['correlation']
        direction = self.config['direction']
        pattern = self.config['pattern']
        
        commands = []
        
        # Build netem command based on pattern
        if pattern == 'random':
            netem_params = ['loss', f'{loss_percent}%']
        elif pattern == 'burst':
            netem_params = ['loss', f'{loss_percent}%', f'{correlation}%']
        elif pattern == 'periodic':
            # Periodic loss: lose every Nth packet
            if loss_percent > 0:
                period = max(1, int(100 / loss_percent))
                netem_params = ['loss', 'gemodel', f'{loss_percent}%', '0%', '0%', '0%']
            else:
                netem_params = ['loss', '0%']
        else:
            netem_params = ['loss', f'{loss_percent}%']
        
        if direction in ['egress', 'both']:
            # Egress (outgoing) traffic
            cmd = [
                'tc', 'qdisc', 'add', 'dev', interface, 'root', 'handle', '1:', 'netem'
            ] + netem_params
            commands.append(cmd)
        
        if direction in ['ingress', 'both']:
            # Ingress (incoming) traffic - requires ifb (intermediate functional block)
            self.logger.warning("Ingress packet loss injection requires advanced setup and may not work in all environments")
        
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
        
        # Add class for packet loss
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
    
    def _monitor_packet_loss(self):
        """Monitor packet loss during injection."""
        baseline_stats = None
        
        while not self.stop_monitoring.is_set():
            try:
                # Get network statistics
                net_stats = psutil.net_io_counters()
                
                timestamp = datetime.now(timezone.utc).isoformat()
                
                # Calculate packet loss if we have baseline
                packet_loss_rate = 0
                if baseline_stats:
                    sent_diff = net_stats.packets_sent - baseline_stats.packets_sent
                    recv_diff = net_stats.packets_recv - baseline_stats.packets_recv
                    drop_diff = (net_stats.dropin + net_stats.dropout) - (baseline_stats.dropin + baseline_stats.dropout)
                    
                    if sent_diff > 0:
                        packet_loss_rate = (drop_diff / sent_diff) * 100
                else:
                    baseline_stats = net_stats
                
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
                    'calculated_loss_rate': packet_loss_rate,
                    'active_loss': len(self.active_rules) > 0
                }
                
                # Try to get more detailed network statistics
                try:
                    # Get interface-specific statistics
                    interface = self._get_network_interface()
                    with open(f'/sys/class/net/{interface}/statistics/tx_packets', 'r') as f:
                        tx_packets = int(f.read().strip())
                    with open(f'/sys/class/net/{interface}/statistics/rx_packets', 'r') as f:
                        rx_packets = int(f.read().strip())
                    with open(f'/sys/class/net/{interface}/statistics/tx_dropped', 'r') as f:
                        tx_dropped = int(f.read().strip())
                    with open(f'/sys/class/net/{interface}/statistics/rx_dropped', 'r') as f:
                        rx_dropped = int(f.read().strip())
                    
                    monitor_data.update({
                        'interface_tx_packets': tx_packets,
                        'interface_rx_packets': rx_packets,
                        'interface_tx_dropped': tx_dropped,
                        'interface_rx_dropped': rx_dropped
                    })
                except (FileNotFoundError, ValueError):
                    # Interface statistics not available
                    pass
                
                self.injection_log.append(monitor_data)
                
                # Log current status
                if len(self.injection_log) % 10 == 0:  # Log every 10 seconds
                    self.logger.info(
                        f"Network: {net_stats.packets_sent} sent, {net_stats.packets_recv} recv, "
                        f"Dropped: {net_stats.dropin + net_stats.dropout}, "
                        f"Loss rate: {packet_loss_rate:.2f}%"
                    )
                
            except Exception as e:
                self.logger.error(f"Error during monitoring: {e}")
            
            time.sleep(self.config['monitoring_interval'])
    
    def start_injection(self) -> bool:
        """
        Start packet loss injection.
        
        Returns:
            bool: True if injection started successfully, False otherwise
        """
        if self.active_rules:
            self.logger.warning("Packet loss injection already running")
            return False
        
        try:
            self.logger.info(
                f"Starting packet loss injection: {self.config['loss_percent']}% loss "
                f"with {self.config['correlation']}% correlation on {self.config['interface']}"
            )
            
            # Check if running in container
            if self.config.get('target_container'):
                return self._start_container_injection()
            else:
                return self._start_host_injection()
            
        except Exception as e:
            self.logger.error(f"Failed to start packet loss injection: {e}")
            return False
    
    def _start_host_injection(self) -> bool:
        """Start packet loss injection on host."""
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
            self.monitoring_thread = threading.Thread(target=self._monitor_packet_loss)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            self.logger.info("Packet loss injection started successfully")
            
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
        """Start packet loss injection in container using docker exec."""
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
            self.monitoring_thread = threading.Thread(target=self._monitor_packet_loss)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            self.logger.info(f"Packet loss injection started in container: {container}")
            
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
            self.logger.info("Duration expired, automatically stopping packet loss injection")
            self.stop_injection()
    
    def stop_injection(self) -> bool:
        """
        Stop packet loss injection.
        
        Returns:
            bool: True if injection stopped successfully, False otherwise
        """
        if not self.active_rules:
            self.logger.warning("No packet loss injection running")
            return False
        
        try:
            # Stop monitoring
            self.stop_monitoring.set()
            
            # Clean up tc rules
            self._cleanup_rules()
            
            # Wait for monitoring thread to finish
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            self.logger.info("Packet loss injection stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping packet loss injection: {e}")
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
            'loss_percent': self.config['loss_percent'],
            'correlation': self.config['correlation'],
            'pattern': self.config['pattern'],
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
                'current_dropped': net_stats.dropin + net_stats.dropout
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


def create_loss_injector(config_file: str) -> PacketLossInjector:
    """
    Create packet loss injector from configuration file.
    
    Args:
        config_file: Path to JSON configuration file
        
    Returns:
        PacketLossInjector instance
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return PacketLossInjector(config)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Packet Loss Injection Tool")
    parser.add_argument('--config', required=True, help='Configuration file path')
    parser.add_argument('--action', choices=['start', 'stop', 'status'], 
                       default='start', help='Action to perform')
    parser.add_argument('--log-file', help='Path to save injection log')
    
    args = parser.parse_args()
    
    try:
        injector = create_loss_injector(args.config)
        
        if args.action == 'start':
            if injector.start_injection():
                print("Packet loss injection started successfully")
                # Wait for completion
                while injector.get_status()['running']:
                    time.sleep(1)
                print("Packet loss injection completed")
            else:
                print("Failed to start packet loss injection")
                exit(1)
        
        elif args.action == 'stop':
            if injector.stop_injection():
                print("Packet loss injection stopped successfully")
            else:
                print("Failed to stop packet loss injection")
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