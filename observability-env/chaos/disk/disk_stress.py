#!/usr/bin/env python3
"""
Disk I/O Stress Injection Module

This module implements disk I/O stress injection using stress-ng with configurable
I/O patterns, bandwidth limits, and safety mechanisms for chaos engineering experiments.
"""

import subprocess
import time
import logging
import json
import threading
import shutil
from typing import Dict, Optional, Any
from datetime import datetime, timezone
import psutil
import signal
import os

class DiskStressInjector:
    """Disk I/O stress injection with monitoring and safety mechanisms."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize disk stress injector.
        
        Args:
            config: Configuration dictionary with stress parameters
        """
        self.config = config
        self.process = None
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
        self.config.setdefault('workers', 1)  # Number of I/O workers
        self.config.setdefault('file_size', '1G')  # Size of files to create
        self.config.setdefault('temp_path', '/tmp')  # Temporary directory for I/O operations
        self.config.setdefault('safety_threshold', 90)  # Disk usage threshold for safety stop
        self.config.setdefault('monitoring_interval', 1)  # seconds
        self.config.setdefault('method', 'sync')  # I/O method (sync, dsync, etc.)
        self.config.setdefault('target_container', None)  # specific container targeting
        self.config.setdefault('iops_limit', None)  # IOPS limit
        self.config.setdefault('bandwidth_limit', None)  # Bandwidth limit
        
        # Validate temp path exists and is writable
        temp_path = self.config['temp_path']
        if not os.path.exists(temp_path):
            raise ValueError(f"Temporary path does not exist: {temp_path}")
        if not os.access(temp_path, os.W_OK):
            raise ValueError(f"Temporary path is not writable: {temp_path}")
    
    def _get_disk_usage(self, path: str) -> Dict[str, float]:
        """Get disk usage statistics for a path."""
        try:
            usage = shutil.disk_usage(path)
            total_gb = usage.total / (1024**3)
            used_gb = usage.used / (1024**3)
            free_gb = usage.free / (1024**3)
            used_percent = (usage.used / usage.total) * 100
            
            return {
                'total_gb': total_gb,
                'used_gb': used_gb,
                'free_gb': free_gb,
                'used_percent': used_percent
            }
        except Exception as e:
            self.logger.error(f"Error getting disk usage for {path}: {e}")
            return {'total_gb': 0, 'used_gb': 0, 'free_gb': 0, 'used_percent': 0}
    
    def _build_stress_command(self) -> list:
        """Build stress-ng command with parameters."""
        duration = int(self.config['duration'])
        workers = self.config['workers']
        file_size = self.config['file_size']
        temp_path = self.config['temp_path']
        
        cmd = [
            'stress-ng',
            '--hdd', str(workers),
            '--hdd-bytes', file_size,
            '--temp-path', temp_path,
            '--timeout', f'{duration}s',
            '--metrics-brief',
            '--verify'
        ]
        
        # Add I/O method if specified
        if 'method' in self.config and self.config['method']:
            if self.config['method'] == 'sync':
                cmd.append('--hdd-opts')
                cmd.append('sync')
            elif self.config['method'] == 'dsync':
                cmd.append('--hdd-opts')
                cmd.append('dsync')
            elif self.config['method'] == 'direct':
                cmd.append('--hdd-opts')
                cmd.append('direct')
        
        # Add IOPS limit if specified
        if self.config.get('iops_limit'):
            cmd.extend(['--hdd-ops', str(self.config['iops_limit'])])
        
        return cmd
    
    def _monitor_disk_usage(self):
        """Monitor disk usage during stress injection."""
        temp_path = self.config['temp_path']
        
        while not self.stop_monitoring.is_set():
            try:
                # Get disk usage for temp path
                disk_usage = self._get_disk_usage(temp_path)
                
                # Get I/O statistics
                io_stats = psutil.disk_io_counters()
                
                timestamp = datetime.now(timezone.utc).isoformat()
                
                # Log monitoring data
                monitor_data = {
                    'timestamp': timestamp,
                    'disk_used_percent': disk_usage['used_percent'],
                    'disk_free_gb': disk_usage['free_gb'],
                    'disk_used_gb': disk_usage['used_gb'],
                    'disk_total_gb': disk_usage['total_gb'],
                    'active_stress': self.process is not None and self.process.poll() is None
                }
                
                # Add I/O stats if available
                if io_stats:
                    monitor_data.update({
                        'read_bytes': io_stats.read_bytes,
                        'write_bytes': io_stats.write_bytes,
                        'read_count': io_stats.read_count,
                        'write_count': io_stats.write_count,
                        'read_time': io_stats.read_time,
                        'write_time': io_stats.write_time
                    })
                
                self.injection_log.append(monitor_data)
                
                # Safety check - stop if disk usage is dangerously high
                safety_threshold = self.config['safety_threshold']
                if disk_usage['used_percent'] > safety_threshold:
                    self.logger.warning(
                        f"Disk usage {disk_usage['used_percent']:.1f}% exceeds safety threshold {safety_threshold}%. "
                        "Stopping stress injection."
                    )
                    self.stop_injection()
                    break
                
                # Safety check - ensure minimum free space (1GB)
                if disk_usage['free_gb'] < 1.0:
                    self.logger.warning(
                        f"Free disk space {disk_usage['free_gb']:.1f}GB is critically low. "
                        "Stopping stress injection."
                    )
                    self.stop_injection()
                    break
                
                # Log current status
                if len(self.injection_log) % 10 == 0:  # Log every 10 seconds
                    self.logger.info(
                        f"Disk: {disk_usage['used_percent']:.1f}% used, "
                        f"Free: {disk_usage['free_gb']:.1f}GB"
                    )
                
            except Exception as e:
                self.logger.error(f"Error during monitoring: {e}")
            
            time.sleep(self.config['monitoring_interval'])
    
    def start_injection(self) -> bool:
        """
        Start disk I/O stress injection.
        
        Returns:
            bool: True if injection started successfully, False otherwise
        """
        if self.process is not None:
            self.logger.warning("Disk stress injection already running")
            return False
        
        try:
            # Check available disk space before starting
            temp_path = self.config['temp_path']
            disk_usage = self._get_disk_usage(temp_path)
            
            self.logger.info(
                f"Starting disk I/O stress injection: {self.config['file_size']} files "
                f"with {self.config['workers']} workers in {temp_path}"
            )
            self.logger.info(
                f"Current disk usage: {disk_usage['used_percent']:.1f}%, "
                f"Free: {disk_usage['free_gb']:.1f}GB"
            )
            
            # Check if there's enough free space
            file_size_str = self.config['file_size']
            if file_size_str.endswith('G'):
                file_size_gb = float(file_size_str[:-1])
            elif file_size_str.endswith('M'):
                file_size_gb = float(file_size_str[:-1]) / 1024
            else:
                file_size_gb = 1.0  # Default assumption
            
            total_size_needed = file_size_gb * self.config['workers']
            
            if disk_usage['free_gb'] < total_size_needed + 1:  # +1GB buffer
                self.logger.error(
                    f"Insufficient disk space. Need {total_size_needed:.1f}GB, "
                    f"but only {disk_usage['free_gb']:.1f}GB available"
                )
                return False
            
            # Build stress command
            cmd = self._build_stress_command()
            self.logger.info(f"Command: {' '.join(cmd)}")
            
            # Record start time
            self.start_time = datetime.now(timezone.utc)
            
            # Start stress process
            if self.config.get('target_container'):
                # Run stress in specific container
                docker_cmd = [
                    'docker', 'exec', self.config['target_container']
                ] + cmd
                self.process = subprocess.Popen(
                    docker_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            else:
                # Run stress on host
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            
            # Start monitoring thread
            self.stop_monitoring.clear()
            self.monitoring_thread = threading.Thread(target=self._monitor_disk_usage)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            self.logger.info(f"Disk stress injection started with PID {self.process.pid}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start disk stress injection: {e}")
            return False
    
    def stop_injection(self) -> bool:
        """
        Stop disk I/O stress injection.
        
        Returns:
            bool: True if injection stopped successfully, False otherwise
        """
        if self.process is None:
            self.logger.warning("No disk stress injection running")
            return False
        
        try:
            # Stop monitoring
            self.stop_monitoring.set()
            
            # Terminate stress process
            if self.process.poll() is None:
                self.logger.info("Stopping disk stress injection...")
                self.process.terminate()
                
                # Wait for graceful termination
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.logger.warning("Stress process did not terminate gracefully, killing...")
                    self.process.kill()
                    self.process.wait()
            
            # Wait for monitoring thread to finish
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            # Clean up temporary files created by stress-ng
            self._cleanup_temp_files()
            
            self.logger.info("Disk stress injection stopped")
            self.process = None
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping disk stress injection: {e}")
            return False
    
    def _cleanup_temp_files(self):
        """Clean up temporary files created during stress testing."""
        try:
            temp_path = self.config['temp_path']
            # Look for stress-ng temporary files
            for filename in os.listdir(temp_path):
                if filename.startswith('stress-ng-hdd-'):
                    filepath = os.path.join(temp_path, filename)
                    try:
                        os.remove(filepath)
                        self.logger.debug(f"Cleaned up temporary file: {filepath}")
                    except Exception as e:
                        self.logger.warning(f"Failed to clean up {filepath}: {e}")
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current injection status.
        
        Returns:
            Dict containing status information
        """
        is_running = self.process is not None and self.process.poll() is None
        
        status = {
            'running': is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'duration': self.config['duration'],
            'file_size': self.config['file_size'],
            'workers': self.config['workers'],
            'temp_path': self.config['temp_path'],
            'target_container': self.config.get('target_container'),
            'monitoring_data_points': len(self.injection_log)
        }
        
        if is_running and self.start_time:
            elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            status['elapsed_seconds'] = elapsed
            status['remaining_seconds'] = max(0, self.config['duration'] - elapsed)
        
        # Add current disk info
        disk_usage = self._get_disk_usage(self.config['temp_path'])
        status.update({
            'current_disk_used_percent': disk_usage['used_percent'],
            'current_disk_free_gb': disk_usage['free_gb']
        })
        
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


def create_disk_injector(config_file: str) -> DiskStressInjector:
    """
    Create disk stress injector from configuration file.
    
    Args:
        config_file: Path to JSON configuration file
        
    Returns:
        DiskStressInjector instance
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return DiskStressInjector(config)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Disk I/O Stress Injection Tool")
    parser.add_argument('--config', required=True, help='Configuration file path')
    parser.add_argument('--action', choices=['start', 'stop', 'status'], 
                       default='start', help='Action to perform')
    parser.add_argument('--log-file', help='Path to save injection log')
    
    args = parser.parse_args()
    
    try:
        injector = create_disk_injector(args.config)
        
        if args.action == 'start':
            if injector.start_injection():
                print("Disk stress injection started successfully")
                # Wait for completion
                while injector.get_status()['running']:
                    time.sleep(1)
                print("Disk stress injection completed")
            else:
                print("Failed to start disk stress injection")
                exit(1)
        
        elif args.action == 'stop':
            if injector.stop_injection():
                print("Disk stress injection stopped successfully")
            else:
                print("Failed to stop disk stress injection")
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