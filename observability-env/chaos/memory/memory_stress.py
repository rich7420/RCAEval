#!/usr/bin/env python3
"""
Memory Stress Injection Module

This module implements memory stress injection using stress-ng with configurable
allocation patterns, limits, and safety mechanisms for chaos engineering experiments.
"""

import subprocess
import time
import logging
import json
import threading
from typing import Dict, Optional, Any
from datetime import datetime, timezone
import psutil
import signal
import os

class MemoryStressInjector:
    """Memory stress injection with monitoring and safety mechanisms."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize memory stress injector.
        
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
        required_fields = ['size', 'duration']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")
        
        # Validate memory size
        size = self.config['size']
        if isinstance(size, str):
            # Parse size with units (e.g., "1G", "512M", "50%")
            if size.endswith('%'):
                percent = int(size[:-1])
                if not 1 <= percent <= 90:  # Max 90% for safety
                    raise ValueError("Memory percentage must be between 1-90%")
            elif not any(size.endswith(unit) for unit in ['K', 'M', 'G', 'T']):
                raise ValueError("Memory size must include unit (K, M, G, T) or be percentage")
        elif isinstance(size, int):
            if size <= 0:
                raise ValueError("Memory size must be positive")
        else:
            raise ValueError("Memory size must be string with unit or integer")
        
        # Validate duration
        if not isinstance(self.config['duration'], (int, float)) or self.config['duration'] <= 0:
            raise ValueError("Duration must be a positive number")
        
        # Set defaults
        self.config.setdefault('workers', 1)  # Number of memory workers
        self.config.setdefault('safety_threshold', 90)  # Memory usage threshold for safety stop
        self.config.setdefault('monitoring_interval', 1)  # seconds
        self.config.setdefault('method', 'all')  # Memory stress method
        self.config.setdefault('target_container', None)  # specific container targeting
        self.config.setdefault('oom_score_adj', None)  # OOM killer adjustment
    
    def _calculate_memory_size(self) -> str:
        """Calculate actual memory size based on configuration."""
        size = self.config['size']
        
        if isinstance(size, str) and size.endswith('%'):
            percent = int(size[:-1])
            total_memory = psutil.virtual_memory().total
            actual_bytes = int(total_memory * percent / 100)
            
            # Convert to appropriate unit
            if actual_bytes >= 1024**3:  # GB
                return f"{actual_bytes // (1024**3)}G"
            elif actual_bytes >= 1024**2:  # MB
                return f"{actual_bytes // (1024**2)}M"
            else:  # KB
                return f"{actual_bytes // 1024}K"
        else:
            return str(size)
    
    def _build_stress_command(self) -> list:
        """Build stress-ng command with parameters."""
        memory_size = self._calculate_memory_size()
        duration = int(self.config['duration'])
        workers = self.config['workers']
        
        cmd = [
            'stress-ng',
            '--vm', str(workers),
            '--vm-bytes', memory_size,
            '--timeout', f'{duration}s',
            '--metrics-brief',
            '--verify'
        ]
        
        # Add memory method if specified
        if 'method' in self.config and self.config['method'] != 'all':
            cmd.extend(['--vm-method', self.config['method']])
        
        # Add memory allocation options
        if 'vm_flags' in self.config:
            for flag in self.config['vm_flags']:
                cmd.extend(['--vm-flags', flag])
        
        # Add OOM score adjustment
        if self.config.get('oom_score_adj') is not None:
            cmd.extend(['--oom-score-adj', str(self.config['oom_score_adj'])])
        
        return cmd
    
    def _monitor_memory_usage(self):
        """Monitor memory usage during stress injection."""
        while not self.stop_monitoring.is_set():
            try:
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()
                
                timestamp = datetime.now(timezone.utc).isoformat()
                
                # Log monitoring data
                monitor_data = {
                    'timestamp': timestamp,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_total_gb': memory.total / (1024**3),
                    'swap_percent': swap.percent,
                    'swap_used_gb': swap.used / (1024**3),
                    'active_stress': self.process is not None and self.process.poll() is None
                }
                
                self.injection_log.append(monitor_data)
                
                # Safety check - stop if memory usage is dangerously high
                safety_threshold = self.config['safety_threshold']
                if memory.percent > safety_threshold:
                    self.logger.warning(
                        f"Memory usage {memory.percent}% exceeds safety threshold {safety_threshold}%. "
                        "Stopping stress injection."
                    )
                    self.stop_injection()
                    break
                
                # Log current status
                if len(self.injection_log) % 10 == 0:  # Log every 10 seconds
                    self.logger.info(
                        f"Memory: {memory.percent}%, Available: {memory.available / (1024**3):.1f}GB, "
                        f"Swap: {swap.percent}%"
                    )
                
            except Exception as e:
                self.logger.error(f"Error during monitoring: {e}")
            
            time.sleep(self.config['monitoring_interval'])
    
    def start_injection(self) -> bool:
        """
        Start memory stress injection.
        
        Returns:
            bool: True if injection started successfully, False otherwise
        """
        if self.process is not None:
            self.logger.warning("Memory stress injection already running")
            return False
        
        try:
            # Check available memory before starting
            memory = psutil.virtual_memory()
            memory_size = self._calculate_memory_size()
            
            self.logger.info(
                f"Starting memory stress injection: {memory_size} with {self.config['workers']} workers"
            )
            self.logger.info(
                f"Current memory usage: {memory.percent}%, Available: {memory.available / (1024**3):.1f}GB"
            )
            
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
            self.monitoring_thread = threading.Thread(target=self._monitor_memory_usage)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            self.logger.info(f"Memory stress injection started with PID {self.process.pid}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start memory stress injection: {e}")
            return False
    
    def stop_injection(self) -> bool:
        """
        Stop memory stress injection.
        
        Returns:
            bool: True if injection stopped successfully, False otherwise
        """
        if self.process is None:
            self.logger.warning("No memory stress injection running")
            return False
        
        try:
            # Stop monitoring
            self.stop_monitoring.set()
            
            # Terminate stress process
            if self.process.poll() is None:
                self.logger.info("Stopping memory stress injection...")
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
            
            self.logger.info("Memory stress injection stopped")
            self.process = None
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping memory stress injection: {e}")
            return False
    
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
            'memory_size': self._calculate_memory_size(),
            'workers': self.config['workers'],
            'target_container': self.config.get('target_container'),
            'monitoring_data_points': len(self.injection_log)
        }
        
        if is_running and self.start_time:
            elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            status['elapsed_seconds'] = elapsed
            status['remaining_seconds'] = max(0, self.config['duration'] - elapsed)
        
        # Add current memory info
        memory = psutil.virtual_memory()
        status['current_memory_percent'] = memory.percent
        status['current_memory_available_gb'] = memory.available / (1024**3)
        
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


def create_memory_injector(config_file: str) -> MemoryStressInjector:
    """
    Create memory stress injector from configuration file.
    
    Args:
        config_file: Path to JSON configuration file
        
    Returns:
        MemoryStressInjector instance
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return MemoryStressInjector(config)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory Stress Injection Tool")
    parser.add_argument('--config', required=True, help='Configuration file path')
    parser.add_argument('--action', choices=['start', 'stop', 'status'], 
                       default='start', help='Action to perform')
    parser.add_argument('--log-file', help='Path to save injection log')
    
    args = parser.parse_args()
    
    try:
        injector = create_memory_injector(args.config)
        
        if args.action == 'start':
            if injector.start_injection():
                print("Memory stress injection started successfully")
                # Wait for completion
                while injector.get_status()['running']:
                    time.sleep(1)
                print("Memory stress injection completed")
            else:
                print("Failed to start memory stress injection")
                exit(1)
        
        elif args.action == 'stop':
            if injector.stop_injection():
                print("Memory stress injection stopped successfully")
            else:
                print("Failed to stop memory stress injection")
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