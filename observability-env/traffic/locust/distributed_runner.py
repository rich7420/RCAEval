#!/usr/bin/env python3
"""
Distributed Locust runner for observability data collection.
Supports master-worker configuration for distributed load generation.
"""

import os
import sys
import json
import argparse
import subprocess
import time
import signal
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DistributedLocustRunner:
    """
    Manages distributed Locust execution for load testing.
    Supports both master and worker modes with configuration management.
    """
    
    def __init__(self, config_file="config.json"):
        """Initialize the distributed runner."""
        self.config_file = config_file
        self.config = self.load_config()
        self.processes = []
        self.master_process = None
        
    def load_config(self):
        """Load configuration from JSON file."""
        config_path = Path(__file__).parent / self.config_file
        
        default_config = {
            "master": {
                "host": "0.0.0.0",
                "port": 8089,
                "web_port": 8089,
                "expect_workers": 1
            },
            "worker": {
                "master_host": "localhost",
                "master_port": 8089
            },
            "load_test": {
                "users": 100,
                "spawn_rate": 10,
                "run_time": "5m",
                "host": "http://localhost:8080"
            },
            "applications": {
                "otel_demo": {
                    "host": "http://localhost:8080",
                    "locustfile": "otel_demo_users.py",
                    "user_class": "OTelDemoUser"
                },
                "online_boutique": {
                    "host": "http://localhost:80", 
                    "locustfile": "online_boutique_users.py",
                    "user_class": "OnlineBoutiqueUser"
                }
            }
        }
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    self._deep_merge(default_config, loaded_config)
            return default_config
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return default_config
    
    def _deep_merge(self, base_dict, update_dict):
        """Deep merge two dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def start_master(self, application="otel_demo", **kwargs):
        """Start Locust master node."""
        app_config = self.config["applications"][application]
        master_config = self.config["master"]
        load_config = self.config["load_test"]
        
        # Override with any provided kwargs
        for key, value in kwargs.items():
            if key in load_config:
                load_config[key] = value
        
        cmd = [
            "locust",
            "--master",
            "--host", app_config["host"],
            "--web-host", master_config["host"],
            "--web-port", str(master_config["web_port"]),
            "--expect-workers", str(master_config["expect_workers"]),
            "--locustfile", app_config["locustfile"],
            "--users", str(load_config["users"]),
            "--spawn-rate", str(load_config["spawn_rate"])
        ]
        
        if "user_class" in app_config:
            cmd.extend(["--class-picker", app_config["user_class"]])
        
        if load_config.get("run_time"):
            cmd.extend(["--run-time", load_config["run_time"]])
        
        logger.info(f"Starting Locust master: {' '.join(cmd)}")
        
        try:
            self.master_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info(f"Locust master started with PID: {self.master_process.pid}")
            return self.master_process
        except Exception as e:
            logger.error(f"Failed to start master: {e}")
            return None
    
    def start_worker(self, worker_id=None, application="otel_demo"):
        """Start Locust worker node."""
        app_config = self.config["applications"][application]
        worker_config = self.config["worker"]
        
        worker_id = worker_id or f"worker_{len(self.processes) + 1}"
        
        cmd = [
            "locust",
            "--worker",
            "--master-host", worker_config["master_host"],
            "--master-port", str(worker_config["master_port"]),
            "--locustfile", app_config["locustfile"]
        ]
        
        # Set environment variables for worker identification
        env = os.environ.copy()
        env["LOCUST_WORKER_ID"] = worker_id
        env["LOCUST_TEST_ID"] = f"test_{int(time.time())}"
        
        logger.info(f"Starting Locust worker {worker_id}: {' '.join(cmd)}")
        
        try:
            worker_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            self.processes.append({
                "id": worker_id,
                "process": worker_process,
                "type": "worker"
            })
            logger.info(f"Locust worker {worker_id} started with PID: {worker_process.pid}")
            return worker_process
        except Exception as e:
            logger.error(f"Failed to start worker {worker_id}: {e}")
            return None
    
    def start_distributed_test(self, application="otel_demo", num_workers=1, **kwargs):
        """Start a complete distributed test with master and workers."""
        logger.info(f"Starting distributed test for {application} with {num_workers} workers")
        
        # Update expected workers in config
        self.config["master"]["expect_workers"] = num_workers
        
        # Start master
        master = self.start_master(application, **kwargs)
        if not master:
            logger.error("Failed to start master, aborting")
            return False
        
        # Wait for master to initialize
        time.sleep(3)
        
        # Start workers
        for i in range(num_workers):
            worker_id = f"worker_{i+1}"
            worker = self.start_worker(worker_id, application)
            if not worker:
                logger.warning(f"Failed to start worker {worker_id}")
        
        logger.info(f"Distributed test started - Master PID: {master.pid}, Workers: {len(self.processes)}")
        return True
    
    def stop_all(self):
        """Stop all Locust processes."""
        logger.info("Stopping all Locust processes...")
        
        # Stop workers first
        for proc_info in self.processes:
            try:
                proc_info["process"].terminate()
                proc_info["process"].wait(timeout=10)
                logger.info(f"Stopped {proc_info['id']}")
            except subprocess.TimeoutExpired:
                proc_info["process"].kill()
                logger.warning(f"Force killed {proc_info['id']}")
            except Exception as e:
                logger.error(f"Error stopping {proc_info['id']}: {e}")
        
        # Stop master
        if self.master_process:
            try:
                self.master_process.terminate()
                self.master_process.wait(timeout=10)
                logger.info("Stopped master process")
            except subprocess.TimeoutExpired:
                self.master_process.kill()
                logger.warning("Force killed master process")
            except Exception as e:
                logger.error(f"Error stopping master: {e}")
        
        self.processes.clear()
        self.master_process = None
    
    def get_status(self):
        """Get status of all running processes."""
        status = {
            "master": None,
            "workers": []
        }
        
        if self.master_process:
            status["master"] = {
                "pid": self.master_process.pid,
                "running": self.master_process.poll() is None
            }
        
        for proc_info in self.processes:
            status["workers"].append({
                "id": proc_info["id"],
                "pid": proc_info["process"].pid,
                "running": proc_info["process"].poll() is None
            })
        
        return status
    
    def wait_for_completion(self):
        """Wait for all processes to complete."""
        logger.info("Waiting for test completion...")
        
        try:
            if self.master_process:
                self.master_process.wait()
            
            for proc_info in self.processes:
                proc_info["process"].wait()
                
            logger.info("All processes completed")
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping all processes")
            self.stop_all()


def signal_handler(signum, frame):
    """Handle interrupt signals."""
    logger.info("Received signal, shutting down...")
    if hasattr(signal_handler, 'runner'):
        signal_handler.runner.stop_all()
    sys.exit(0)


def main():
    """Main entry point for distributed runner."""
    parser = argparse.ArgumentParser(description="Distributed Locust Runner")
    parser.add_argument("--mode", choices=["master", "worker", "distributed"], 
                       default="distributed", help="Run mode")
    parser.add_argument("--application", choices=["otel_demo", "online_boutique"],
                       default="otel_demo", help="Target application")
    parser.add_argument("--workers", type=int, default=1, 
                       help="Number of workers (distributed mode only)")
    parser.add_argument("--users", type=int, default=100, help="Number of users")
    parser.add_argument("--spawn-rate", type=int, default=10, help="Spawn rate")
    parser.add_argument("--run-time", default="5m", help="Run time")
    parser.add_argument("--config", default="config.json", help="Configuration file")
    
    args = parser.parse_args()
    
    # Setup signal handlers
    runner = DistributedLocustRunner(args.config)
    signal_handler.runner = runner
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if args.mode == "master":
            runner.start_master(
                application=args.application,
                users=args.users,
                spawn_rate=args.spawn_rate,
                run_time=args.run_time
            )
            runner.wait_for_completion()
            
        elif args.mode == "worker":
            runner.start_worker(application=args.application)
            runner.wait_for_completion()
            
        elif args.mode == "distributed":
            runner.start_distributed_test(
                application=args.application,
                num_workers=args.workers,
                users=args.users,
                spawn_rate=args.spawn_rate,
                run_time=args.run_time
            )
            runner.wait_for_completion()
            
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        runner.stop_all()
        sys.exit(1)


if __name__ == "__main__":
    main()