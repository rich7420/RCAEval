#!/usr/bin/env python3
"""
Standalone runner for User Behavior Simulator.

This script allows running the behavior simulator independently of Locust
for testing, validation, and data generation purposes.
"""

import argparse
import json
import time
import threading
import signal
import sys
from datetime import datetime
from typing import List, Optional
import logging

from user_behavior_simulator import UserBehaviorSimulator, UserJourney
from config_loader import BehaviorConfigLoader, list_available_profiles

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BehaviorSimulationRunner:
    """
    Runner class for executing behavior simulations with multiple concurrent users.
    """
    
    def __init__(self, profile: str = "default", config_file: str = "behavior_config.yaml"):
        """
        Initialize the simulation runner.
        
        Args:
            profile: Behavior configuration profile to use
            config_file: Path to configuration file
        """
        self.profile = profile
        self.config_file = config_file
        self.simulator = None
        self.running = False
        self.user_threads: List[threading.Thread] = []
        self.completed_journeys: List[UserJourney] = []
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.setup_simulator()
    
    def setup_simulator(self):
        """Initialize the behavior simulator."""
        try:
            config_loader = BehaviorConfigLoader(self.config_file)
            self.simulator = config_loader.create_simulator(self.profile)
            logger.info(f"Initialized behavior simulator with profile: {self.profile}")
        except Exception as e:
            logger.error(f"Failed to initialize simulator: {e}")
            sys.exit(1)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop_simulation()
    
    def simulate_single_user(self, user_id: str, duration: Optional[float] = None) -> UserJourney:
        """
        Simulate a single user session.
        
        Args:
            user_id: Unique identifier for the user
            duration: Optional session duration override
            
        Returns:
            Completed UserJourney object
        """
        try:
            # Create a temporary simulator instance for this user to avoid conflicts
            config_loader = BehaviorConfigLoader(self.config_file)
            user_simulator = config_loader.create_simulator(self.profile)
            
            logger.info(f"Starting simulation for user: {user_id}")
            journey = user_simulator.simulate_user_session(duration, user_id)
            
            # Store completed journey
            self.completed_journeys.append(journey)
            
            logger.info(f"Completed simulation for user: {user_id} "
                       f"({len(journey.behaviors)} behaviors, "
                       f"{journey.total_duration:.1f}s duration)")
            
            return journey
            
        except Exception as e:
            logger.error(f"Error simulating user {user_id}: {e}")
            return None
    
    def run_concurrent_simulation(self, 
                                num_users: int, 
                                duration: float = 300.0,
                                spawn_rate: float = 1.0) -> List[UserJourney]:
        """
        Run simulation with multiple concurrent users.
        
        Args:
            num_users: Number of concurrent users to simulate
            duration: Total simulation duration in seconds
            spawn_rate: Rate at which to spawn new users (users per second)
            
        Returns:
            List of completed UserJourney objects
        """
        logger.info(f"Starting concurrent simulation: {num_users} users, "
                   f"{duration}s duration, {spawn_rate} spawn rate")
        
        self.running = True
        self.completed_journeys.clear()
        
        # Calculate spawn timing
        spawn_interval = 1.0 / spawn_rate if spawn_rate > 0 else 0
        
        # Spawn users gradually
        for i in range(num_users):
            if not self.running:
                break
                
            user_id = f"sim_user_{i+1:04d}"
            
            # Calculate individual user session duration
            # Users spawned later get proportionally less time
            remaining_time = duration - (i * spawn_interval)
            user_duration = max(30.0, remaining_time)  # Minimum 30 seconds
            
            # Create and start user thread
            user_thread = threading.Thread(
                target=self.simulate_single_user,
                args=(user_id, user_duration),
                name=f"UserThread-{user_id}"
            )
            
            user_thread.start()
            self.user_threads.append(user_thread)
            
            logger.info(f"Spawned user {i+1}/{num_users}: {user_id}")
            
            # Wait before spawning next user
            if i < num_users - 1:  # Don't wait after the last user
                time.sleep(spawn_interval)
        
        # Wait for all users to complete or timeout
        logger.info("Waiting for all user sessions to complete...")
        
        for thread in self.user_threads:
            thread.join(timeout=duration + 60)  # Extra 60 seconds grace period
            if thread.is_alive():
                logger.warning(f"Thread {thread.name} did not complete in time")
        
        self.running = False
        
        logger.info(f"Concurrent simulation completed. "
                   f"Total journeys: {len(self.completed_journeys)}")
        
        return self.completed_journeys
    
    def stop_simulation(self):
        """Stop the running simulation."""
        logger.info("Stopping simulation...")
        self.running = False
        
        # Wait for threads to complete
        for thread in self.user_threads:
            if thread.is_alive():
                thread.join(timeout=5)
    
    def generate_report(self, output_file: Optional[str] = None) -> dict:
        """
        Generate a comprehensive report of the simulation results.
        
        Args:
            output_file: Optional file path to save the report
            
        Returns:
            Dictionary containing the report data
        """
        if not self.completed_journeys:
            logger.warning("No completed journeys to report on")
            return {}
        
        # Calculate statistics
        total_journeys = len(self.completed_journeys)
        total_behaviors = sum(len(j.behaviors) for j in self.completed_journeys)
        total_errors = sum(j.error_count for j in self.completed_journeys)
        completed_checkouts = sum(1 for j in self.completed_journeys if j.completed_checkout)
        
        # Behavior distribution
        behavior_counts = {}
        for journey in self.completed_journeys:
            for behavior in journey.behaviors:
                behavior_counts[behavior.value] = behavior_counts.get(behavior.value, 0) + 1
        
        behavior_percentages = {}
        if total_behaviors > 0:
            for behavior, count in behavior_counts.items():
                behavior_percentages[behavior] = (count / total_behaviors) * 100
        
        # Duration statistics
        durations = [j.total_duration for j in self.completed_journeys]
        avg_duration = sum(durations) / len(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        
        # Products viewed statistics
        products_viewed = [len(j.products_viewed) for j in self.completed_journeys]
        avg_products = sum(products_viewed) / len(products_viewed) if products_viewed else 0
        
        # Cart statistics
        cart_sizes = [len(j.cart_items) for j in self.completed_journeys]
        avg_cart_size = sum(cart_sizes) / len(cart_sizes) if cart_sizes else 0
        
        report = {
            "simulation_summary": {
                "profile": self.profile,
                "total_journeys": total_journeys,
                "total_behaviors": total_behaviors,
                "total_errors": total_errors,
                "completed_checkouts": completed_checkouts,
                "checkout_completion_rate": (completed_checkouts / total_journeys * 100) if total_journeys > 0 else 0
            },
            "behavior_distribution": {
                "actual": behavior_percentages,
                "target": {
                    "browse": self.simulator.config.browse_percentage,
                    "cart": self.simulator.config.cart_percentage,
                    "checkout": self.simulator.config.checkout_percentage,
                    "error": self.simulator.config.error_percentage
                }
            },
            "duration_statistics": {
                "average_seconds": avg_duration,
                "minimum_seconds": min_duration,
                "maximum_seconds": max_duration
            },
            "user_behavior_statistics": {
                "average_products_viewed": avg_products,
                "average_cart_size": avg_cart_size,
                "average_behaviors_per_session": total_behaviors / total_journeys if total_journeys > 0 else 0
            },
            "configuration": {
                "application_type": self.simulator.config.application_type.value,
                "think_time_range": [self.simulator.config.min_think_time, self.simulator.config.max_think_time],
                "session_duration_range": [self.simulator.config.session_duration_min, self.simulator.config.session_duration_max],
                "max_cart_items": self.simulator.config.max_cart_items,
                "checkout_completion_rate": self.simulator.config.checkout_completion_rate
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Save report if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to: {output_file}")
        
        return report
    
    def export_journey_details(self, output_file: str):
        """
        Export detailed journey data for analysis.
        
        Args:
            output_file: File path to save the detailed journey data
        """
        journey_data = []
        
        for journey in self.completed_journeys:
            journey_detail = {
                "journey_id": journey.journey_id,
                "user_id": journey.user_id,
                "session_id": journey.session_id,
                "start_time": journey.start_time.isoformat(),
                "total_duration": journey.total_duration,
                "behaviors": [b.value for b in journey.behaviors],
                "behavior_count": len(journey.behaviors),
                "products_viewed": journey.products_viewed,
                "products_viewed_count": len(journey.products_viewed),
                "cart_items": journey.cart_items,
                "cart_items_count": len(journey.cart_items),
                "completed_checkout": journey.completed_checkout,
                "error_count": journey.error_count
            }
            journey_data.append(journey_detail)
        
        with open(output_file, 'w') as f:
            json.dump(journey_data, f, indent=2)
        
        logger.info(f"Journey details exported to: {output_file}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="User Behavior Simulator Runner")
    
    parser.add_argument("--profile", "-p", default="default",
                       help="Behavior configuration profile to use")
    parser.add_argument("--config", "-c", default="behavior_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--users", "-u", type=int, default=10,
                       help="Number of concurrent users to simulate")
    parser.add_argument("--duration", "-d", type=float, default=300.0,
                       help="Total simulation duration in seconds")
    parser.add_argument("--spawn-rate", "-s", type=float, default=1.0,
                       help="User spawn rate (users per second)")
    parser.add_argument("--output", "-o", 
                       help="Output file for simulation report")
    parser.add_argument("--export-journeys", 
                       help="Export detailed journey data to file")
    parser.add_argument("--list-profiles", action="store_true",
                       help="List available behavior profiles")
    parser.add_argument("--single-user", action="store_true",
                       help="Run single user simulation for testing")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # List profiles if requested
    if args.list_profiles:
        profiles = list_available_profiles(args.config)
        print("\nAvailable behavior profiles:")
        print("=" * 50)
        for name, summary in profiles.items():
            print(f"\n{name}:")
            for key, value in summary.items():
                if key != 'profile_name':
                    print(f"  {key}: {value}")
        return
    
    # Create and run simulation
    runner = BehaviorSimulationRunner(args.profile, args.config)
    
    try:
        if args.single_user:
            # Single user test
            logger.info("Running single user simulation for testing...")
            journey = runner.simulate_single_user("test_user", args.duration)
            if journey:
                print(f"\nSingle user simulation completed:")
                print(f"  Duration: {journey.total_duration:.1f}s")
                print(f"  Behaviors: {len(journey.behaviors)}")
                print(f"  Products viewed: {len(journey.products_viewed)}")
                print(f"  Cart items: {len(journey.cart_items)}")
                print(f"  Completed checkout: {journey.completed_checkout}")
                print(f"  Errors: {journey.error_count}")
        else:
            # Multi-user simulation
            journeys = runner.run_concurrent_simulation(
                num_users=args.users,
                duration=args.duration,
                spawn_rate=args.spawn_rate
            )
            
            # Generate and display report
            report = runner.generate_report(args.output)
            
            print(f"\nSimulation Report:")
            print("=" * 50)
            print(f"Profile: {report['simulation_summary']['profile']}")
            print(f"Total journeys: {report['simulation_summary']['total_journeys']}")
            print(f"Total behaviors: {report['simulation_summary']['total_behaviors']}")
            print(f"Checkout completion rate: {report['simulation_summary']['checkout_completion_rate']:.1f}%")
            print(f"Average session duration: {report['duration_statistics']['average_seconds']:.1f}s")
            
            print(f"\nBehavior Distribution:")
            actual = report['behavior_distribution']['actual']
            target = report['behavior_distribution']['target']
            for behavior in ['browse', 'cart', 'checkout', 'error']:
                actual_pct = actual.get(behavior, 0)
                target_pct = target.get(behavior, 0)
                print(f"  {behavior}: {actual_pct:.1f}% (target: {target_pct:.1f}%)")
            
            # Export journey details if requested
            if args.export_journeys:
                runner.export_journey_details(args.export_journeys)
    
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()