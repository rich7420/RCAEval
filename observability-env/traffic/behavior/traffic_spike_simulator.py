"""
Traffic Spike Simulator for Realistic Load Patterns

This module implements irregular traffic patterns including sudden spikes,
gradual increases, flash crowds, and realistic e-commerce traffic variations.
"""

import random
import time
import threading
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

from user_behavior_simulator import UserBehaviorSimulator, BehaviorConfig, ApplicationType
from config_loader import BehaviorConfigLoader

logger = logging.getLogger(__name__)

class TrafficPatternType(Enum):
    """Types of traffic patterns."""
    STEADY = "steady"           # 穩定流量
    GRADUAL_INCREASE = "gradual_increase"  # 漸增
    SUDDEN_SPIKE = "sudden_spike"          # 突然突波
    FLASH_CROWD = "flash_crowd"            # 閃電人群
    WAVE_PATTERN = "wave_pattern"          # 波浪模式
    RANDOM_BURSTS = "random_bursts"        # 隨機爆發
    BLACK_FRIDAY = "black_friday"          # 黑色星期五模式
    DDOS_SIMULATION = "ddos_simulation"    # DDoS 模擬

@dataclass
class TrafficEvent:
    """Represents a traffic event/spike."""
    event_type: TrafficPatternType
    start_time: float
    duration: float
    peak_users: int
    base_users: int
    description: str

class TrafficSpikeSimulator:
    """
    Simulates realistic traffic spikes and irregular patterns.
    
    Features:
    - 突然的流量突波 (Sudden traffic spikes)
    - 漸進式流量增長 (Gradual traffic growth)
    - 閃電人群效應 (Flash crowd effects)
    - 隨機流量爆發 (Random traffic bursts)
    - 真實的電商流量模式 (Realistic e-commerce patterns)
    """
    
    def __init__(self, behavior_profile: str = "default"):
        """Initialize traffic spike simulator."""
        self.behavior_profile = behavior_profile
        self.base_simulator = None
        self.active_users: Dict[str, threading.Thread] = {}
        self.user_counter = 0
        self.running = False
        self.traffic_events: List[TrafficEvent] = []
        self.current_load = 0
        
        self.setup_simulator()
    
    def setup_simulator(self):
        """Setup the base behavior simulator."""
        config_loader = BehaviorConfigLoader()
        self.base_simulator = config_loader.create_simulator(self.behavior_profile)
        logger.info(f"Traffic spike simulator initialized with profile: {self.behavior_profile}")
    
    def calculate_user_load(self, current_time: float) -> int:
        """
        Calculate current user load based on active traffic events.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Number of users that should be active
        """
        base_load = 1  # Minimum base load
        spike_load = 0
        
        # Check all active traffic events
        for event in self.traffic_events:
            if event.start_time <= current_time <= event.start_time + event.duration:
                # Calculate load contribution from this event
                event_progress = (current_time - event.start_time) / event.duration
                
                if event.event_type == TrafficPatternType.SUDDEN_SPIKE:
                    # Immediate spike then gradual decline
                    if event_progress < 0.1:  # First 10% - rapid increase
                        spike_load += int(event.peak_users * (event_progress / 0.1))
                    else:  # Remaining 90% - gradual decline
                        decline_factor = 1 - ((event_progress - 0.1) / 0.9)
                        spike_load += int(event.peak_users * decline_factor)
                
                elif event.event_type == TrafficPatternType.GRADUAL_INCREASE:
                    # Smooth increase over time
                    spike_load += int(event.peak_users * event_progress)
                
                elif event.event_type == TrafficPatternType.FLASH_CROWD:
                    # Very sharp spike then quick drop
                    if event_progress < 0.05:  # First 5% - very rapid increase
                        spike_load += int(event.peak_users * (event_progress / 0.05))
                    elif event_progress < 0.2:  # Next 15% - maintain peak
                        spike_load += event.peak_users
                    else:  # Remaining 80% - rapid decline
                        decline_factor = 1 - ((event_progress - 0.2) / 0.8)
                        spike_load += int(event.peak_users * decline_factor * 0.3)
                
                elif event.event_type == TrafficPatternType.WAVE_PATTERN:
                    # Sine wave pattern
                    wave_factor = math.sin(event_progress * math.pi * 4)  # 4 waves during duration
                    spike_load += int(event.peak_users * max(0, wave_factor))
                
                elif event.event_type == TrafficPatternType.RANDOM_BURSTS:
                    # Random spikes throughout the duration
                    if random.random() < 0.1:  # 10% chance of burst each check
                        spike_load += random.randint(event.base_users, event.peak_users)
                    else:
                        spike_load += event.base_users
                
                elif event.event_type == TrafficPatternType.BLACK_FRIDAY:
                    # Realistic Black Friday pattern - multiple waves
                    # Morning rush (0-20%), lunch dip (20-40%), afternoon peak (40-70%), evening surge (70-100%)
                    if event_progress < 0.2:  # Morning rush
                        spike_load += int(event.peak_users * 0.6 * (event_progress / 0.2))
                    elif event_progress < 0.4:  # Lunch dip
                        spike_load += int(event.peak_users * 0.3)
                    elif event_progress < 0.7:  # Afternoon peak
                        afternoon_factor = (event_progress - 0.4) / 0.3
                        spike_load += int(event.peak_users * (0.3 + 0.7 * afternoon_factor))
                    else:  # Evening surge
                        spike_load += int(event.peak_users * 1.2)  # 120% of peak
                
                elif event.event_type == TrafficPatternType.DDOS_SIMULATION:
                    # Sustained high load with random variations
                    variation = random.uniform(0.8, 1.2)
                    spike_load += int(event.peak_users * variation)
        
        return max(base_load, base_load + spike_load)
    
    def schedule_traffic_spike(self, 
                              spike_type: TrafficPatternType,
                              delay_seconds: float = 0,
                              duration: float = 60,
                              peak_users: int = 50,
                              base_users: int = 5,
                              description: str = "") -> TrafficEvent:
        """
        Schedule a traffic spike event.
        
        Args:
            spike_type: Type of traffic pattern
            delay_seconds: Delay before spike starts
            duration: Duration of the spike in seconds
            peak_users: Maximum number of concurrent users
            base_users: Base number of users for gradual patterns
            description: Description of the event
            
        Returns:
            TrafficEvent object
        """
        start_time = time.time() + delay_seconds
        
        event = TrafficEvent(
            event_type=spike_type,
            start_time=start_time,
            duration=duration,
            peak_users=peak_users,
            base_users=base_users,
            description=description or f"{spike_type.value} spike"
        )
        
        self.traffic_events.append(event)
        
        logger.info(f"Scheduled {spike_type.value} spike: "
                   f"starts in {delay_seconds}s, duration {duration}s, peak {peak_users} users")
        
        return event
    
    def create_user_thread(self, user_id: str, duration: float = None) -> threading.Thread:
        """Create a user simulation thread."""
        if duration is None:
            duration = random.uniform(30, 300)  # 30 seconds to 5 minutes
        
        def user_simulation():
            try:
                # Create individual simulator for this user
                config_loader = BehaviorConfigLoader()
                user_simulator = config_loader.create_simulator(self.behavior_profile)
                
                # Simulate user session
                journey = user_simulator.simulate_user_session(duration)
                
                logger.info(f"User {user_id} completed: {len(journey.behaviors)} behaviors, "
                           f"{journey.total_duration:.1f}s, checkout: {journey.completed_checkout}")
                
            except Exception as e:
                logger.error(f"Error in user {user_id} simulation: {e}")
            finally:
                # Remove from active users
                if user_id in self.active_users:
                    del self.active_users[user_id]
        
        thread = threading.Thread(target=user_simulation, name=f"User-{user_id}")
        return thread
    
    def spawn_users(self, target_users: int):
        """Spawn users to reach target concurrent user count."""
        current_users = len(self.active_users)
        
        if target_users > current_users:
            # Need to spawn more users
            users_to_spawn = target_users - current_users
            
            for _ in range(users_to_spawn):
                self.user_counter += 1
                user_id = f"spike_user_{self.user_counter:05d}"
                
                # Create and start user thread
                user_thread = self.create_user_thread(user_id)
                user_thread.start()
                
                self.active_users[user_id] = user_thread
                
                # Small delay between spawns to avoid overwhelming
                time.sleep(random.uniform(0.1, 0.5))
        
        elif target_users < current_users:
            # Users will naturally complete and reduce load
            # We don't forcefully terminate users for realistic behavior
            pass
    
    def run_traffic_simulation(self, total_duration: float = 600):
        """
        Run the traffic simulation with scheduled spikes.
        
        Args:
            total_duration: Total simulation duration in seconds
        """
        logger.info(f"Starting traffic spike simulation for {total_duration} seconds")
        logger.info(f"Scheduled events: {len(self.traffic_events)}")
        
        self.running = True
        start_time = time.time()
        end_time = start_time + total_duration
        
        last_log_time = start_time
        
        try:
            while time.time() < end_time and self.running:
                current_time = time.time()
                
                # Calculate target user load
                target_load = self.calculate_user_load(current_time)
                current_load = len(self.active_users)
                
                # Adjust user load
                if target_load != current_load:
                    self.spawn_users(target_load)
                
                # Log status every 10 seconds
                if current_time - last_log_time >= 10:
                    active_events = [e for e in self.traffic_events 
                                   if e.start_time <= current_time <= e.start_time + e.duration]
                    
                    event_names = [e.event_type.value for e in active_events]
                    
                    logger.info(f"Traffic status: {len(self.active_users)} active users, "
                               f"target: {target_load}, active events: {event_names}")
                    
                    last_log_time = current_time
                
                # Update current load for monitoring
                self.current_load = len(self.active_users)
                
                # Check every second
                time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("Traffic simulation interrupted by user")
        
        finally:
            self.running = False
            
            # Wait for remaining users to complete
            logger.info("Waiting for remaining users to complete...")
            for user_id, thread in list(self.active_users.items()):
                thread.join(timeout=30)
                if thread.is_alive():
                    logger.warning(f"User {user_id} did not complete in time")
            
            logger.info("Traffic spike simulation completed")
    
    def stop_simulation(self):
        """Stop the traffic simulation."""
        logger.info("Stopping traffic simulation...")
        self.running = False
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get current simulation status."""
        current_time = time.time()
        active_events = [e for e in self.traffic_events 
                        if e.start_time <= current_time <= e.start_time + e.duration]
        
        return {
            "running": self.running,
            "active_users": len(self.active_users),
            "total_events_scheduled": len(self.traffic_events),
            "active_events": len(active_events),
            "active_event_types": [e.event_type.value for e in active_events],
            "current_load": self.current_load
        }


class PresetTrafficScenarios:
    """Predefined traffic scenarios for common use cases."""
    
    @staticmethod
    def create_ecommerce_day(simulator: TrafficSpikeSimulator):
        """Create a realistic e-commerce day with multiple spikes."""
        # Morning rush (9 AM equivalent)
        simulator.schedule_traffic_spike(
            TrafficPatternType.GRADUAL_INCREASE,
            delay_seconds=10,
            duration=120,
            peak_users=30,
            description="Morning rush hour"
        )
        
        # Lunch time spike (12 PM equivalent)
        simulator.schedule_traffic_spike(
            TrafficPatternType.SUDDEN_SPIKE,
            delay_seconds=180,
            duration=90,
            peak_users=50,
            description="Lunch time shopping"
        )
        
        # Afternoon steady increase
        simulator.schedule_traffic_spike(
            TrafficPatternType.WAVE_PATTERN,
            delay_seconds=300,
            duration=180,
            peak_users=40,
            description="Afternoon browsing waves"
        )
        
        # Evening flash crowd (sale announcement)
        simulator.schedule_traffic_spike(
            TrafficPatternType.FLASH_CROWD,
            delay_seconds=500,
            duration=60,
            peak_users=100,
            description="Flash sale announcement"
        )
    
    @staticmethod
    def create_black_friday(simulator: TrafficSpikeSimulator):
        """Create Black Friday traffic pattern."""
        simulator.schedule_traffic_spike(
            TrafficPatternType.BLACK_FRIDAY,
            delay_seconds=5,
            duration=600,  # 10 minutes of Black Friday madness
            peak_users=200,
            description="Black Friday shopping frenzy"
        )
    
    @staticmethod
    def create_ddos_test(simulator: TrafficSpikeSimulator):
        """Create DDoS simulation for testing."""
        simulator.schedule_traffic_spike(
            TrafficPatternType.DDOS_SIMULATION,
            delay_seconds=10,
            duration=120,
            peak_users=500,
            description="DDoS attack simulation"
        )
    
    @staticmethod
    def create_random_chaos(simulator: TrafficSpikeSimulator):
        """Create chaotic random traffic for stress testing."""
        # Multiple random bursts
        for i in range(5):
            simulator.schedule_traffic_spike(
                TrafficPatternType.RANDOM_BURSTS,
                delay_seconds=i * 60 + random.randint(0, 30),
                duration=random.randint(30, 90),
                peak_users=random.randint(20, 80),
                base_users=random.randint(5, 15),
                description=f"Random chaos burst {i+1}"
            )
    
    @staticmethod
    def create_viral_event(simulator: TrafficSpikeSimulator):
        """Create viral social media event traffic."""
        # Initial small spike
        simulator.schedule_traffic_spike(
            TrafficPatternType.SUDDEN_SPIKE,
            delay_seconds=10,
            duration=60,
            peak_users=25,
            description="Initial viral post"
        )
        
        # Viral explosion
        simulator.schedule_traffic_spike(
            TrafficPatternType.FLASH_CROWD,
            delay_seconds=80,
            duration=120,
            peak_users=150,
            description="Viral explosion"
        )
        
        # Sustained interest
        simulator.schedule_traffic_spike(
            TrafficPatternType.GRADUAL_INCREASE,
            delay_seconds=220,
            duration=180,
            peak_users=60,
            description="Sustained viral interest"
        )


def main():
    """Example usage of traffic spike simulator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Traffic Spike Simulator")
    parser.add_argument("--scenario", choices=["ecommerce", "black_friday", "ddos", "chaos", "viral"], 
                       default="ecommerce", help="Traffic scenario to simulate")
    parser.add_argument("--profile", default="default", help="Behavior profile to use")
    parser.add_argument("--duration", type=int, default=600, help="Simulation duration in seconds")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create simulator
    simulator = TrafficSpikeSimulator(args.profile)
    
    # Setup scenario
    if args.scenario == "ecommerce":
        PresetTrafficScenarios.create_ecommerce_day(simulator)
    elif args.scenario == "black_friday":
        PresetTrafficScenarios.create_black_friday(simulator)
    elif args.scenario == "ddos":
        PresetTrafficScenarios.create_ddos_test(simulator)
    elif args.scenario == "chaos":
        PresetTrafficScenarios.create_random_chaos(simulator)
    elif args.scenario == "viral":
        PresetTrafficScenarios.create_viral_event(simulator)
    
    print(f"Starting {args.scenario} traffic simulation...")
    print(f"Duration: {args.duration} seconds")
    print(f"Behavior profile: {args.profile}")
    print("Press Ctrl+C to stop early")
    
    try:
        simulator.run_traffic_simulation(args.duration)
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    
    print("Traffic spike simulation completed!")


if __name__ == "__main__":
    main()