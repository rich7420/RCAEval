"""
Configuration loader for User Behavior Simulator.

This module provides utilities to load and validate behavior configurations
from YAML files and create appropriate simulator instances.
"""

import yaml
import os
from typing import Dict, Any, Optional
from user_behavior_simulator import BehaviorConfig, ApplicationType, UserBehaviorSimulator
import logging

logger = logging.getLogger(__name__)

class BehaviorConfigLoader:
    """Loads and validates behavior configurations from YAML files."""
    
    def __init__(self, config_file: str = "behavior_config.yaml"):
        """
        Initialize the config loader.
        
        Args:
            config_file: Path to the YAML configuration file
        """
        self.config_file = config_file
        self.config_data = None
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from YAML file."""
        config_path = os.path.join(os.path.dirname(__file__), self.config_file)
        
        try:
            with open(config_path, 'r') as f:
                self.config_data = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
    
    def get_available_profiles(self) -> list:
        """Get list of available behavior profiles."""
        if not self.config_data:
            return []
        return list(self.config_data.keys())
    
    def create_behavior_config(self, profile: str = "default") -> BehaviorConfig:
        """
        Create a BehaviorConfig object from a configuration profile.
        
        Args:
            profile: Name of the configuration profile to use
            
        Returns:
            BehaviorConfig object with loaded settings
        """
        if not self.config_data or profile not in self.config_data:
            logger.error(f"Profile '{profile}' not found in configuration")
            raise ValueError(f"Profile '{profile}' not found")
        
        profile_config = self.config_data[profile]
        
        # Extract configuration sections
        traffic = profile_config.get('traffic_distribution', {})
        timing = profile_config.get('timing', {})
        journey = profile_config.get('journey', {})
        errors = profile_config.get('errors', {})
        application = profile_config.get('application', {})
        
        # Map application type
        app_type_str = application.get('type', 'otel_demo')
        if app_type_str == 'otel_demo':
            app_type = ApplicationType.OTEL_DEMO
        elif app_type_str == 'online_boutique':
            app_type = ApplicationType.ONLINE_BOUTIQUE
        else:
            logger.warning(f"Unknown application type: {app_type_str}, defaulting to otel_demo")
            app_type = ApplicationType.OTEL_DEMO
        
        # Create BehaviorConfig object
        config = BehaviorConfig(
            # Traffic distribution
            browse_percentage=traffic.get('browse_percentage', 70.0),
            cart_percentage=traffic.get('cart_percentage', 20.0),
            checkout_percentage=traffic.get('checkout_percentage', 8.0),
            error_percentage=traffic.get('error_percentage', 2.0),
            
            # Timing
            min_think_time=timing.get('min_think_time', 1.0),
            max_think_time=timing.get('max_think_time', 10.0),
            session_duration_min=timing.get('session_duration_min', 60.0),
            session_duration_max=timing.get('session_duration_max', 600.0),
            
            # Journey
            max_products_per_session=journey.get('max_products_per_session', 10),
            max_cart_items=journey.get('max_cart_items', 5),
            checkout_completion_rate=journey.get('checkout_completion_rate', 0.6),
            
            # Errors
            error_retry_attempts=errors.get('retry_attempts', 2),
            error_scenarios_enabled=errors.get('scenarios_enabled', True),
            
            # Application
            application_type=app_type,
            base_url=application.get('base_url', 'http://localhost:8080')
        )
        
        logger.info(f"Created behavior config for profile: {profile}")
        return config
    
    def create_simulator(self, profile: str = "default") -> UserBehaviorSimulator:
        """
        Create a UserBehaviorSimulator from a configuration profile.
        
        Args:
            profile: Name of the configuration profile to use
            
        Returns:
            UserBehaviorSimulator instance
        """
        config = self.create_behavior_config(profile)
        simulator = UserBehaviorSimulator(config)
        
        logger.info(f"Created behavior simulator for profile: {profile}")
        return simulator
    
    def validate_profile(self, profile: str) -> bool:
        """
        Validate a configuration profile.
        
        Args:
            profile: Name of the profile to validate
            
        Returns:
            True if profile is valid, False otherwise
        """
        try:
            config = self.create_behavior_config(profile)
            return config.validate()
        except Exception as e:
            logger.error(f"Profile validation failed for '{profile}': {e}")
            return False
    
    def get_profile_summary(self, profile: str) -> Dict[str, Any]:
        """
        Get a summary of a configuration profile.
        
        Args:
            profile: Name of the profile to summarize
            
        Returns:
            Dictionary with profile summary information
        """
        if not self.config_data or profile not in self.config_data:
            return {}
        
        profile_config = self.config_data[profile]
        
        traffic = profile_config.get('traffic_distribution', {})
        timing = profile_config.get('timing', {})
        journey = profile_config.get('journey', {})
        application = profile_config.get('application', {})
        
        return {
            'profile_name': profile,
            'application_type': application.get('type', 'otel_demo'),
            'traffic_distribution': {
                'browse': f"{traffic.get('browse_percentage', 70)}%",
                'cart': f"{traffic.get('cart_percentage', 20)}%",
                'checkout': f"{traffic.get('checkout_percentage', 8)}%",
                'error': f"{traffic.get('error_percentage', 2)}%"
            },
            'session_duration': f"{timing.get('session_duration_min', 60)}-{timing.get('session_duration_max', 600)}s",
            'think_time': f"{timing.get('min_think_time', 1)}-{timing.get('max_think_time', 10)}s",
            'checkout_completion_rate': f"{journey.get('checkout_completion_rate', 0.6) * 100}%",
            'max_cart_items': journey.get('max_cart_items', 5)
        }


def load_simulator_from_config(profile: str = "default", config_file: str = "behavior_config.yaml") -> UserBehaviorSimulator:
    """
    Convenience function to load a simulator from configuration.
    
    Args:
        profile: Configuration profile name
        config_file: Path to configuration file
        
    Returns:
        UserBehaviorSimulator instance
    """
    loader = BehaviorConfigLoader(config_file)
    return loader.create_simulator(profile)


def list_available_profiles(config_file: str = "behavior_config.yaml") -> Dict[str, Dict[str, Any]]:
    """
    List all available configuration profiles with summaries.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Dictionary mapping profile names to their summaries
    """
    loader = BehaviorConfigLoader(config_file)
    profiles = {}
    
    for profile_name in loader.get_available_profiles():
        profiles[profile_name] = loader.get_profile_summary(profile_name)
    
    return profiles


# Example usage
if __name__ == "__main__":
    # List available profiles
    profiles = list_available_profiles()
    print("Available behavior profiles:")
    for name, summary in profiles.items():
        print(f"\n{name}:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    
    # Create simulator from default profile
    simulator = load_simulator_from_config("default")
    print(f"\nCreated simulator with config: {simulator.config.application_type.value}")
    
    # Validate all profiles
    loader = BehaviorConfigLoader()
    print("\nProfile validation:")
    for profile in loader.get_available_profiles():
        is_valid = loader.validate_profile(profile)
        print(f"  {profile}: {'✓' if is_valid else '✗'}")