# User Behavior Simulator

A comprehensive user behavior simulation system for generating realistic e-commerce traffic patterns with configurable behavior distributions and user journey flows.

## Overview

The User Behavior Simulator implements realistic e-commerce user behavior patterns with the following traffic distribution:

- **70% Browse behavior**: Homepage visits, product browsing, category exploration, search
- **20% Cart operations**: Adding items to cart, viewing cart, modifying quantities
- **8% Checkout process**: Complete purchase flow with realistic completion rates
- **2% Error scenarios**: Invalid requests, timeouts, malformed data for testing

## Features

### Configurable Behavior Patterns
- Multiple behavior profiles (default, mobile, high-value, error-testing, etc.)
- Realistic traffic distribution matching e-commerce patterns
- Configurable timing and user journey parameters
- Support for both OTel Demo and Online Boutique applications

### Realistic User Journeys
- Intelligent behavior selection based on user context
- Realistic think times between actions (1-10 seconds)
- Session-based user progression and learning
- Cart abandonment and checkout completion simulation

### Integration Support
- Locust integration for load testing
- Standalone simulation runner for testing and validation
- Comprehensive statistics and reporting
- Export capabilities for analysis

## Quick Start

### 1. Basic Usage

```python
from user_behavior_simulator import UserBehaviorSimulator, BehaviorConfig, ApplicationType

# Create default simulator
config = BehaviorConfig(application_type=ApplicationType.OTEL_DEMO)
simulator = UserBehaviorSimulator(config)

# Simulate a user session
journey = simulator.simulate_user_session(duration_seconds=300)
print(f"Session completed: {len(journey.behaviors)} behaviors")
```

### 2. Using Configuration Profiles

```python
from config_loader import load_simulator_from_config

# Load simulator with mobile behavior profile
simulator = load_simulator_from_config("mobile")

# Simulate user session
journey = simulator.simulate_user_session()
```

### 3. Standalone Simulation

```bash
# Run simulation with 50 users for 10 minutes
python run_behavior_simulation.py --users 50 --duration 600 --profile default

# List available profiles
python run_behavior_simulation.py --list-profiles

# Run single user test
python run_behavior_simulation.py --single-user --duration 120
```

### 4. Locust Integration

```python
# Use in locustfile.py
from locust_integration import BehaviorDrivenUser

class WebsiteUser(BehaviorDrivenUser):
    pass

# Set behavior profile via environment variable
# BEHAVIOR_PROFILE=mobile locust -f locustfile.py
```

## Configuration

### Behavior Profiles

The system includes several pre-configured behavior profiles:

#### Default Profile
- Standard e-commerce behavior (70/20/8/2 distribution)
- 1-10 second think times
- 60-600 second sessions
- 60% checkout completion rate

#### Mobile Profile
- Higher browsing rate (80%)
- Lower checkout completion (3%)
- Faster interactions (0.5-5s think time)
- Shorter sessions (30-300s)

#### High-Value Profile
- More focused shopping (50% browse)
- Higher checkout rate (18%)
- Longer deliberation (2-15s think time)
- Extended sessions (120-900s)

#### Error Testing Profile
- High error rate (30%)
- Quick error generation (0.1-2s think time)
- Focused on testing error handling

### Custom Configuration

Create custom behavior patterns in `behavior_config.yaml`:

```yaml
custom_profile:
  traffic_distribution:
    browse_percentage: 65.0
    cart_percentage: 25.0
    checkout_percentage: 8.0
    error_percentage: 2.0
  
  timing:
    min_think_time: 2.0
    max_think_time: 8.0
    session_duration_min: 120.0
    session_duration_max: 480.0
  
  journey:
    max_products_per_session: 12
    max_cart_items: 6
    checkout_completion_rate: 0.7
  
  application:
    type: "otel_demo"
    base_url: "http://localhost:8080"
```

## Behavior Types

### Browse Behavior (70%)
- Homepage visits
- Category browsing
- Product detail viewing
- Search functionality
- Realistic product discovery patterns

### Cart Behavior (20%)
- Add items to cart
- View cart contents
- Modify item quantities
- Remove items from cart
- Cart abandonment simulation

### Checkout Behavior (8%)
- Start checkout process
- Enter shipping information
- Payment processing
- Checkout completion/abandonment
- Realistic form-filling timing

### Error Behavior (2%)
- Invalid product access (404 errors)
- Malformed requests (400 errors)
- Timeout scenarios
- Invalid cart operations
- Checkout validation errors

## Application Support

### OpenTelemetry Demo
- Product catalog API (`/api/products`)
- Cart management API (`/api/cart`)
- Checkout API (`/api/checkout`)
- Recommendations API (`/api/recommendations`)

### Online Boutique
- Web frontend interactions
- Product browsing (`/product/{id}`)
- Cart operations (`/cart`)
- Checkout process (`/cart/checkout`)
- Search functionality

## Statistics and Reporting

### Session Statistics
```python
stats = simulator.get_session_statistics()
print(f"Total sessions: {stats['total_sessions']}")
print(f"Behavior distribution: {stats['behavior_distribution']}")
print(f"Checkout completion rate: {stats['checkout_completion_rate']}")
```

### Journey Export
```python
# Export detailed journey data
simulator.export_journey_data("user_journeys.json")

# Export from standalone runner
python run_behavior_simulation.py --export-journeys journeys.json
```

## Advanced Usage

### Factory Patterns

```python
from user_behavior_simulator import BehaviorSimulatorFactory, ApplicationType

# Create specialized simulators
mobile_sim = BehaviorSimulatorFactory.create_mobile_simulator(ApplicationType.ONLINE_BOUTIQUE)
high_value_sim = BehaviorSimulatorFactory.create_high_value_simulator(ApplicationType.OTEL_DEMO)
error_sim = BehaviorSimulatorFactory.create_error_focused_simulator()
```

### Concurrent Simulation

```python
from run_behavior_simulation import BehaviorSimulationRunner

runner = BehaviorSimulationRunner("default")
journeys = runner.run_concurrent_simulation(
    num_users=100,
    duration=600,
    spawn_rate=2.0
)

report = runner.generate_report("simulation_report.json")
```

### Custom Behavior Implementation

```python
class CustomBehaviorSimulator(UserBehaviorSimulator):
    def execute_custom_behavior(self, journey):
        # Implement custom behavior logic
        pass
    
    def select_next_behavior(self, journey):
        # Override behavior selection logic
        return super().select_next_behavior(journey)
```

## Integration Examples

### With Locust Load Testing

```python
# locustfile.py
import os
from locust_integration import BehaviorDrivenUser, MobileBehaviorUser, HighValueBehaviorUser

# Set behavior profile via environment
# BEHAVIOR_PROFILE=mobile locust -f locustfile.py --users 100 --spawn-rate 10

class DefaultUser(BehaviorDrivenUser):
    weight = 70

class MobileUser(MobileBehaviorUser):
    weight = 20

class HighValueUser(HighValueBehaviorUser):
    weight = 10
```

### With Monitoring and Observability

```python
# Monitor behavior patterns in real-time
def monitor_behavior_patterns(simulator):
    while True:
        stats = simulator.get_session_statistics()
        
        # Check if behavior distribution matches targets
        actual_browse = stats['behavior_distribution'].get('browse', 0)
        target_browse = simulator.config.browse_percentage
        
        if abs(actual_browse - target_browse) > 5:  # 5% tolerance
            logger.warning(f"Browse behavior deviation: {actual_browse}% vs {target_browse}%")
        
        time.sleep(30)  # Check every 30 seconds
```

## Validation and Testing

### Profile Validation

```python
from config_loader import BehaviorConfigLoader

loader = BehaviorConfigLoader()
for profile in loader.get_available_profiles():
    is_valid = loader.validate_profile(profile)
    print(f"{profile}: {'✓' if is_valid else '✗'}")
```

### Behavior Distribution Testing

```bash
# Test behavior distribution accuracy
python run_behavior_simulation.py --users 1000 --duration 300 --profile default --output test_report.json

# Analyze results
python -c "
import json
with open('test_report.json') as f:
    report = json.load(f)
    
actual = report['behavior_distribution']['actual']
target = report['behavior_distribution']['target']

for behavior in ['browse', 'cart', 'checkout', 'error']:
    diff = abs(actual.get(behavior, 0) - target.get(behavior, 0))
    print(f'{behavior}: {diff:.1f}% deviation')
"
```

## Performance Considerations

- **Memory Usage**: Each active session uses ~1KB memory
- **CPU Usage**: Minimal CPU overhead per user
- **Scalability**: Tested with 1000+ concurrent users
- **Thread Safety**: All operations are thread-safe

## Troubleshooting

### Common Issues

1. **Configuration not found**: Ensure `behavior_config.yaml` is in the correct directory
2. **Profile validation fails**: Check traffic distribution sums to 100%
3. **Import errors**: Ensure all dependencies are installed
4. **Thread timeouts**: Increase duration for large simulations

### Debug Mode

```bash
# Enable verbose logging
python run_behavior_simulation.py --verbose --single-user

# Check configuration
python config_loader.py
```

## Requirements

- Python 3.7+
- PyYAML
- Locust (for integration)
- Standard library modules (threading, json, logging, etc.)

## License

This module is part of the observability data collection environment and follows the same licensing terms.