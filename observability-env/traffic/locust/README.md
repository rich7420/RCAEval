# Locust Traffic Generator

This directory contains Locust-based traffic generation for observability data collection. It provides realistic user behavior simulation with configurable patterns and distributed load generation capabilities.

## Features

- **Realistic User Behavior**: Simulates e-commerce user patterns (browsing, cart operations, checkout, errors)
- **Application-Specific Users**: Specialized user classes for OpenTelemetry Demo and Online Boutique
- **Distributed Load Generation**: Master-worker configuration for scalable load testing
- **Configurable Patterns**: JSON-based configuration for different load scenarios
- **Docker Support**: Containerized deployment with Docker Compose
- **Multiple Load Patterns**: Light, normal, heavy, and burst load patterns

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Basic Usage

```bash
# Start standalone Locust for OTel Demo
./start-locust.sh -a otel_demo -u 50 -r 5

# Start distributed Locust with 3 workers
./start-locust.sh -m distributed -w 3 -a online_boutique

# Run headless test for 10 minutes
./start-locust.sh --headless -t 10m -u 100
```

### 3. Docker Usage

```bash
# Start OTel Demo Locust cluster
./start-locust.sh -m docker -a otel_demo

# Start Online Boutique Locust cluster  
./start-locust.sh -m docker -a online_boutique
```

## Configuration

### Load Patterns

The `config.json` file defines different load patterns:

- **Light**: 10 users, slow spawn rate, longer think times
- **Normal**: 50 users, moderate spawn rate, normal think times  
- **Heavy**: 100 users, fast spawn rate, shorter think times
- **Burst**: 200 users with alternating high/low activity periods

### Traffic Distribution

Default traffic distribution follows realistic e-commerce patterns:

- **70%** - Product browsing and catalog viewing
- **20%** - Cart operations (add/remove items)
- **8%** - Checkout process completion
- **2%** - Error scenarios and edge cases

### User Behavior Scenarios

- **browse_only**: 100% browsing, no purchases
- **shopping_focused**: Higher cart and checkout rates
- **error_testing**: Increased error scenario generation

## User Classes

### OpenTelemetry Demo Users

- `OTelDemoUser`: Standard e-commerce behavior for OTel Demo API
- `OTelDemoHeavyUser`: High-intensity load testing
- `OTelDemoErrorUser`: Focused error generation

### Online Boutique Users

- `OnlineBoutiqueUser`: Standard web frontend behavior
- `OnlineBoutiqueMobileUser`: Mobile-optimized patterns
- `OnlineBoutiqueHighValueUser`: High-value customer simulation

## Distributed Testing

### Master-Worker Setup

1. **Start Master Node**:
   ```bash
   python distributed_runner.py --mode=master --application=otel_demo
   ```

2. **Start Worker Nodes**:
   ```bash
   python distributed_runner.py --mode=worker --application=otel_demo
   ```

3. **All-in-One Distributed**:
   ```bash
   python distributed_runner.py --mode=distributed --workers=3
   ```

### Docker Distributed Setup

The Docker Compose configuration provides:
- 1 Master node with web UI (port 8089)
- 2 Worker nodes for OTel Demo
- 1 Worker node for Online Boutique (profile: boutique)

## Monitoring and Metrics

### Web UI Access

- **Standalone/Master**: http://localhost:8089
- **Docker OTel Demo**: http://localhost:8089  
- **Docker Online Boutique**: http://localhost:8090

### Key Metrics

- **Request Rate**: Requests per second
- **Response Times**: P50, P95, P99 percentiles
- **Error Rate**: Failed requests percentage
- **User Count**: Active concurrent users

## Advanced Usage

### Custom User Classes

Create custom user classes by extending base classes:

```python
from locust import HttpUser, task, between

class CustomUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def custom_behavior(self):
        # Your custom behavior here
        pass
```

### Environment Variables

- `LOCUST_WORKER_ID`: Worker identification
- `LOCUST_TEST_ID`: Test session identification
- `TARGET_HOST`: Override target host URL

### Configuration Override

Override configuration programmatically:

```python
runner = DistributedLocustRunner("custom_config.json")
runner.start_distributed_test(
    application="otel_demo",
    num_workers=5,
    users=200,
    spawn_rate=20
)
```

## Troubleshooting

### Common Issues

1. **Connection Refused**: Ensure target application is running
2. **Import Errors**: Install requirements with `pip install -r requirements.txt`
3. **Port Conflicts**: Change web UI port with `-p` option
4. **Worker Connection**: Check master host/port configuration

### Debug Mode

Enable debug logging:

```bash
export LOCUST_LOGLEVEL=DEBUG
./start-locust.sh -a otel_demo
```

### Health Checks

Verify application endpoints before testing:

```bash
curl http://localhost:8080/api/products  # OTel Demo
curl http://localhost:80/               # Online Boutique
```

## Integration with Chaos Engineering

Locust traffic generation integrates with chaos engineering experiments:

1. **Pre-Chaos**: Generate baseline traffic
2. **During Chaos**: Maintain load during fault injection
3. **Post-Chaos**: Continue traffic for recovery observation

Example workflow:
```bash
# Start background traffic
./start-locust.sh --headless -t 30m -u 100 &

# Run chaos experiment
../chaos/cpu/inject-cpu-stress.sh

# Traffic continues automatically during and after chaos
```

## Performance Tuning

### Resource Optimization

- **CPU**: Use multiple workers for CPU-intensive scenarios
- **Memory**: Monitor memory usage with high user counts
- **Network**: Adjust spawn rates to avoid overwhelming targets

### Scaling Guidelines

- **Single Machine**: Up to 1000 users per machine
- **Distributed**: Add workers for higher user counts
- **Docker**: Use resource limits to prevent interference

## Files Overview

- `locustfile.py`: Base user classes and behaviors
- `otel_demo_users.py`: OpenTelemetry Demo specific users
- `online_boutique_users.py`: Online Boutique specific users
- `distributed_runner.py`: Distributed testing orchestration
- `config.json`: Load patterns and configuration
- `start-locust.sh`: Startup script with options
- `docker-compose.locust.yml`: Docker deployment configuration
- `Dockerfile`: Container image definition
- `requirements.txt`: Python dependencies