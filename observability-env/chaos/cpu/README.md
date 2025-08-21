# CPU Stress Injection Module

This module provides CPU stress injection capabilities for chaos engineering experiments using stress-ng with comprehensive monitoring and safety mechanisms.

## Features

- **Configurable Intensity**: Support for percentage-based (e.g., "80%") or worker-count based CPU stress
- **Safety Mechanisms**: Automatic shutdown when CPU usage exceeds safety thresholds
- **Real-time Monitoring**: Continuous monitoring of CPU and memory usage during injection
- **Container Support**: Can target specific Docker containers or run on host
- **Logging**: Comprehensive logging of injection events and monitoring data
- **Multiple Interfaces**: Python API, shell script wrapper, and Docker container

## Files

- `cpu_stress.py` - Main Python implementation with monitoring and safety features
- `inject_cpu_stress.sh` - Shell script wrapper for easy command-line usage
- `cpu_config.json` - Default configuration template
- `Dockerfile` - Docker image for containerized stress injection
- `README.md` - This documentation file

## Configuration

The CPU stress injection is configured via JSON files. Here's the default configuration:

```json
{
  "intensity": "80%",           // CPU intensity: percentage or worker count
  "duration": 300,              // Duration in seconds
  "safety_threshold": 95,       // CPU % threshold for emergency stop
  "monitoring_interval": 1,     // Monitoring frequency in seconds
  "ramp_up_time": 0,           // Gradual ramp-up time (future feature)
  "method": "all",             // stress-ng CPU method
  "load_percent": 100,         // CPU load percentage per worker
  "target_container": null     // Target container name (null for host)
}
```

### Configuration Parameters

- **intensity**: CPU stress intensity
  - Percentage format: "50%", "80%", "100%" 
  - Worker count: 1, 2, 4 (number of stress-ng workers)
- **duration**: How long to run stress injection (seconds)
- **safety_threshold**: CPU usage percentage that triggers emergency stop
- **monitoring_interval**: How often to check CPU usage (seconds)
- **method**: stress-ng CPU stress method ("all", "ackermann", "bitops", etc.)
- **load_percent**: CPU load percentage per worker (1-100)
- **target_container**: Docker container name to inject stress into

## Usage

### Shell Script Interface (Recommended)

```bash
# Start CPU stress with default configuration
./inject_cpu_stress.sh start

# Start with custom parameters
./inject_cpu_stress.sh -i 50% -d 180 start

# Target specific container
./inject_cpu_stress.sh -t checkoutservice -i 80% start

# Check status
./inject_cpu_stress.sh status

# Stop injection
./inject_cpu_stress.sh stop

# View logs
./inject_cpu_stress.sh logs
```

### Python API

```python
from cpu_stress import CPUStressInjector

# Create injector with configuration
config = {
    "intensity": "80%",
    "duration": 300,
    "safety_threshold": 95,
    "target_container": "checkoutservice"
}

injector = CPUStressInjector(config)

# Start injection
if injector.start_injection():
    print("Stress injection started")
    
    # Monitor status
    while injector.get_status()['running']:
        status = injector.get_status()
        print(f"Running for {status['elapsed_seconds']}s")
        time.sleep(10)
    
    # Save monitoring data
    injector.save_injection_log("cpu_stress_log.json")
```

### Docker Container

```bash
# Build the Docker image
docker build -t cpu-stress-injector .

# Run stress injection in container
docker run --rm \
  --name cpu-stress \
  -v $(pwd)/cpu_config.json:/chaos/config.json \
  -v $(pwd)/logs:/chaos/logs \
  cpu-stress-injector \
  python3 /usr/local/bin/cpu_stress.py \
  --config /chaos/config.json \
  --action start \
  --log-file /chaos/logs/cpu_stress.json
```

## Safety Features

### Automatic Safety Shutdown
- Monitors CPU usage every second during injection
- Automatically stops injection if CPU usage exceeds safety threshold (default 95%)
- Prevents system lockup or unresponsive conditions

### Graceful Termination
- Uses SIGTERM for graceful process termination
- Falls back to SIGKILL if process doesn't respond within 10 seconds
- Cleans up monitoring threads and resources

### Resource Monitoring
- Tracks CPU and memory usage throughout injection
- Logs all monitoring data for post-experiment analysis
- Provides real-time status information

## Integration with Chaos Experiments

### Experiment Configuration
```json
{
  "experiment_name": "checkoutservice_cpu_001",
  "target_service": "checkoutservice",
  "fault_type": "cpu",
  "chaos_config": {
    "intensity": "80%",
    "duration": 300,
    "safety_threshold": 90
  }
}
```

### Automated Execution
```bash
# Part of larger experiment script
./inject_cpu_stress.sh -t checkoutservice -i 80% -d 300 start

# Wait for completion or monitor
./inject_cpu_stress.sh status

# Collect logs for analysis
./inject_cpu_stress.sh logs
```

## Monitoring Data Format

The injection creates detailed monitoring logs in JSON format:

```json
{
  "config": {
    "intensity": "80%",
    "duration": 300,
    "safety_threshold": 95
  },
  "start_time": "2024-01-15T10:30:00Z",
  "monitoring_data": [
    {
      "timestamp": "2024-01-15T10:30:01Z",
      "cpu_percent": 82.5,
      "memory_percent": 45.2,
      "active_stress": true
    }
  ]
}
```

## Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   chmod +x inject_cpu_stress.sh
   ```

2. **stress-ng Not Found**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install stress-ng
   
   # CentOS/RHEL
   sudo yum install stress-ng
   ```

3. **Python Dependencies**
   ```bash
   pip3 install psutil
   ```

4. **Docker Container Access**
   ```bash
   # Ensure container exists and is running
   docker ps | grep checkoutservice
   ```

### Debugging

Enable debug logging by modifying the Python script:
```python
logging.basicConfig(level=logging.DEBUG)
```

Check system resources:
```bash
# CPU usage
top -bn1 | grep "Cpu(s)"

# Memory usage  
free -h

# Running processes
ps aux | grep stress
```

## Requirements Satisfied

This CPU stress injection module satisfies the following requirements:

- **Requirement 3.1**: CPU stress injection (cpu) support
- **Requirement 3.7**: Records injection timestamp and provides monitoring
- **Requirement 6.2**: Configurable target services for chaos injection
- **Requirement 6.3**: Configurable chaos intensity levels
- **Requirement 6.5**: Configuration parameter validation

## Next Steps

After implementing CPU stress injection, the next modules to implement are:
1. Memory stress injection (memory/)
2. Disk I/O stress injection (disk/)
3. Network delay injection (network/)
4. Packet loss injection (network/)
5. Socket/connection failure injection (socket/)