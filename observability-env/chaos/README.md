# Chaos Engineering Module

This module provides comprehensive chaos engineering capabilities for the observability data collection environment. It implements various fault injection types to simulate real-world failure scenarios for root cause analysis research.

## Overview

The chaos engineering module supports the following fault injection types:

- **CPU Stress** (`cpu/`) - High CPU utilization injection
- **Memory Stress** (`memory/`) - Memory pressure and allocation stress
- **Disk I/O Stress** (`disk/`) - Disk I/O bottlenecks and storage stress
- **Network Delay** (`network/`) - Network latency injection
- **Packet Loss** (`network/`) - Network packet loss simulation
- **Socket/Connection Failures** (`socket/`) - Connection blocking and socket failures

## Current Status & Limitations

**✅ Completed (Task 5):**
- All chaos injection capabilities are implemented and tested
- Can inject faults into Docker containers and host system
- Comprehensive monitoring and logging of injection events
- Safety mechanisms and automatic cleanup

**⏳ Pending (Task 6):**
- Automated data collection system
- Integration with observability data streams
- Automated experiment orchestration
- RE2-compatible dataset generation

**Current Usage:**
- Manual chaos injection with observability through existing tools (Grafana, Prometheus, Jaeger)
- Chaos injection logs are saved locally for analysis
- Visual verification through monitoring dashboards

## Architecture

Each fault injection type follows a consistent architecture:

```
chaos/
├── cpu/                    # CPU stress injection
│   ├── cpu_stress.py      # Main implementation
│   ├── inject_cpu_stress.sh # Shell wrapper
│   ├── cpu_config.json    # Configuration template
│   ├── Dockerfile         # Container image
│   └── README.md          # Documentation
├── memory/                # Memory stress injection
├── disk/                  # Disk I/O stress injection
├── network/               # Network chaos (delay & loss)
├── socket/                # Socket/connection failures
└── README.md              # This file
```

## Common Features

All chaos injection modules provide:

- **Safety Mechanisms**: Automatic shutdown on dangerous conditions
- **Real-time Monitoring**: Continuous monitoring during injection
- **Configurable Parameters**: JSON-based configuration
- **Multiple Interfaces**: Python API, shell scripts, Docker containers
- **Comprehensive Logging**: Detailed injection logs for analysis
- **Container Support**: Can target specific Docker containers

## Quick Start

### Target Environment
This chaos engineering module is designed to work with:
- **Docker containers** (primary target - microservices from task 3)
- **Host system** (affects entire Docker environment)
- **Container networks** (Docker bridge networks)

### 1. CPU Stress Injection

```bash
# Target specific microservice container (recommended)
cd chaos/cpu
./inject_cpu_stress.sh -t checkoutservice -i 80% -d 300 start

# Host-level CPU stress (affects all containers)
./inject_cpu_stress.sh -i 50% -d 180 start

# Check status
./inject_cpu_stress.sh status

# Stop injection
./inject_cpu_stress.sh stop
```

### 2. Memory Stress Injection

```bash
# Target specific microservice container
cd chaos/memory
./inject_memory_stress.sh -t paymentservice -s 512M -d 300 start

# Host-level memory stress
./inject_memory_stress.sh -s 30% -w 4 start
```

### 3. Disk I/O Stress Injection

```bash
# Start disk stress with 2GB files
cd chaos/disk
./inject_disk_stress.sh -s 2G -d 300 start

# Use specific I/O method
./inject_disk_stress.sh -m sync -w 4 start
```

### 4. Network Delay Injection

```bash
# Add delay to specific microservice container
cd chaos/network
# Edit delay_config.json to set target_container: "checkoutservice"
python3 network_delay.py --config delay_config.json --action start

# Add delay to Docker bridge network (affects container communication)
# Edit delay_config.json to set interface: "docker0"
python3 network_delay.py --config delay_config.json --action start
```

### 5. Packet Loss Injection

```bash
# Inject 5% packet loss
cd chaos/network
python3 packet_loss.py --config loss_config.json --action start
```

### 6. Socket/Connection Failure Injection

```bash
# Block specific ports
cd chaos/socket
./inject_socket_failure.sh -p 80,443,8080 start

# Complete network failure
./inject_socket_failure.sh -f complete start
```

## Configuration

Each module uses JSON configuration files with common structure:

```json
{
  "duration": 300,              // Duration in seconds
  "monitoring_interval": 1,     // Monitoring frequency
  "target_container": null,     // Target container name
  "safety_threshold": 95,       // Safety threshold (varies by type)
  // Module-specific parameters...
}
```

### CPU Configuration (`cpu/cpu_config.json`)
```json
{
  "intensity": "80%",           // CPU intensity (% or worker count)
  "duration": 300,
  "safety_threshold": 95,       // CPU usage safety limit
  "method": "all",              // stress-ng CPU method
  "target_container": null
}
```

### Memory Configuration (`memory/memory_config.json`)
```json
{
  "size": "50%",                // Memory size (% or absolute)
  "duration": 300,
  "workers": 2,                 // Number of memory workers
  "safety_threshold": 90,       // Memory usage safety limit
  "target_container": null
}
```

### Disk Configuration (`disk/disk_config.json`)
```json
{
  "duration": 300,
  "workers": 2,                 // Number of I/O workers
  "file_size": "1G",           // Size of files to create
  "temp_path": "/tmp",         // Temporary directory
  "method": "sync",            // I/O method
  "target_container": null
}
```

### Network Delay Configuration (`network/delay_config.json`)
```json
{
  "delay": "100ms",            // Network delay
  "jitter": "10ms",            // Delay variation
  "duration": 300,
  "interface": "eth0",         // Network interface
  "direction": "both",         // ingress, egress, or both
  "target_container": null
}
```

### Packet Loss Configuration (`network/loss_config.json`)
```json
{
  "loss_percent": 5.0,         // Packet loss percentage
  "correlation": 25,           // Burst correlation
  "duration": 300,
  "pattern": "random",         // random, burst, or periodic
  "target_container": null
}
```

### Socket Failure Configuration (`socket/socket_config.json`)
```json
{
  "duration": 300,
  "failure_pattern": "selective", // complete, selective, intermittent
  "target_ports": [80, 443, 8080], // Ports to block
  "target_ips": [],            // IPs to block
  "protocols": ["tcp"],        // Protocols to affect
  "target_container": null
}
```

## Integration with Experiments

### Experiment Workflow

1. **Setup**: Deploy observability stack and demo applications
2. **Baseline**: Collect baseline metrics before chaos
3. **Chaos Injection**: Execute fault injection
4. **Monitoring**: Collect telemetry data during chaos
5. **Recovery**: Stop chaos and collect post-chaos data
6. **Analysis**: Export data in RE2-compatible format

### Automated Experiment Script

```bash
#!/bin/bash
# Example experiment automation

EXPERIMENT_NAME="checkoutservice_cpu_001"
SERVICE="checkoutservice"
FAULT_TYPE="cpu"
DURATION=300

# Start data collection
echo "Starting experiment: $EXPERIMENT_NAME"

# Inject chaos
cd chaos/cpu
./inject_cpu_stress.sh -t $SERVICE -i 80% -d $DURATION start

# Wait for completion
sleep $DURATION

# Export data
echo "Experiment completed. Exporting data..."
# Data export logic here...
```

## Safety Features

### Automatic Safety Mechanisms

1. **Resource Monitoring**: Continuous monitoring of system resources
2. **Safety Thresholds**: Automatic shutdown when limits exceeded
3. **Graceful Termination**: Proper cleanup of injected faults
4. **Emergency Cleanup**: Manual cleanup scripts for stuck processes

### Safety Thresholds

- **CPU**: 95% usage triggers automatic shutdown
- **Memory**: 90% usage triggers automatic shutdown  
- **Disk**: 90% usage or <1GB free triggers shutdown
- **Network**: No automatic limits (user-controlled)
- **Socket**: No automatic limits (user-controlled)

### Emergency Recovery

```bash
# Emergency stop all chaos injections
pkill -f "stress-ng"
pkill -f "_stress.py"

# Clean up network rules
tc qdisc del dev eth0 root 2>/dev/null || true
iptables -F 2>/dev/null || true

# Clean up temporary files
rm -f /tmp/stress-ng-*
```

## Monitoring and Logging

### Real-time Monitoring

Each module provides real-time monitoring of:
- Resource utilization (CPU, memory, disk, network)
- Injection status and parameters
- Safety threshold violations
- Error conditions and recovery actions

### Log Format

All modules generate JSON logs with consistent structure:

```json
{
  "config": { /* injection configuration */ },
  "start_time": "2024-01-15T10:30:00Z",
  "monitoring_data": [
    {
      "timestamp": "2024-01-15T10:30:01Z",
      "resource_metrics": { /* resource-specific metrics */ },
      "active_injection": true
    }
  ]
}
```

## Requirements Satisfied

This chaos engineering module satisfies the following requirements:

- **Requirement 3.1**: CPU stress injection (cpu)
- **Requirement 3.2**: Memory stress injection (mem)  
- **Requirement 3.3**: Disk I/O stress injection (disk)
- **Requirement 3.4**: Network delay injection (delay)
- **Requirement 3.5**: Network packet loss injection (loss)
- **Requirement 3.6**: Socket/connection failure injection (socket)
- **Requirement 3.7**: Records injection timestamp and provides monitoring
- **Requirement 6.2**: Configurable target services for chaos injection
- **Requirement 6.3**: Configurable chaos intensity levels
- **Requirement 6.5**: Configuration parameter validation

## Dependencies

### System Requirements

- **Linux**: Ubuntu 18.04+ or CentOS 7+
- **Python**: 3.7+
- **Docker**: 20.10+ (for container targeting)
- **Root/Sudo**: Required for network and socket chaos

### Python Packages

```bash
pip3 install psutil
```

### System Tools

```bash
# Ubuntu/Debian
sudo apt-get install stress-ng iproute2 iptables jq

# CentOS/RHEL
sudo yum install stress-ng iproute iptables jq
```

## Troubleshooting

### Common Issues

1. **Permission Denied**
   - Solution: Run with sudo or as root for network/socket chaos
   - Check: `sudo ./inject_*_stress.sh start`

2. **stress-ng Not Found**
   - Solution: Install stress-ng package
   - Check: `which stress-ng`

3. **Network Interface Not Found**
   - Solution: Update interface name in config
   - Check: `ip link show`

4. **Container Not Found**
   - Solution: Verify container is running
   - Check: `docker ps | grep container_name`

### Debug Mode

Enable debug logging in Python modules:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Checks

```bash
# Check system resources
top -bn1 | head -20
free -h
df -h
ss -tuln

# Check running chaos processes
ps aux | grep stress
ps aux | grep python3
```

## Contributing

When adding new chaos injection types:

1. Follow the established directory structure
2. Implement Python module with monitoring
3. Create shell script wrapper
4. Add configuration template
5. Include comprehensive documentation
6. Add safety mechanisms and thresholds
7. Implement proper cleanup procedures

## License

This chaos engineering module is part of the observability data collection environment and follows the same licensing terms as the parent project.