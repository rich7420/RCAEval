# Real-Time OpenTelemetry Demo Validation System

This document describes the consolidated real-time validation system for OpenTelemetry Demo environments with automated chaos injection and monitoring.

## Overview

The Real-Time OpenTelemetry Demo Validation System consolidates all Task 8 test functionality into a single, comprehensive validation tool that:

- ✅ Automatically discovers OpenTelemetry Demo services
- ✅ Validates observability stack health (Prometheus, Loki, Jaeger, Grafana)
- ✅ Injects controlled chaos (CPU, memory, network) into application services only
- ✅ Monitors system behavior in real-time during chaos
- ✅ Validates data collection quality across metrics, logs, and traces
- ✅ Generates comprehensive validation reports

## Files

### Core Validation System
- **`real_time_otel_validation.py`** - Main validation system (single file, ~500 lines)
- **`test_real_time_validation.py`** - Test suite to verify the system works
- **`REAL_TIME_VALIDATION_README.md`** - This documentation

### Replaced Files
This system consolidates and replaces:
- `demo_real_experiment.py` - Basic experiment demo
- `run_automated_experiment.py` - Automated experiment runner  
- `debug_service_discovery.py` - Service discovery debugging
- Various scattered test files in `chaos/*/test_*.py`

## Prerequisites

1. **OpenTelemetry Demo Running**:
   ```bash
   docker-compose -f docker-compose.otel-demo.yml up -d
   ```

2. **Observability Stack Accessible**:
   - Prometheus: http://localhost:9090
   - Loki: http://localhost:3100
   - Jaeger: http://localhost:16686
   - Grafana: http://localhost:3000

3. **Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Test the System
```bash
# Run the test suite first
python test_real_time_validation.py
```

### 2. Run Basic Validation
```bash
# Run 60-second CPU stress validation
python real_time_otel_validation.py

# Or specify parameters
python real_time_otel_validation.py --chaos-type cpu --duration 60
```

### 3. Different Chaos Types
```bash
# CPU stress test
python real_time_otel_validation.py --chaos-type cpu --duration 60

# Memory stress test  
python real_time_otel_validation.py --chaos-type memory --duration 60

# Network delay test
python real_time_otel_validation.py --chaos-type network --duration 60
```

## Usage Examples

### Basic CPU Stress Test
```bash
python real_time_otel_validation.py --chaos-type cpu --duration 60
```

**What it does:**
- Discovers running OpenTelemetry Demo services
- Validates observability backends are accessible
- Injects 50% CPU stress into 2-3 application services
- Monitors metrics, logs, and traces in real-time for 60 seconds
- Validates data quality and generates a report

### Memory Stress Test
```bash
python real_time_otel_validation.py --chaos-type memory --duration 90
```

**What it does:**
- Injects memory stress (128MB allocation) into target services
- Monitors for 90 seconds with real-time data collection
- Detects anomalies and validates data completeness

### Network Delay Test
```bash
python real_time_otel_validation.py --chaos-type network --duration 45
```

**What it does:**
- Adds 100ms network delay to target services
- Monitors trace latency changes and service dependencies
- Validates distributed tracing data quality

## Output

### Console Output
```
🚀 Real-Time OpenTelemetry Demo Validation
============================================================
🔍 Validating OpenTelemetry Demo environment...
📦 Found 8 running containers
🎯 Found 5 OpenTelemetry Demo services: ['frontend', 'emailservice', ...]
✅ Prometheus accessible at http://localhost:9090
✅ Loki accessible at http://localhost:3100
✅ Jaeger accessible at http://localhost:16686
✅ Grafana accessible at http://localhost:3000
📊 Initializing data collectors...
✅ All data collectors initialized
🎯 Discovering target services...
✅ Discovered 3 available services: ['frontend', 'emailservice', 'paymentservice']
🎯 Will target services: ['frontend', 'emailservice']
🏁 Starting 60s validation with cpu chaos
💥 Injecting cpu chaos into ['frontend', 'emailservice'] for 60s
✅ cpu chaos injected into frontend
✅ cpu chaos injected into emailservice
🔥 Chaos active on 2 services
📊 Starting real-time monitoring for 60s
📊 Monitoring: 60s remaining | Metrics: 0 | Logs: 0 | Traces: 0
📊 Monitoring: 45s remaining | Metrics: 156 | Logs: 23 | Traces: 12
📊 Monitoring: 30s remaining | Metrics: 312 | Logs: 45 | Traces: 28
📊 Monitoring: 15s remaining | Metrics: 468 | Logs: 67 | Traces: 41
✅ Real-time monitoring complete
🧹 Cleaning up chaos injection...
✅ Chaos cleanup complete
🔍 Validating data quality...
✅ Data quality validation passed
```

### Generated Report
```
🚀 Real-Time OpenTelemetry Demo Validation Report
============================================================

📋 Experiment Details:
   • Chaos Type: cpu
   • Target Services: ['frontend', 'emailservice']
   • Duration: 60 seconds
   • Start Time: 2025-01-21 14:30:15
   • End Time: 2025-01-21 14:31:18

📊 Data Collection Summary:
   • Metrics Collections: 6
   • Total Metrics: 624
   • Total Logs: 89
   • Total Traces: 53
   • Anomalies Detected: 2

✅ Validation Results:
   • Overall Status: PASSED
   • Metrics Valid: ✅
   • Logs Valid: ✅
   • Traces Valid: ✅

🎯 Recommendations:
   • ✅ System is functioning correctly
   • ✅ Ready for production experiments
```

## Safety Features

### Service Targeting Safety
- **Application Services Only**: Only targets OpenTelemetry Demo application services
- **Infrastructure Protection**: Never targets observability infrastructure (Prometheus, Loki, Jaeger, Grafana)
- **Limited Scope**: Maximum 3 services targeted simultaneously
- **Health Checks**: Validates service health before targeting

### Chaos Injection Safety
- **Controlled Intensity**: Moderate stress levels (50% CPU, 128MB memory, 100ms delay)
- **Time Limits**: Automatic cleanup after specified duration
- **Graceful Cleanup**: Comprehensive cleanup on completion or interruption
- **Signal Handling**: Proper cleanup on Ctrl+C or termination

### Monitoring Safety
- **Non-Intrusive**: Read-only data collection
- **Error Handling**: Continues monitoring even if individual collections fail
- **Resource Limits**: Bounded memory usage and collection frequency

## Troubleshooting

### Common Issues

#### "No target services found"
```bash
# Check if OpenTelemetry Demo is running
docker ps | grep -E "(frontend|emailservice|paymentservice)"

# Start the demo if needed
docker-compose -f docker-compose.otel-demo.yml up -d
```

#### "Observability backends not accessible"
```bash
# Check if observability stack is running
curl -s http://localhost:9090/api/v1/query?query=up
curl -s http://localhost:3100/ready
curl -s http://localhost:16686/api/services
```

#### "Chaos injection failed"
- Some containers may not have `stress-ng` or `tc` tools installed
- The system will continue with available services
- Check container logs: `docker logs <service-name>`

#### "Data collection issues"
- Verify observability backends are collecting data
- Check network connectivity between services
- Ensure proper service discovery configuration

### Debug Mode
```bash
# Enable verbose logging
python real_time_otel_validation.py --verbose --chaos-type cpu --duration 30
```

### Manual Cleanup
If the system is interrupted and chaos remains active:
```bash
# Kill stress processes in all containers
docker ps --format "{{.Names}}" | xargs -I {} docker exec {} pkill -f stress-ng

# Remove network rules
docker ps --format "{{.Names}}" | xargs -I {} docker exec {} tc qdisc del dev eth0 root 2>/dev/null || true
```

## Integration with Task 8

This validation system uses the Task 8 configuration management system:

- **ServiceTargetingSystem**: For safe service discovery and targeting
- **Data Collectors**: Prometheus, Loki, and Jaeger clients from Task 6
- **Safety Rules**: Built-in safety checks from Task 8 configuration validation
- **Export System**: Compatible with Task 7 data export formats

## Next Steps

After successful validation:

1. **Run Production Experiments**: Use the full experiment system
   ```bash
   python run_automated_experiment.py
   ```

2. **Collect Research Data**: Use the data collection system
   ```bash
   python collectors/collect_otel_demo_v2_data.py
   ```

3. **Export to RE2 Format**: Use the export system for analysis
   ```bash
   python collectors/exporters/data_export_manager.py
   ```

## Summary

The Real-Time OpenTelemetry Demo Validation System provides:

- ✅ **Single Command**: One script to validate the entire system
- ✅ **Real-Time Monitoring**: Live feedback during chaos injection
- ✅ **Comprehensive Validation**: Metrics, logs, traces, and data quality
- ✅ **Safety First**: Multiple safety mechanisms and automatic cleanup
- ✅ **Production Ready**: Validates system readiness for research experiments

This consolidates all Task 8 test functionality into a single, reliable validation tool specifically designed for OpenTelemetry Demo environments.