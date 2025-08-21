# Observability Data Collection Environment

A comprehensive observability data collection environment that integrates Prometheus, Loki, Jaeger, OpenTelemetry, and chaos engineering tools to generate structured datasets for root cause analysis research.

## Project Structure

```
data-collection/
├── README.md                    # This file
├── docker-compose.yml          # Main orchestration file
├── configs/                    # Configuration files for all services
│   ├── otel-collector.yaml     # OpenTelemetry Collector config
│   ├── prometheus.yml          # Prometheus configuration
│   ├── loki.yml               # Loki configuration
│   ├── promtail.yml           # Promtail configuration
│   ├── jaeger.yml             # Jaeger configuration
│   └── grafana/               # Grafana configurations and dashboards
├── scripts/                   # Automation scripts
│   ├── setup.sh              # Environment setup script
│   ├── run_experiment.sh      # Experiment execution script
│   ├── export_data.sh         # Data export script
│   └── cleanup.sh             # Cleanup script
├── data/                      # Collected datasets (organized by service_faulttype/experiment_number)
├── traffic/                   # Traffic generation configurations
│   ├── k6/                   # K6 load test scripts
│   ├── locust/               # Locust traffic generation
│   └── behavior/             # User behavior simulation
├── chaos/                     # Chaos engineering modules
│   ├── cpu/                  # CPU stress injection
│   ├── memory/               # Memory stress injection
│   ├── disk/                 # Disk I/O stress injection
│   ├── network/              # Network delay/loss injection
│   └── socket/               # Socket/connection failure injection
└── collectors/               # Data collection and processing modules
    ├── metrics/              # Metrics data collector
    ├── logs/                 # Logs data collector
    ├── traces/               # Traces data collector
    └── exporters/            # Data export modules
```

## Quick Start

1. **Setup Environment**: `./scripts/setup.sh`
2. **Run Experiment**: `./scripts/run_experiment.sh config/experiment.yaml`
3. **Export Data**: `./scripts/export_data.sh`
4. **Cleanup**: `./scripts/cleanup.sh`

## Requirements

- Docker and Docker Compose
- At least 8GB RAM
- 20GB free disk space

## Features

- Complete observability stack (Prometheus, Loki, Jaeger, Grafana)
- Demo applications (OpenTelemetry Demo, Online Boutique)
- Traffic generation (K6, Locust)
- Chaos engineering (CPU, Memory, Disk, Network faults)
- Automated data collection and export in RE2-compatible format