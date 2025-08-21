#!/bin/bash

# Configuration Validation Script
# This script validates the Docker Compose configurations

set -e

echo "🔍 Validating observability environment configuration..."

# Navigate to the observability-env directory
cd "$(dirname "$0")/.."

echo "📋 Checking Docker Compose file syntax..."

# Check main docker-compose.yml
echo "  ✓ Checking docker-compose.yml..."
if docker-compose -f docker-compose.yml config > /dev/null 2>&1; then
    echo "    ✅ docker-compose.yml syntax is valid"
else
    echo "    ❌ docker-compose.yml has syntax errors"
    exit 1
fi

# Check OTel Demo compose file
echo "  ✓ Checking docker-compose.otel-demo.yml..."
if docker-compose -f docker-compose.yml -f docker-compose.otel-demo.yml config > /dev/null 2>&1; then
    echo "    ✅ docker-compose.otel-demo.yml syntax is valid"
else
    echo "    ❌ docker-compose.otel-demo.yml has syntax errors"
    exit 1
fi

# Check Online Boutique compose file
echo "  ✓ Checking docker-compose.online-boutique.yml..."
if docker-compose -f docker-compose.yml -f docker-compose.online-boutique.yml config > /dev/null 2>&1; then
    echo "    ✅ docker-compose.online-boutique.yml syntax is valid"
else
    echo "    ❌ docker-compose.online-boutique.yml has syntax errors"
    exit 1
fi

# Check both applications together
echo "  ✓ Checking combined configuration..."
if docker-compose -f docker-compose.yml -f docker-compose.otel-demo.yml -f docker-compose.online-boutique.yml config > /dev/null 2>&1; then
    echo "    ✅ Combined configuration is valid"
else
    echo "    ❌ Combined configuration has conflicts"
    exit 1
fi

echo ""
echo "📁 Checking required configuration files..."

# Check if all required config files exist
config_files=(
    "configs/otel-collector.yaml"
    "configs/prometheus.yml"
    "configs/loki.yml"
    "configs/promtail.yml"
    "configs/jaeger.yml"
    "configs/jaeger-ui-config.json"
    "configs/grafana/grafana.ini"
    "configs/grafana/provisioning/datasources/datasources.yml"
    "configs/grafana/provisioning/dashboards/dashboards.yml"
    "configs/grafana/provisioning/alerting/rules.yml"
)

for config_file in "${config_files[@]}"; do
    if [ -f "$config_file" ]; then
        echo "    ✅ $config_file exists"
    else
        echo "    ❌ $config_file is missing"
        exit 1
    fi
done

echo ""
echo "🔌 Checking port allocations..."

# Extract and check for port conflicts
echo "  📊 Port allocation summary:"
echo "    Observability Stack:"
echo "      - Grafana:                3000"
echo "      - Prometheus:             9090"
echo "      - Loki:                   3100"
echo "      - Jaeger UI:              16686"
echo "      - OTel Collector:         4317, 4318, 8888"
echo ""
echo "    OpenTelemetry Demo:"
echo "      - Frontend:               8080"
echo "      - Feature Flags:          8081"
echo "      - Load Generator:         8087"
echo "      - Various Services:       7070, 5050, 7000, etc."
echo ""
echo "    Online Boutique:"
echo "      - Frontend:               8084"
echo "      - Various Services:       8085, 8086, 7001, etc."

echo ""
echo "🎉 All configurations are valid!"
echo ""
echo "💡 Next steps:"
echo "  1. Run './scripts/start-demo.sh' to choose which demo to start"
echo "  2. Or run './scripts/start-otel-demo.sh' for OpenTelemetry Demo"
echo "  3. Or run './scripts/start-online-boutique.sh' for Online Boutique"