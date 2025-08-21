#!/bin/bash

# OpenTelemetry Demo Startup Script
# This script starts the observability stack with the OTel Demo application

set -e

echo "🚀 Starting OpenTelemetry Demo Environment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Navigate to the observability-env directory
cd "$(dirname "$0")/.."

echo "📦 Starting observability stack..."

# Start the observability stack first with progress
if docker-compose up -d; then
    echo "✅ Observability stack started"
else
    echo "❌ Failed to start observability stack"
    exit 1
fi

echo "⏳ Waiting for observability stack to be ready (30s)..."
for i in {1..30}; do
    echo -n "."
    sleep 1
done
echo ""

echo "📦 Starting OpenTelemetry Demo application..."

# Start the OTel Demo application with progress
if docker-compose -f docker-compose.yml -f docker-compose.otel-demo.yml up -d; then
    echo "✅ OpenTelemetry Demo started"
else
    echo "❌ Failed to start OpenTelemetry Demo"
    echo "🔍 Checking for issues..."
    docker-compose -f docker-compose.yml -f docker-compose.otel-demo.yml logs --tail=10
    exit 1
fi

echo "⏳ Waiting for services to be ready (60s)..."
for i in {1..60}; do
    echo -n "."
    sleep 1
done
echo ""

# Check service status
echo "📊 Service Status:"
docker-compose -f docker-compose.yml -f docker-compose.otel-demo.yml ps

echo ""
echo "🎉 OpenTelemetry Demo Environment is ready!"
echo ""
echo "📱 Access URLs:"
echo "  🌐 Frontend (OTel Demo):     http://localhost:8080"
echo "  📊 Grafana:                  http://localhost:3000 (admin/admin)"
echo "  📈 Prometheus:               http://localhost:9090"
echo "  🔍 Jaeger:                   http://localhost:16686"
echo "  📝 Loki (via Grafana):       http://localhost:3000/explore"
echo "  🚛 Load Generator:           http://localhost:8087"
echo ""
echo "🔧 Monitoring:"
echo "  📊 OTel Collector Metrics:   http://localhost:8888/metrics"
echo "  🏥 OTel Collector Health:    http://localhost:13133"
echo ""
echo "💡 Tips:"
echo "  - The demo application includes automatic load generation"
echo "  - Check Grafana dashboards for observability data"
echo "  - Use Jaeger to explore distributed traces"
echo "  - Monitor logs in Grafana using Loki data source"
echo ""
echo "🛑 To stop: ./scripts/stop-otel-demo.sh"