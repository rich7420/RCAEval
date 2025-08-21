#!/bin/bash

# Google Online Boutique Startup Script
# This script starts the observability stack with the Online Boutique application

set -e

echo "🛍️  Starting Google Online Boutique Environment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Navigate to the observability-env directory
cd "$(dirname "$0")/.."

echo "📦 Building and starting services..."

# Start the observability stack first
docker-compose up -d

echo "⏳ Waiting for observability stack to be ready..."
sleep 30

# Start the Online Boutique application
docker-compose -f docker-compose.yml -f docker-compose.online-boutique.yml up -d

echo "🔍 Checking service health..."

# Wait for services to be healthy
echo "⏳ Waiting for services to start (this may take a few minutes)..."
sleep 60

# Check service status
echo "📊 Service Status:"
docker-compose -f docker-compose.yml -f docker-compose.online-boutique.yml ps

echo ""
echo "🎉 Google Online Boutique Environment is ready!"
echo ""
echo "📱 Access URLs:"
echo "  🛍️  Frontend (Online Boutique): http://localhost:8084"
echo "  📊 Grafana:                     http://localhost:3000 (admin/admin)"
echo "  📈 Prometheus:                  http://localhost:9090"
echo "  🔍 Jaeger:                      http://localhost:16686"
echo "  📝 Loki (via Grafana):          http://localhost:3000/explore"
echo ""
echo "🔧 Service Endpoints:"
echo "  🛒 Product Catalog:             http://localhost:3551"
echo "  💰 Currency Service:            http://localhost:7001"
echo "  🛍️  Cart Service:               http://localhost:7071"
echo "  📦 Recommendation Service:      http://localhost:8085"
echo "  🚚 Shipping Service:            http://localhost:50053"
echo "  💳 Payment Service:             http://localhost:50054"
echo "  📧 Email Service:               http://localhost:8086"
echo "  📢 Ad Service:                  http://localhost:9556"
echo ""
echo "🔧 Monitoring:"
echo "  📊 OTel Collector Metrics:      http://localhost:8888/metrics"
echo "  🏥 OTel Collector Health:       http://localhost:13133"
echo ""
echo "💡 Tips:"
echo "  - Browse the online store to generate telemetry data"
echo "  - Check Grafana dashboards for observability data"
echo "  - Use Jaeger to explore distributed traces"
echo "  - Monitor logs in Grafana using Loki data source"
echo ""
echo "🛑 To stop: ./scripts/stop-online-boutique.sh"