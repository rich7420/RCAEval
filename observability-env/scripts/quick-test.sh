#!/bin/bash

# Quick Test Script
# This script performs a quick test to identify potential issues

set -e

echo "🔍 Quick Environment Test"
echo "========================"

# Navigate to the observability-env directory
cd "$(dirname "$0")/.."

echo ""
echo "1️⃣ Testing Docker..."
if docker info > /dev/null 2>&1; then
    echo "✅ Docker is running"
else
    echo "❌ Docker is not running"
    exit 1
fi

echo ""
echo "2️⃣ Testing Docker Compose..."
if command -v docker-compose > /dev/null 2>&1; then
    echo "✅ Docker Compose is available"
    docker-compose version
else
    echo "❌ Docker Compose not found"
    exit 1
fi

echo ""
echo "3️⃣ Testing configuration syntax..."
if docker-compose config > /dev/null 2>&1; then
    echo "✅ Main configuration is valid"
else
    echo "❌ Main configuration has errors"
    exit 1
fi

echo ""
echo "4️⃣ Testing image availability (sample)..."
test_images=(
    "otel/opentelemetry-collector-contrib:latest"
    "prom/prometheus:latest"
    "grafana/loki:latest"
    "grafana/grafana:latest"
    "jaegertracing/all-in-one:latest"
)

for image in "${test_images[@]}"; do
    echo -n "  Testing $image... "
    if docker pull "$image" > /dev/null 2>&1; then
        echo "✅"
    else
        echo "❌ (may need internet connection)"
    fi
done

echo ""
echo "5️⃣ Testing basic startup (observability stack only)..."
echo "Starting basic services..."

if docker-compose up -d prometheus loki jaeger grafana > /dev/null 2>&1; then
    echo "✅ Basic services started"
    
    echo "Waiting 10 seconds..."
    sleep 10
    
    echo "Checking service status..."
    docker-compose ps
    
    echo ""
    echo "Stopping test services..."
    docker-compose down > /dev/null 2>&1
    echo "✅ Test completed successfully"
else
    echo "❌ Failed to start basic services"
    docker-compose logs --tail=5
    exit 1
fi

echo ""
echo "🎉 Quick test passed! Environment is ready."
echo ""
echo "💡 You can now run:"
echo "  ./scripts/start-demo.sh"