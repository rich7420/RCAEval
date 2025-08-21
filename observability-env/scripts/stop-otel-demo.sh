#!/bin/bash

# OpenTelemetry Demo Stop Script
# This script stops the observability stack and OTel Demo application

set -e

echo "🛑 Stopping OpenTelemetry Demo Environment..."

# Navigate to the observability-env directory
cd "$(dirname "$0")/.."

# Stop OTel Demo application
echo "📦 Stopping OTel Demo application..."
docker-compose -f docker-compose.yml -f docker-compose.otel-demo.yml down

# Stop observability stack
echo "📦 Stopping observability stack..."
docker-compose down

echo ""
echo "✅ OpenTelemetry Demo Environment stopped successfully!"
echo ""
echo "💡 To preserve data volumes, use: docker-compose down --volumes"
echo "🗑️  To remove everything including volumes: ./scripts/cleanup.sh"