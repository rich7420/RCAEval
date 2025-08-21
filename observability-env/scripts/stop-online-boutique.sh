#!/bin/bash

# Google Online Boutique Stop Script
# This script stops the observability stack and Online Boutique application

set -e

echo "🛑 Stopping Google Online Boutique Environment..."

# Navigate to the observability-env directory
cd "$(dirname "$0")/.."

# Stop Online Boutique application
echo "📦 Stopping Online Boutique application..."
docker-compose -f docker-compose.yml -f docker-compose.online-boutique.yml down

# Stop observability stack
echo "📦 Stopping observability stack..."
docker-compose down

echo ""
echo "✅ Google Online Boutique Environment stopped successfully!"
echo ""
echo "💡 To preserve data volumes, use: docker-compose down --volumes"
echo "🗑️  To remove everything including volumes: ./scripts/cleanup.sh"