#!/bin/bash

# Demo Application Selector Script
# This script allows you to choose which demo application to start

set -e

echo "🚀 Observability Demo Environment Selector"
echo ""
echo "Please choose which demo application to start:"
echo ""
echo "1) OpenTelemetry Demo (recommended)"
echo "2) Google Online Boutique"
echo "3) Both applications (for comparison)"
echo "4) Observability stack only"
echo ""

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "🎯 Starting OpenTelemetry Demo..."
        cd "$(dirname "$0")/.."
        ./scripts/start-otel-demo.sh
        ;;
    2)
        echo "🛍️  Starting Google Online Boutique..."
        cd "$(dirname "$0")/.."
        ./scripts/start-online-boutique.sh
        ;;
    3)
        echo "🔄 Starting both applications..."
        cd "$(dirname "$0")/.."
        
        # Start observability stack
        docker-compose up -d
        echo "⏳ Waiting for observability stack..."
        sleep 30
        
        # Start both applications
        docker-compose -f docker-compose.yml -f docker-compose.otel-demo.yml -f docker-compose.online-boutique.yml up -d
        
        echo "⏳ Waiting for applications to start..."
        sleep 60
        
        echo ""
        echo "🎉 Both applications are ready!"
        echo ""
        echo "📱 Access URLs:"
        echo "  🌐 OpenTelemetry Demo:          http://localhost:8080"
        echo "  🚛 OTel Demo Load Generator:    http://localhost:8087"
        echo "  🛍️  Google Online Boutique:     http://localhost:8084"
        echo "  📊 Grafana:                     http://localhost:3000"
        echo "  📈 Prometheus:                  http://localhost:9090"
        echo "  🔍 Jaeger:                      http://localhost:16686"
        echo ""
        ;;
    4)
        echo "🔧 Starting observability stack only..."
        cd "$(dirname "$0")/.."
        docker-compose up -d
        
        echo "⏳ Waiting for services to start..."
        sleep 30
        
        echo ""
        echo "🎉 Observability stack is ready!"
        echo ""
        echo "📱 Access URLs:"
        echo "  📊 Grafana:                     http://localhost:3000"
        echo "  📈 Prometheus:                  http://localhost:9090"
        echo "  🔍 Jaeger:                      http://localhost:16686"
        echo "  📊 OTel Collector Metrics:      http://localhost:8888/metrics"
        echo ""
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again and choose 1-4."
        exit 1
        ;;
esac

echo ""
echo "💡 Use 'docker-compose ps' to check service status"
echo "🛑 Use './scripts/cleanup.sh' to stop and clean up everything"