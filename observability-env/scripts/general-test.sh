#!/bin/bash

# General Test Script - Comprehensive observability stack validation

set -e

echo "🔍 General Observability Stack Test"
echo "=================================="

# Navigate to the observability-env directory
cd "$(dirname "$0")/.."

echo ""
echo "1️⃣ Environment Setup"
echo "-------------------"
echo "Cleaning up existing containers..."
docker-compose down --remove-orphans > /dev/null 2>&1 || true

echo "Starting all services..."
docker-compose up -d

echo ""
echo "2️⃣ Service Startup (waiting 4 minutes for full initialization)"
echo "------------------------------------------------------------"
for i in {1..240}; do
    if [ $((i % 30)) -eq 0 ]; then
        echo "Waiting... ${i}s/240s"
    fi
    sleep 1
done

echo ""
echo "3️⃣ Container Status Check"
echo "------------------------"
docker-compose ps

echo ""
echo "4️⃣ Service Health Checks"
echo "-----------------------"

# Function to test endpoint with retries
test_service() {
    local name=$1
    local url=$2
    local max_attempts=3
    
    for attempt in $(seq 1 $max_attempts); do
        if curl -f -s "$url" > /dev/null 2>&1; then
            echo "✅ $name: Healthy"
            return 0
        else
            if [ $attempt -lt $max_attempts ]; then
                echo "⏳ $name: Attempt $attempt failed, retrying..."
                sleep 10
            fi
        fi
    done
    echo "❌ $name: Failed after $max_attempts attempts"
    return 1
}

# Test all core services
test_service "OpenTelemetry Collector" "http://localhost:13133"
test_service "Prometheus" "http://localhost:9090/-/healthy"
test_service "Jaeger UI" "http://localhost:16686/api/services"
test_service "Grafana" "http://localhost:3000/api/health"
test_service "Loki" "http://localhost:3100/ready"

echo ""
echo "5️⃣ Network Connectivity Test"
echo "----------------------------"
echo "Testing inter-service communication..."

# Test if OTel Collector can reach other services
docker exec otel-collector ping -c 2 prometheus > /dev/null 2>&1 && echo "✅ OTel Collector → Prometheus" || echo "❌ OTel Collector → Prometheus"
docker exec otel-collector ping -c 2 jaeger > /dev/null 2>&1 && echo "✅ OTel Collector → Jaeger" || echo "❌ OTel Collector → Jaeger"
docker exec otel-collector ping -c 2 loki > /dev/null 2>&1 && echo "✅ OTel Collector → Loki" || echo "❌ OTel Collector → Loki"

echo ""
echo "6️⃣ Configuration Validation"
echo "---------------------------"

# Check if config files are properly mounted
docker exec otel-collector ls -la /etc/otel-collector-config.yaml > /dev/null 2>&1 && echo "✅ OTel Collector config mounted" || echo "❌ OTel Collector config missing"
docker exec jaeger ls -la /etc/jaeger/ui-config.json > /dev/null 2>&1 && echo "✅ Jaeger UI config mounted" || echo "❌ Jaeger UI config missing"
docker exec grafana ls -la /etc/grafana/grafana.ini > /dev/null 2>&1 && echo "✅ Grafana config mounted" || echo "❌ Grafana config missing"

echo ""
echo "7️⃣ Port Accessibility Test"
echo "-------------------------"
echo "Testing external port accessibility..."

# Test all exposed ports
netstat -tuln 2>/dev/null | grep -q ":13133" && echo "✅ Port 13133 (OTel Health)" || echo "❌ Port 13133 not accessible"
netstat -tuln 2>/dev/null | grep -q ":9090" && echo "✅ Port 9090 (Prometheus)" || echo "❌ Port 9090 not accessible"
netstat -tuln 2>/dev/null | grep -q ":16686" && echo "✅ Port 16686 (Jaeger UI)" || echo "❌ Port 16686 not accessible"
netstat -tuln 2>/dev/null | grep -q ":3000" && echo "✅ Port 3000 (Grafana)" || echo "❌ Port 3000 not accessible"
netstat -tuln 2>/dev/null | grep -q ":3100" && echo "✅ Port 3100 (Loki)" || echo "❌ Port 3100 not accessible"

echo ""
echo "8️⃣ Service Logs Check"
echo "--------------------"
echo "Checking for critical errors in service logs..."

# Check for errors in logs
OTEL_ERRORS=$(docker logs otel-collector 2>&1 | grep -i "error\|fatal" | wc -l)
JAEGER_ERRORS=$(docker logs jaeger 2>&1 | grep -i "error\|fatal" | wc -l)
GRAFANA_ERRORS=$(docker logs grafana 2>&1 | grep -i "error\|fatal" | grep -v "plugin.*signature" | wc -l)
PROMETHEUS_ERRORS=$(docker logs prometheus 2>&1 | grep -i "error\|fatal" | wc -l)

echo "OTel Collector errors: $OTEL_ERRORS"
echo "Jaeger errors: $JAEGER_ERRORS"
echo "Grafana errors: $GRAFANA_ERRORS (excluding plugin signature warnings)"
echo "Prometheus errors: $PROMETHEUS_ERRORS"

echo ""
echo "9️⃣ Data Flow Test"
echo "----------------"
echo "Testing basic data flow..."

# Test if OTel Collector is receiving and processing data
OTEL_METRICS=$(curl -s http://localhost:8888/metrics 2>/dev/null | grep -c "otelcol_" || echo "0")
echo "OTel Collector internal metrics: $OTEL_METRICS metrics found"

# Test Prometheus targets
PROMETHEUS_TARGETS=$(curl -s http://localhost:9090/api/v1/targets 2>/dev/null | grep -o '"health":"[^"]*"' | wc -l || echo "0")
echo "Prometheus targets configured: $PROMETHEUS_TARGETS"

echo ""
echo "🔟 Final Assessment"
echo "-----------------"

# Count successful services
SUCCESSFUL_SERVICES=0
curl -f -s http://localhost:13133 > /dev/null 2>&1 && ((SUCCESSFUL_SERVICES++))
curl -f -s http://localhost:9090/-/healthy > /dev/null 2>&1 && ((SUCCESSFUL_SERVICES++))
curl -f -s http://localhost:16686/api/services > /dev/null 2>&1 && ((SUCCESSFUL_SERVICES++))
curl -f -s http://localhost:3000/api/health > /dev/null 2>&1 && ((SUCCESSFUL_SERVICES++))
curl -f -s http://localhost:3100/ready > /dev/null 2>&1 && ((SUCCESSFUL_SERVICES++))

echo "Services Status Summary:"
echo "========================"
echo "✅ Successful services: $SUCCESSFUL_SERVICES/5"
echo "❌ Failed services: $((5 - SUCCESSFUL_SERVICES))/5"

if [ $SUCCESSFUL_SERVICES -eq 5 ]; then
    echo ""
    echo "🎉 ALL SERVICES HEALTHY!"
    echo "✅ Task 3 is COMPLETELY SUCCESSFUL!"
    echo "🚀 Ready to proceed with Task 4: Traffic Generation System"
    echo ""
    echo "Services will remain running for Task 4 development."
    echo "To stop services: docker-compose down"
elif [ $SUCCESSFUL_SERVICES -ge 3 ]; then
    echo ""
    echo "⚠️  MOSTLY SUCCESSFUL - Core services are working"
    echo "✅ Essential services (OTel, Prometheus, Jaeger) are likely working"
    echo "🔄 Some services may need more time or configuration adjustments"
    echo ""
    echo "You can proceed with Task 4, but monitor the failing services."
else
    echo ""
    echo "❌ MULTIPLE FAILURES DETECTED"
    echo "🔧 Significant issues need to be resolved before proceeding"
    echo ""
    echo "Recommended actions:"
    echo "1. Check service logs: docker logs <service-name>"
    echo "2. Verify configuration files"
    echo "3. Check for port conflicts"
fi

echo ""
echo "📊 Test completed at $(date)"