#!/bin/bash

# Final Verification for Task 3 Fixes
# Tests only the two specific issues that were fixed

set -e

echo "🎯 Task 3 最終驗證測試"
echo "======================"
echo "專注測試兩個已修正的問題："
echo "1. 健康檢查端點 (端口 13133)"
echo "2. Jaeger 連接問題"
echo ""

# Navigate to the observability-env directory
cd "$(dirname "$0")/.."

echo "1️⃣ 清理並啟動服務..."
docker-compose down --remove-orphans > /dev/null 2>&1 || true

# Start only essential services for testing
docker-compose up -d prometheus jaeger otel-collector

echo "等待 30 秒讓服務啟動..."
for i in {1..30}; do
    echo -n "."
    sleep 1
done
echo ""

echo ""
echo "2️⃣ 測試修正結果..."

# Test 1: Health check endpoint
echo ""
echo "🩺 測試 1: 健康檢查端點"
echo "------------------------"
if curl -f -s http://localhost:13133 > /dev/null 2>&1; then
    HEALTH_RESPONSE=$(curl -s http://localhost:13133)
    echo "✅ 成功: 健康檢查端點正常回應"
    echo "   端點: http://localhost:13133"
    echo "   回應: $HEALTH_RESPONSE"
    HEALTH_TEST="PASS"
else
    echo "❌ 失敗: 健康檢查端點無法訪問"
    HEALTH_TEST="FAIL"
fi

# Test 2: Jaeger connection (check logs for DNS errors)
echo ""
echo "🔗 測試 2: Jaeger 連接狀態"
echo "-------------------------"

# Wait a bit for logs to accumulate
sleep 5

# Check for Jaeger connection errors in logs
JAEGER_DNS_ERRORS=$(docker logs otel-collector 2>&1 | grep -i "lookup jaeger.*no such host" | wc -l)
COLLECTOR_RUNNING=$(docker ps --filter "name=otel-collector" --filter "status=running" | wc -l)

if [ "$JAEGER_DNS_ERRORS" -eq 0 ] && [ "$COLLECTOR_RUNNING" -gt 1 ]; then
    echo "✅ 成功: 沒有 Jaeger DNS 解析錯誤"
    echo "✅ 成功: OTel Collector 正常運行"
    JAEGER_TEST="PASS"
else
    echo "❌ 失敗: 發現 $JAEGER_DNS_ERRORS 個 Jaeger DNS 錯誤"
    echo "❌ 失敗: OTel Collector 運行狀態異常"
    JAEGER_TEST="FAIL"
fi

echo ""
echo "3️⃣ 服務狀態檢查..."
echo "當前運行的服務:"
docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "4️⃣ 清理測試環境..."
docker-compose down > /dev/null 2>&1

echo ""
echo "🏁 最終測試結果"
echo "==============="
echo "健康檢查端點測試: $HEALTH_TEST"
echo "Jaeger 連接測試:   $JAEGER_TEST"
echo ""

if [ "$HEALTH_TEST" = "PASS" ] && [ "$JAEGER_TEST" = "PASS" ]; then
    echo "🎉 所有測試通過！Task 3 修正成功完成！"
    echo ""
    echo "✅ 問題 1: 健康檢查端點 (端口 13133) - 已修正"
    echo "✅ 問題 2: Jaeger 連接 DNS 解析錯誤 - 已修正"
    echo ""
    echo "🚀 準備開始 Task 4: Implement traffic generation system"
    exit 0
else
    echo "❌ 部分測試失敗，需要進一步檢查"
    exit 1
fi