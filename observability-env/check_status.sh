#!/bin/bash

# Task 6 系統狀態檢查腳本
# 快速檢查觀測性服務和微服務的運行狀態

echo "🔍 Task 6 系統狀態檢查"
echo "======================="

# 檢查 Docker 服務
echo ""
echo "📦 Docker 容器狀態:"
if command -v docker-compose &> /dev/null; then
    docker-compose ps
else
    echo "❌ Docker Compose 未安裝"
fi

# 檢查觀測性服務
echo ""
echo "🔧 觀測性服務狀態:"

check_service() {
    local name=$1
    local url=$2
    local timeout=${3:-5}
    
    if curl -s --max-time $timeout "$url" > /dev/null 2>&1; then
        echo "✅ $name: 運行中"
    else
        echo "❌ $name: 無法連接"
    fi
}

check_service "Prometheus" "http://localhost:9090/api/v1/query?query=up"
check_service "Loki" "http://localhost:3100/ready"
check_service "Jaeger" "http://localhost:16686/api/services"

# 檢查微服務應用
echo ""
echo "🚀 微服務應用狀態:"
check_service "OpenTelemetry Demo" "http://localhost:8080" 3
check_service "Online Boutique" "http://localhost:8084" 3

# 檢查 Grafana
echo ""
echo "📊 監控界面:"
check_service "Grafana" "http://localhost:3000" 3

# 檢查數據收集器
echo ""
echo "🧪 數據收集器狀態:"
if [ -f "collectors/requirements.txt" ]; then
    echo "✅ 收集器代碼: 已部署"
    
    # 檢查 Python 依賴
    if python3 -c "import pandas, requests, numpy" 2>/dev/null; then
        echo "✅ Python 依賴: 已安裝"
    else
        echo "⚠️ Python 依賴: 需要安裝"
        echo "   運行: cd collectors && pip install -r requirements.txt"
    fi
else
    echo "❌ 收集器代碼: 未找到"
fi

# 檢查測試文件
echo ""
echo "🧪 測試文件狀態:"
test_files=(
    "test_task6.sh"
    "validate_task6.py"
    "collectors/test_data_collection.py"
    "collectors/quick_test.py"
    "collectors/demo_usage.py"
)

for file in "${test_files[@]}"; do
    if [ -f "$file" ]; then
        if [ -x "$file" ]; then
            echo "✅ $file: 可執行"
        else
            echo "⚠️ $file: 存在但不可執行"
        fi
    else
        echo "❌ $file: 不存在"
    fi
done

# 檢查輸出目錄
echo ""
echo "📁 輸出目錄狀態:"
output_dirs=(
    "collectors/test_output"
    "collectors/demo_output"
)

for dir in "${output_dirs[@]}"; do
    if [ -d "$dir" ]; then
        file_count=$(find "$dir" -type f | wc -l)
        echo "✅ $dir: 存在 ($file_count 個文件)"
    else
        echo "⚠️ $dir: 不存在 (將在測試時創建)"
    fi
done

# 提供建議
echo ""
echo "💡 建議操作:"

# 檢查是否有服務未運行
services_down=false
if ! curl -s --max-time 3 "http://localhost:9090/api/v1/query?query=up" > /dev/null 2>&1; then
    services_down=true
fi

if [ "$services_down" = true ]; then
    echo "🚀 啟動觀測性服務:"
    echo "   ./scripts/start-demo.sh"
    echo ""
fi

# 檢查是否有應用運行
apps_running=false
if curl -s --max-time 3 "http://localhost:8080" > /dev/null 2>&1 || \
   curl -s --max-time 3 "http://localhost:8084" > /dev/null 2>&1; then
    apps_running=true
fi

if [ "$apps_running" = true ]; then
    echo "🧪 運行測試:"
    echo "   ./test_task6.sh"
    echo ""
    echo "🎯 運行演示:"
    echo "   cd collectors && python3 demo_usage.py"
else
    echo "⏳ 等待應用啟動後運行測試"
fi

echo ""
echo "📖 查看完整使用指南:"
echo "   cat TASK6_USAGE_GUIDE.md"

echo ""
echo "======================="
echo "✅ 狀態檢查完成"