#!/bin/bash

# Task 6 數據收集系統完整使用腳本
# 在 start-demo.sh 啟動的微服務上運行數據收集

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${CYAN}================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}================================${NC}"
}

print_step() {
    echo -e "\n${BLUE}[步驟 $1]${NC} $2"
    echo -e "${BLUE}$(printf '%.0s-' {1..40})${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 檢查當前目錄
check_directory() {
    if [ ! -f "docker-compose.yml" ] || [ ! -d "scripts" ]; then
        print_error "請在 observability-env 目錄中運行此腳本"
        exit 1
    fi
}

# 檢查先決條件
check_prerequisites() {
    print_step 1 "檢查先決條件"
    
    # 檢查 Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker 未安裝"
        exit 1
    fi
    print_success "Docker 已安裝"
    
    # 檢查 Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose 未安裝"
        exit 1
    fi
    print_success "Docker Compose 已安裝"
    
    # 檢查 Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安裝"
        exit 1
    fi
    print_success "Python3 已安裝"
    
    # 檢查 curl
    if ! command -v curl &> /dev/null; then
        print_error "curl 未安裝"
        exit 1
    fi
    print_success "curl 已安裝"
}

# 啟動觀測性服務
start_observability_services() {
    print_step 2 "啟動觀測性服務"
    
    # 檢查服務是否已運行
    if curl -s --max-time 3 "http://localhost:9090/api/v1/query?query=up" > /dev/null 2>&1; then
        print_success "觀測性服務已在運行"
        return 0
    fi
    
    print_info "啟動觀測性堆棧..."
    docker-compose up -d
    
    print_info "等待觀測性服務啟動 (60秒)..."
    sleep 60
    
    # 驗證服務啟動
    services_ok=true
    
    if curl -s --max-time 10 "http://localhost:9090/api/v1/query?query=up" > /dev/null 2>&1; then
        print_success "Prometheus 啟動成功"
    else
        print_error "Prometheus 啟動失敗"
        services_ok=false
    fi
    
    if curl -s --max-time 10 "http://localhost:3100/ready" > /dev/null 2>&1; then
        print_success "Loki 啟動成功"
    else
        print_error "Loki 啟動失敗"
        services_ok=false
    fi
    
    if curl -s --max-time 10 "http://localhost:16686/api/services" > /dev/null 2>&1; then
        print_success "Jaeger 啟動成功"
    else
        print_error "Jaeger 啟動失敗"
        services_ok=false
    fi
    
    if [ "$services_ok" = false ]; then
        print_error "部分觀測性服務啟動失敗"
        print_info "檢查服務狀態:"
        docker-compose ps
        exit 1
    fi
}

# 選擇並啟動微服務應用
start_microservices() {
    print_step 3 "選擇並啟動微服務應用"
    
    # 檢查是否已有應用運行
    otel_running=false
    boutique_running=false
    
    if curl -s --max-time 3 "http://localhost:8080" > /dev/null 2>&1; then
        print_success "OpenTelemetry Demo 已在運行"
        otel_running=true
    fi
    
    if curl -s --max-time 3 "http://localhost:8084" > /dev/null 2>&1; then
        print_success "Online Boutique 已在運行"
        boutique_running=true
    fi
    
    if [ "$otel_running" = true ] || [ "$boutique_running" = true ]; then
        print_info "檢測到微服務應用已在運行，跳過啟動步驟"
        return 0
    fi
    
    # 選擇要啟動的應用
    echo ""
    echo "請選擇要啟動的微服務應用:"
    echo "1) OpenTelemetry Demo (推薦)"
    echo "2) Online Boutique"
    echo "3) 兩個應用都啟動"
    echo ""
    
    read -p "請輸入選擇 (1-3): " app_choice
    
    case $app_choice in
        1)
            print_info "啟動 OpenTelemetry Demo..."
            ./scripts/start-otel-demo.sh > /dev/null 2>&1 &
            
            print_info "等待 OpenTelemetry Demo 啟動 (120秒)..."
            sleep 120
            
            if curl -s --max-time 10 "http://localhost:8080" > /dev/null 2>&1; then
                print_success "OpenTelemetry Demo 啟動成功"
                print_info "應用訪問地址: http://localhost:8080"
                print_info "負載生成器: http://localhost:8087"
            else
                print_warning "OpenTelemetry Demo 可能需要更多時間啟動"
            fi
            ;;
        2)
            print_info "啟動 Online Boutique..."
            ./scripts/start-online-boutique.sh > /dev/null 2>&1 &
            
            print_info "等待 Online Boutique 啟動 (120秒)..."
            sleep 120
            
            if curl -s --max-time 10 "http://localhost:8084" > /dev/null 2>&1; then
                print_success "Online Boutique 啟動成功"
                print_info "應用訪問地址: http://localhost:8084"
            else
                print_warning "Online Boutique 可能需要更多時間啟動"
            fi
            ;;
        3)
            print_info "啟動兩個應用..."
            docker-compose -f docker-compose.yml -f docker-compose.otel-demo.yml -f docker-compose.online-boutique.yml up -d
            
            print_info "等待應用啟動 (180秒)..."
            sleep 180
            
            otel_ok=false
            boutique_ok=false
            
            if curl -s --max-time 10 "http://localhost:8080" > /dev/null 2>&1; then
                print_success "OpenTelemetry Demo 啟動成功"
                otel_ok=true
            fi
            
            if curl -s --max-time 10 "http://localhost:8084" > /dev/null 2>&1; then
                print_success "Online Boutique 啟動成功"
                boutique_ok=true
            fi
            
            if [ "$otel_ok" = true ] || [ "$boutique_ok" = true ]; then
                print_success "至少一個應用啟動成功"
                if [ "$otel_ok" = true ]; then
                    print_info "OpenTelemetry Demo: http://localhost:8080"
                fi
                if [ "$boutique_ok" = true ]; then
                    print_info "Online Boutique: http://localhost:8084"
                fi
            else
                print_warning "應用可能需要更多時間啟動"
            fi
            ;;
        *)
            print_error "無效選擇"
            exit 1
            ;;
    esac
}

# 啟動流量生成
start_traffic_generation() {
    print_step 4 "啟動流量生成"
    
    echo ""
    echo "是否要啟動流量生成器來產生更多觀測性數據？"
    echo "1) 是，啟動流量生成"
    echo "2) 否，跳過流量生成"
    echo ""
    
    read -p "請輸入選擇 (1-2): " traffic_choice
    
    if [ "$traffic_choice" = "1" ]; then
        print_info "啟動流量生成器..."
        
        # 檢查哪個應用在運行
        if curl -s --max-time 3 "http://localhost:8080" > /dev/null 2>&1; then
            print_info "為 OpenTelemetry Demo 啟動流量生成..."
            cd traffic/locust
            python3 -m locust -f otel_demo_users.py --host=http://localhost:8080 --users=5 --spawn-rate=1 --headless --run-time=300s > /dev/null 2>&1 &
            cd ../..
            print_success "OpenTelemetry Demo 流量生成已啟動 (5分鐘)"
        fi
        
        if curl -s --max-time 3 "http://localhost:8084" > /dev/null 2>&1; then
            print_info "為 Online Boutique 啟動流量生成..."
            cd traffic/locust
            python3 -m locust -f online_boutique_users.py --host=http://localhost:8084 --users=3 --spawn-rate=1 --headless --run-time=300s > /dev/null 2>&1 &
            cd ../..
            print_success "Online Boutique 流量生成已啟動 (5分鐘)"
        fi
        
        print_info "等待流量生成數據 (60秒)..."
        sleep 60
    else
        print_info "跳過流量生成，等待現有數據 (30秒)..."
        sleep 30
    fi
}

# 安裝數據收集器依賴
install_collector_dependencies() {
    print_step 5 "安裝數據收集器依賴"
    
    cd collectors
    
    if [ -f "requirements.txt" ]; then
        print_info "安裝 Python 依賴..."
        pip3 install -r requirements.txt --quiet
        print_success "依賴安裝完成"
    else
        print_error "requirements.txt 文件不存在"
        exit 1
    fi
    
    cd ..
}

# 運行數據收集
run_data_collection() {
    print_step 6 "運行數據收集"
    
    cd collectors
    
    echo ""
    echo "選擇數據收集模式:"
    echo "1) 快速收集 (5分鐘數據)"
    echo "2) 標準收集 (10分鐘數據)"
    echo "3) 長時間收集 (30分鐘數據)"
    echo "4) 自定義時間範圍"
    echo ""
    
    read -p "請輸入選擇 (1-4): " collection_choice
    
    case $collection_choice in
        1)
            duration_minutes=5
            ;;
        2)
            duration_minutes=10
            ;;
        3)
            duration_minutes=30
            ;;
        4)
            read -p "請輸入收集時間長度 (分鐘): " duration_minutes
            ;;
        *)
            print_error "無效選擇"
            exit 1
            ;;
    esac
    
    print_info "開始收集 ${duration_minutes} 分鐘的觀測性數據..."
    
    # 創建數據收集腳本
    cat > collect_data.py << EOF
#!/usr/bin/env python3
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# 添加 collectors 模組到路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_collector import ObservabilityDataCollector

def main():
    print("🚀 開始數據收集...")
    
    # 初始化收集器
    collector = ObservabilityDataCollector()
    
    # 設置時間範圍
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=${duration_minutes})
    
    print(f"⏰ 收集時間範圍: {start_time.strftime('%Y-%m-%d %H:%M:%S')} - {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 收集數據
    try:
        data = collector.collect_all_data(start_time, end_time)
        
        print("📊 收集結果:")
        for data_type, df in data.items():
            if not df.empty:
                print(f"  ✅ {data_type.capitalize()}: {len(df)} 條記錄")
            else:
                print(f"  ⚠️ {data_type.capitalize()}: 無數據")
        
        # 生成分析
        print("\n🔍 生成數據分析...")
        analysis = collector.generate_comprehensive_analysis(data)
        
        # 導出數據
        print("\n📤 導出數據...")
        output_dir = Path("production_output")
        output_dir.mkdir(exist_ok=True)
        
        experiment_name = f"microservices_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exported_files = collector.export_data_for_re2(data, output_dir, experiment_name)
        
        print("✅ 數據導出完成:")
        for file_type, file_path in exported_files.items():
            if Path(file_path).exists():
                file_size = Path(file_path).stat().st_size
                print(f"  📄 {file_type}: {file_path} ({file_size} bytes)")
        
        # 保存分析報告
        analysis_file = output_dir / f"{experiment_name}_analysis.json"
        import json
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        print(f"  📊 分析報告: {analysis_file}")
        
        print("\n🎉 數據收集完成！")
        return True
        
    except Exception as e:
        print(f"\n❌ 數據收集失敗: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF
    
    # 運行數據收集
    python3 collect_data.py
    collection_result=$?
    
    # 清理臨時腳本
    rm -f collect_data.py
    
    cd ..
    
    if [ $collection_result -eq 0 ]; then
        print_success "數據收集成功完成"
        return 0
    else
        print_error "數據收集失敗"
        return 1
    fi
}

# 顯示收集結果
show_results() {
    print_step 7 "顯示收集結果"
    
    output_dir="collectors/production_output"
    
    if [ -d "$output_dir" ]; then
        print_info "生成的文件:"
        find "$output_dir" -type f -name "*.csv" -o -name "*.json" | while read file; do
            size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "unknown")
            echo "  📄 $(basename "$file"): $size bytes"
        done
        
        print_info "文件位置: $output_dir"
    else
        print_warning "未找到輸出目錄"
    fi
}

# 提供後續操作建議
show_next_steps() {
    print_step 8 "後續操作建議"
    
    echo ""
    echo "🎯 可用的操作:"
    echo ""
    echo "📊 查看監控界面:"
    echo "  • Grafana:    http://localhost:3000 (admin/admin)"
    echo "  • Prometheus: http://localhost:9090"
    echo "  • Jaeger:     http://localhost:16686"
    echo ""
    
    echo "🔧 重新運行數據收集:"
    echo "  cd collectors && python3 -c \"from main_collector import *; ...\""
    echo ""
    
    echo "🧪 運行測試:"
    echo "  ./test_task6.sh"
    echo ""
    
    echo "🛑 停止服務:"
    echo "  docker-compose down"
    echo ""
    
    echo "🧹 清理所有數據:"
    echo "  docker-compose down -v"
}

# 主函數
main() {
    print_header "Task 6 數據收集系統 - 完整使用流程"
    
    print_info "此腳本將引導您完成以下步驟:"
    print_info "1. 檢查先決條件"
    print_info "2. 啟動觀測性服務"
    print_info "3. 選擇並啟動微服務應用"
    print_info "4. 啟動流量生成 (可選)"
    print_info "5. 安裝數據收集器依賴"
    print_info "6. 運行數據收集"
    print_info "7. 顯示收集結果"
    print_info "8. 提供後續操作建議"
    
    echo ""
    read -p "按 Enter 鍵開始，或 Ctrl+C 取消..."
    
    # 檢查目錄
    check_directory
    
    # 執行各個步驟
    check_prerequisites
    start_observability_services
    start_microservices
    start_traffic_generation
    install_collector_dependencies
    
    if run_data_collection; then
        show_results
        show_next_steps
        
        print_header "數據收集完成"
        print_success "Task 6 數據收集系統已成功運行！"
        print_info "所有觀測性數據已收集並導出為 RE2 兼容格式"
    else
        print_header "數據收集失敗"
        print_error "數據收集過程中出現錯誤"
        print_info "請檢查服務狀態和日誌"
    fi
}

# 運行主函數
main "$@"