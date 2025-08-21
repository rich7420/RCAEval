# Task 6 數據收集系統完整使用指南

## 📋 概述

本指南提供了在 `./observability-env/scripts/start-demo.sh` 啟動的微服務（OpenTelemetry Demo 或 Online Boutique）上使用 Task 6 數據收集系統的完整流程。

## 🎯 支持的微服務應用

### OpenTelemetry Demo
- **訪問地址**: http://localhost:8080
- **負載生成器**: http://localhost:8087
- **服務數量**: 11 個微服務
- **特點**: 完整的電商演示應用，包含前端、後端、數據庫等服務

### Online Boutique (Google)
- **訪問地址**: http://localhost:8084
- **服務數量**: 10 個微服務
- **特點**: Google 開源的微服務演示應用

## 🚀 完整使用流程

### 方法一：使用自動化腳本（推薦）

#### 步驟 1: 運行完整自動化腳本
```bash
cd observability-env
./run_data_collection.sh
```

這個腳本會自動執行以下操作：
1. 檢查先決條件（Docker、Python、curl 等）
2. 啟動觀測性服務（Prometheus、Loki、Jaeger）
3. 讓您選擇並啟動微服務應用
4. 可選啟動流量生成器
5. 安裝 Python 依賴
6. 運行數據收集
7. 顯示結果和後續操作建議

#### 腳本選項說明
當運行腳本時，您會看到以下選擇：

**微服務應用選擇：**
- 選項 1: OpenTelemetry Demo（推薦，數據更豐富）
- 選項 2: Online Boutique
- 選項 3: 兩個應用都啟動（需要更多資源）

**流量生成選擇：**
- 選項 1: 啟動流量生成（推薦，產生更多觀測性數據）
- 選項 2: 跳過流量生成

**數據收集時間選擇：**
- 選項 1: 快速收集（5分鐘）
- 選項 2: 標準收集（10分鐘）
- 選項 3: 長時間收集（30分鐘）
- 選項 4: 自定義時間範圍

### 方法二：手動步驟執行

#### 步驟 1: 啟動觀測性服務
```bash
cd observability-env
docker-compose up -d
```

等待服務啟動（約 60 秒）：
```bash
# 檢查服務狀態
./check_status.sh
```

#### 步驟 2: 啟動微服務應用
選擇以下方式之一：

**啟動 OpenTelemetry Demo：**
```bash
./scripts/start-otel-demo.sh
```

**啟動 Online Boutique：**
```bash
./scripts/start-online-boutique.sh
```

**使用 start-demo.sh 交互式啟動：**
```bash
./scripts/start-demo.sh
# 選擇 1 (OpenTelemetry Demo) 或 2 (Online Boutique)
```

**啟動兩個應用：**
```bash
docker-compose -f docker-compose.yml -f docker-compose.otel-demo.yml -f docker-compose.online-boutique.yml up -d
```

#### 步驟 3: 等待應用完全啟動
```bash
# 等待 2-3 分鐘讓應用完全啟動
sleep 180

# 檢查應用狀態
curl -s http://localhost:8080 > /dev/null && echo "✅ OpenTelemetry Demo 運行中" || echo "❌ OpenTelemetry Demo 未運行"
curl -s http://localhost:8084 > /dev/null && echo "✅ Online Boutique 運行中" || echo "❌ Online Boutique 未運行"
```

#### 步驟 4: 生成流量（可選但推薦）
為了產生更豐富的觀測性數據，建議生成一些流量：

**手動訪問應用：**
```bash
# 訪問 OpenTelemetry Demo
curl http://localhost:8080
curl http://localhost:8080/api/products
curl http://localhost:8080/cart

# 訪問 Online Boutique
curl http://localhost:8084
curl http://localhost:8084/api/products
```

**使用 Locust 生成流量：**
```bash
cd traffic/locust

# 為 OpenTelemetry Demo 生成流量
python3 -m locust -f otel_demo_users.py --host=http://localhost:8080 --users=5 --spawn-rate=1 --headless --run-time=300s

# 為 Online Boutique 生成流量
python3 -m locust -f online_boutique_users.py --host=http://localhost:8084 --users=3 --spawn-rate=1 --headless --run-time=300s

cd ../..
```

#### 步驟 5: 安裝數據收集器依賴
```bash
cd collectors
pip3 install -r requirements.txt
cd ..
```

#### 步驟 6: 運行數據收集
有多種方式運行數據收集：

**方式 A: 使用專用的微服務數據收集腳本（推薦）**
```bash
cd collectors

# 基本收集（10分鐘數據）
python3 collect_microservices_data.py

# 帶流量生成的收集
python3 collect_microservices_data.py --traffic --traffic-duration 5 --duration 15

# 長時間收集
python3 collect_microservices_data.py --duration 30 --output long_term_data

# 自定義輸出目錄
python3 collect_microservices_data.py --duration 20 --output experiment_001 --traffic
```

**方式 B: 使用 Python 程式化收集**
```python
cd collectors
python3 -c "
from main_collector import ObservabilityDataCollector
from datetime import datetime, timedelta
from pathlib import Path

# 初始化收集器
collector = ObservabilityDataCollector()

# 設置時間範圍（收集最近 10 分鐘的數據）
end_time = datetime.now()
start_time = end_time - timedelta(minutes=10)

print(f'收集時間範圍: {start_time} - {end_time}')

# 收集數據
data = collector.collect_all_data(start_time, end_time)

# 顯示收集結果
for data_type, df in data.items():
    if not df.empty:
        print(f'{data_type}: {len(df)} 條記錄')
        if 'service' in df.columns:
            print(f'  服務: {list(df[\"service\"].unique())}')

# 生成分析
analysis = collector.generate_comprehensive_analysis(data)

# 導出數據
output_dir = Path('microservices_output')
exported_files = collector.export_data_for_re2(data, output_dir, 'microservices_experiment')

print('導出文件:')
for file_type, file_path in exported_files.items():
    print(f'  {file_type}: {file_path}')
"
```

**方式 C: 使用測試腳本**
```bash
# 快速測試
./test_task6.sh
# 選擇選項 1 (快速測試)

# 完整測試
./test_task6.sh
# 選擇選項 2 (完整測試)
```

## 📊 輸出文件說明

### CSV 數據文件（RE2 兼容格式）
數據收集完成後，會在指定的輸出目錄中生成以下文件：

```
microservices_data/
├── metrics.csv              # 指標數據
├── logs.csv                 # 日誌數據
├── logts.csv               # 日誌時間戳
├── traces.csv              # 追蹤數據
├── tracets_err.csv         # 錯誤追蹤數據
├── tracets_lat.csv         # 延遲追蹤數據
├── cluster_info.json       # RE2 格式集群信息
└── microservices_YYYYMMDD_HHMMSS_analysis.json  # 分析報告
```

### 文件內容說明

**metrics.csv**
```csv
timestamp,service,metric_name,value,labels
1640995200,frontend,cpu_usage_percent,45.2,"{\"container\":\"frontend\"}"
1640995215,backend,memory_usage_bytes,1048576,"{\"container\":\"backend\"}"
```

**logs.csv**
```csv
timestamp,service,level,message,labels
1640995200,frontend,INFO,"User request processed","{\"container\":\"frontend\"}"
1640995201,backend,ERROR,"Database connection failed","{\"container\":\"backend\"}"
```

**traces.csv**
```csv
trace_id,span_id,parent_span_id,service,operation,start_time,duration_ms,error
trace1,span1,,frontend,GET /,1640995200,150.5,false
trace1,span2,span1,backend,query_db,1640995201,45.2,false
```

**cluster_info.json**
```json
{
  "services": {
    "frontend": {
      "name": "frontend",
      "type": "frontend",
      "containers": ["frontend-container"],
      "endpoints": ["GET /", "POST /cart"],
      "health_status": "healthy"
    }
  },
  "containers": {
    "frontend-container": {
      "container_id": "frontend-container",
      "service": "frontend",
      "image": "otel-demo-frontend:latest"
    }
  },
  "log_templates": {
    "1": {
      "template": "User <*> logged in successfully",
      "service": "frontend",
      "count": 25
    }
  }
}
```

## 🔧 高級配置選項

### 自定義收集配置
創建配置文件 `custom_config.yaml`：

```yaml
# 服務連接配置
services:
  prometheus:
    url: "http://localhost:9090"
  loki:
    url: "http://localhost:3100"
  jaeger:
    url: "http://localhost:16686"

# 數據收集配置
collection:
  time_range:
    default_duration_minutes: 15
  services_filter: ["frontend", "backend", "database"]  # 只收集特定服務

# Metrics 配置
metrics:
  aggregate: true
  aggregation_window: "1min"
  sample: true
  sample_rate: 0.1
  custom_queries:
    error_rate: 'rate(http_requests_total{status=~"5.."}[1m])'

# Logs 配置
logs:
  extract_templates: true
  clustering:
    method: "drain"
    min_cluster_size: 3

# Traces 配置
traces:
  analyze_dependencies: true
  performance_analysis:
    enabled: true
    slow_trace_percentile: 0.95
```

使用配置文件：
```python
import yaml
from main_collector import ObservabilityDataCollector

# 載入配置
with open('custom_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 使用配置收集數據
collector = ObservabilityDataCollector()
data = collector.collect_all_data(start_time, end_time, config=config)
```

### 特定服務收集
```python
# 只收集特定服務的數據
services = ['frontend', 'backend', 'productcatalogservice']
data = collector.collect_all_data(start_time, end_time, services=services)
```

### 高級分析選項
```python
# 生成詳細分析
analysis = collector.generate_comprehensive_analysis(data)

# 訪問特定分析結果
metrics_analysis = analysis['metrics_analysis']
logs_analysis = analysis['logs_analysis']
traces_analysis = analysis['traces_analysis']
cross_analysis = analysis['cross_data_analysis']

# 服務健康分析
service_health = traces_analysis.get('service_health', {})
for service, health in service_health.items():
    print(f"{service}: 健康分數 {health['health_score']:.1f}")

# 瓶頸檢測
bottlenecks = traces_analysis.get('bottlenecks', [])
for bottleneck in bottlenecks[:3]:
    print(f"瓶頸服務: {bottleneck['service']}")
    print(f"原因: {', '.join(bottleneck['reasons'])}")
```

## 🔍 監控和驗證

### 實時監控
在數據收集過程中，您可以通過以下界面監控系統狀態：

**Grafana 儀表板**
```bash
# 訪問 Grafana
open http://localhost:3000
# 用戶名: admin, 密碼: admin
```

**Prometheus 查詢**
```bash
# 訪問 Prometheus
open http://localhost:9090

# 常用查詢
# 查看所有服務: up
# CPU 使用率: rate(container_cpu_usage_seconds_total[1m]) * 100
# 內存使用: container_memory_usage_bytes
# HTTP 請求率: rate(http_requests_total[1m])
```

**Jaeger 追蹤**
```bash
# 訪問 Jaeger
open http://localhost:16686

# 搜索追蹤
# 選擇服務: frontend, backend 等
# 設置時間範圍查看追蹤數據
```

### 數據驗證
```bash
cd collectors

# 驗證收集的數據
python3 -c "
import pandas as pd
from pathlib import Path

output_dir = Path('microservices_data')
if output_dir.exists():
    # 檢查 CSV 文件
    for csv_file in output_dir.glob('*.csv'):
        df = pd.read_csv(csv_file)
        print(f'{csv_file.name}: {len(df)} 行')
        if not df.empty:
            print(f'  列: {list(df.columns)}')
            if 'service' in df.columns:
                print(f'  服務: {list(df[\"service\"].unique())}')
        print()
"
```

## 🚨 故障排除

### 常見問題及解決方案

#### 1. 觀測性服務連接失敗
```bash
# 檢查服務狀態
docker-compose ps

# 重新啟動服務
docker-compose down
docker-compose up -d

# 等待服務啟動
sleep 60

# 檢查服務健康狀態
curl http://localhost:9090/api/v1/query?query=up
curl http://localhost:3100/ready
curl http://localhost:16686/api/services
```

#### 2. 微服務應用未啟動
```bash
# 檢查應用狀態
curl -I http://localhost:8080  # OpenTelemetry Demo
curl -I http://localhost:8084  # Online Boutique

# 查看容器日誌
docker-compose logs otel-demo-frontend
docker-compose logs online-boutique-frontend

# 重新啟動應用
./scripts/start-demo.sh
```

#### 3. 無數據收集
```bash
# 檢查數據源
curl "http://localhost:9090/api/v1/query?query=up"
curl "http://localhost:3100/loki/api/v1/query?query={job=~\".+\"}"
curl "http://localhost:16686/api/services"

# 生成測試流量
for i in {1..10}; do
    curl http://localhost:8080/ > /dev/null 2>&1
    curl http://localhost:8084/ > /dev/null 2>&1
    sleep 2
done

# 等待數據傳播
sleep 30

# 重新收集
python3 collect_microservices_data.py --duration 5
```

#### 4. Python 依賴問題
```bash
cd collectors

# 重新安裝依賴
pip3 install --upgrade pip
pip3 install -r requirements.txt

# 檢查關鍵依賴
python3 -c "import pandas, requests, numpy, sklearn; print('依賴正常')"
```

#### 5. 內存不足
```bash
# 檢查系統資源
docker stats

# 減少收集時間範圍
python3 collect_microservices_data.py --duration 5

# 使用採樣
python3 -c "
from main_collector import ObservabilityDataCollector
config = {
    'metrics': {'sample': True, 'sample_rate': 0.1},
    'logs': {'extract_templates': False}
}
collector = ObservabilityDataCollector()
# ... 使用 config
"
```

### 日誌檢查
```bash
# 檢查容器日誌
docker-compose logs prometheus
docker-compose logs loki
docker-compose logs jaeger
docker-compose logs otel-collector

# 檢查應用日誌
docker-compose logs otel-demo-frontend
docker-compose logs otel-demo-backend
```

## 📈 性能優化

### 大規模數據收集
```python
# 使用採樣減少數據量
config = {
    'metrics': {
        'sample': True,
        'sample_rate': 0.05,  # 5% 採樣
        'aggregate': True,
        'aggregation_window': '5min'
    },
    'logs': {
        'extract_templates': True,
        'clustering': {'method': 'drain', 'min_cluster_size': 5}
    }
}
```

### 並行處理
```python
# 啟用並行收集（默認已啟用）
collector = ObservabilityDataCollector()
# 內部使用 ThreadPoolExecutor 並行收集
```

### 資源監控
```bash
# 監控資源使用
watch -n 5 'docker stats --no-stream'

# 監控磁盤使用
df -h
du -sh collectors/*/
```

## 🎯 最佳實踐

### 1. 數據收集時機
- **冷啟動後等待**: 應用啟動後等待 2-3 分鐘
- **流量預熱**: 收集前生成一些流量
- **穩定期收集**: 在系統穩定運行時收集數據

### 2. 收集時間範圍
- **測試環境**: 5-10 分鐘足夠
- **演示用途**: 10-15 分鐘
- **分析研究**: 30-60 分鐘
- **生產模擬**: 數小時

### 3. 服務選擇
- **全量收集**: 收集所有服務（默認）
- **核心服務**: 只收集關鍵業務服務
- **問題服務**: 針對特定問題服務收集

### 4. 數據質量
- **多次收集**: 進行多次收集確保數據一致性
- **交叉驗證**: 使用不同時間段的數據進行驗證
- **異常檢測**: 檢查並處理異常數據點

## 📚 進階使用案例

### 案例 1: 性能基線建立
```bash
# 收集正常運行時的基線數據
python3 collect_microservices_data.py --duration 30 --output baseline_data --traffic

# 分析基線性能
python3 -c "
import pandas as pd
import json

# 載入分析報告
with open('baseline_data/microservices_*_analysis.json', 'r') as f:
    analysis = json.load(f)

# 提取關鍵指標
service_health = analysis['traces_analysis']['service_health']
for service, health in service_health.items():
    print(f'{service}: 平均延遲 {health[\"avg_duration_ms\"]:.1f}ms, 錯誤率 {health[\"error_rate\"]:.2%}')
"
```

### 案例 2: 故障注入實驗
```bash
# 1. 收集正常狀態數據
python3 collect_microservices_data.py --duration 10 --output normal_state

# 2. 注入故障（例如 CPU 壓力）
cd ../chaos/cpu
./inject_cpu_stress.sh otel-demo-backend 80 300  # 80% CPU 壓力 5 分鐘

# 3. 收集故障期間數據
cd ../../collectors
python3 collect_microservices_data.py --duration 10 --output fault_injection

# 4. 比較分析
python3 -c "
import pandas as pd

normal_metrics = pd.read_csv('normal_state/metrics.csv')
fault_metrics = pd.read_csv('fault_injection/metrics.csv')

# 比較 CPU 使用率
normal_cpu = normal_metrics[normal_metrics['metric_name'].str.contains('cpu')]['value'].mean()
fault_cpu = fault_metrics[fault_metrics['metric_name'].str.contains('cpu')]['value'].mean()

print(f'正常狀態 CPU: {normal_cpu:.1f}%')
print(f'故障注入 CPU: {fault_cpu:.1f}%')
print(f'CPU 增長: {((fault_cpu - normal_cpu) / normal_cpu * 100):.1f}%')
"
```

### 案例 3: 長期趨勢分析
```bash
# 定期收集數據
for i in {1..6}; do
    echo "收集第 $i 次數據..."
    python3 collect_microservices_data.py --duration 10 --output "trend_data_$i" --traffic
    sleep 600  # 等待 10 分鐘
done

# 分析趨勢
python3 -c "
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 合併所有數據
all_metrics = []
for i in range(1, 7):
    metrics_file = Path(f'trend_data_{i}/metrics.csv')
    if metrics_file.exists():
        df = pd.read_csv(metrics_file)
        df['collection_round'] = i
        all_metrics.append(df)

if all_metrics:
    combined = pd.concat(all_metrics, ignore_index=True)
    
    # 分析 CPU 趨勢
    cpu_data = combined[combined['metric_name'].str.contains('cpu')]
    cpu_trend = cpu_data.groupby('collection_round')['value'].mean()
    
    print('CPU 使用率趨勢:')
    for round_num, avg_cpu in cpu_trend.items():
        print(f'  第 {round_num} 次: {avg_cpu:.1f}%')
"
```

## 🔄 自動化和調度

### 定期數據收集
創建 cron 任務進行定期收集：

```bash
# 編輯 crontab
crontab -e

# 添加每小時收集一次的任務
0 * * * * cd /path/to/observability-env/collectors && python3 collect_microservices_data.py --duration 10 --output "hourly_$(date +\%Y\%m\%d_\%H)" --traffic > /tmp/data_collection.log 2>&1
```

### 批量實驗腳本
```bash
#!/bin/bash
# batch_experiments.sh

experiments=(
    "normal:10:normal_load"
    "high_traffic:15:high_load"
    "cpu_stress:20:cpu_fault"
    "memory_stress:20:memory_fault"
)

for experiment in "${experiments[@]}"; do
    IFS=':' read -r name duration output <<< "$experiment"
    
    echo "運行實驗: $name"
    
    # 根據實驗類型執行不同操作
    case $name in
        "cpu_stress")
            ../chaos/cpu/inject_cpu_stress.sh otel-demo-backend 70 $((duration * 60)) &
            ;;
        "memory_stress")
            ../chaos/memory/inject_memory_stress.sh otel-demo-backend 80 $((duration * 60)) &
            ;;
    esac
    
    # 收集數據
    python3 collect_microservices_data.py --duration $duration --output $output --traffic
    
    # 等待故障注入結束
    wait
    
    echo "實驗 $name 完成"
    sleep 300  # 等待 5 分鐘讓系統恢復
done
```

這個完整的指南涵蓋了 Task 6 數據收集系統的所有使用方式，從基本的自動化腳本到高級的自定義配置和批量實驗。用戶可以根據自己的需求選擇合適的方法來收集和分析微服務的觀測性數據。