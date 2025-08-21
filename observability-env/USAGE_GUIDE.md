# 🚀 觀測性數據收集環境使用指南

## 📋 修正後的指令

### 1. 核心數據收集（最重要）
```bash
# 基本收集（5分鐘）
cd observability-env/collectors && python3 simple_re2_collector.py

# 自定義時長和輸出目錄
cd observability-env/collectors && python3 simple_re2_collector.py --duration 10 --output chaos_experiment_data

# 查看幫助
cd observability-env/collectors && python3 simple_re2_collector.py --help
```

### 2. Docker 環境管理
```bash
# 啟動 OpenTelemetry Demo
cd observability-env && bash scripts/start-otel-demo.sh

# 檢查服務狀態
cd observability-env && bash check_status.sh

# 檢查 Docker 容器
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### 3. 混沌工程測試
```bash
# 快速測試（從 observability-env 目錄）
cd observability-env/chaos && python3 quick_test.py

# 在背景運行混沌測試
cd observability-env/chaos && python3 quick_test.py &
```

### 4. 流量生成
```bash
# Locust 流量生成
cd observability-env/traffic/locust && bash start-locust.sh

# 用戶行為模擬
cd observability-env/traffic/behavior && python3 run_behavior_simulation.py
```

## 🔧 修正後的完整工作流程

### 標準數據收集流程：
```bash
# 1. 確保在正確目錄
cd observability-env

# 2. 啟動環境
bash scripts/start-otel-demo.sh

# 3. 等待 2-3 分鐘讓服務穩定
sleep 180

# 4. 收集數據
cd collectors && python3 simple_re2_collector.py --duration 5 --output production_data

# 5. 檢查結果
ls -la production_data/
```

### 混沌實驗 + 數據收集：
```bash
# 1. 確保在正確目錄
cd observability-env

# 2. 啟動環境
bash scripts/start-otel-demo.sh

# 3. 等待服務穩定
sleep 120

# 4. 在背景運行混沌實驗
cd chaos && python3 quick_test.py &

# 5. 回到主目錄並收集數據
cd .. && cd collectors && python3 simple_re2_collector.py --duration 10 --output chaos_experiment_data

# 6. 檢查結果
ls -la chaos_experiment_data/
```

## 📊 觀測性工具訪問

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Jaeger**: http://localhost:16686
- **Frontend**: http://localhost:8080

## 🔍 故障排除

### 常見問題：
1. **路徑錯誤**: 確保從 `observability-env` 目錄開始
2. **容器未啟動**: 運行 `docker ps` 檢查容器狀態
3. **依賴缺失**: 運行 `pip install -r requirements.txt`

### 檢查命令：
```bash
# 檢查當前目錄
pwd

# 檢查容器狀態
docker ps

# 檢查服務健康狀態
cd observability-env && bash check_status.sh
```

## 📁 目錄結構說明

```
observability-env/
├── collectors/          # 數據收集器
│   └── simple_re2_collector.py  # 主要收集工具
├── chaos/              # 混沌工程測試
├── traffic/            # 流量生成工具
├── scripts/            # 啟動腳本
└── configs/            # 配置文件
```

## ✅ 成功指標

數據收集成功後，你應該看到：
- `cluster_info.json` - 集群信息
- `metrics.csv` - 指標數據
- `logs.csv` - 日誌數據
- `traces.csv` - 追蹤數據
- `logts.csv` - 日誌時間序列
- `tracets_err.csv` - 錯誤追蹤時間序列
- `tracets_lat.csv` - 延遲追蹤時間序列