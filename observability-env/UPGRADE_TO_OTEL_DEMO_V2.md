# OpenTelemetry Demo v2.0.0+ 升級指南

## 🚀 概述

本指南說明如何從舊版本的 OpenTelemetry Demo (v1.10.0) 升級到最新的 v2.0.0+ 版本，以獲得完整的微服務觀測性數據。

## 📋 主要變更

### 新增的微服務 (v2.0.0+)
- **Accounting Service** (Go) - 會計服務
- **Quote Service** (PHP) - 報價服務  
- **Frontend Proxy** (Envoy) - 前端代理
- **Feature Flag Service** (Elixir) - 功能標誌服務

### 架構改進
- 使用 **Valkey** 替代 Redis
- 新增 **Kafka** 用於事件流
- 新增 **PostgreSQL** 用於功能標誌存儲
- 改進的 **OpenTelemetry Collector** 配置
- 更好的服務間通信和追蹤

### 完整的微服務列表
1. **accountingservice** (Go) - 處理訂單會計
2. **adservice** (Java) - 廣告推薦服務
3. **cartservice** (C#) - 購物車服務
4. **checkoutservice** (Go) - 結帳服務
5. **currencyservice** (C++) - 貨幣轉換服務
6. **emailservice** (Ruby) - 郵件服務
7. **featureflagservice** (Elixir) - 功能標誌服務
8. **frontend** (JavaScript) - 前端服務
9. **loadgenerator** (Python) - 負載生成器
10. **paymentservice** (JavaScript) - 支付服務
11. **productcatalogservice** (Go) - 產品目錄服務
12. **quoteservice** (PHP) - 運費報價服務
13. **recommendationservice** (Python) - 推薦服務
14. **shippingservice** (Rust) - 運輸服務
15. **frontendproxy** (Envoy) - 前端代理

## 🔧 升級步驟

### 1. 停止舊版本服務
```bash
cd observability-env
docker-compose -f docker-compose.otel-demo.yml down --remove-orphans
```

### 2. 使用新版本配置
```bash
# 啟動 OpenTelemetry Demo v2.0.0+
./scripts/start-otel-demo-v2.sh
```

### 3. 驗證服務狀態
```bash
# 檢查所有服務是否運行
docker-compose -f docker-compose.otel-demo-v2.yml ps

# 檢查服務健康狀態
curl http://localhost:8080/  # Frontend
curl http://localhost:9090/-/healthy  # Prometheus
curl http://localhost:16686/  # Jaeger
curl http://localhost:13133/  # OpenTelemetry Collector
```

### 4. 使用新的數據收集器
```bash
cd collectors
python3 collect_otel_demo_v2_data.py --duration 10 --output otel_demo_v2_data
```

## 📊 數據收集改進

### 更豐富的指標數據
- **HTTP 指標**: 請求數、延遲、錯誤率
- **gRPC 指標**: RPC 調用統計
- **語言特定指標**: 
  - Go: goroutines, GC 統計
  - Java: JVM 記憶體、GC
  - C#: .NET 運行時指標
  - Node.js: 事件循環延遲
  - Python: 進程統計
  - Rust: 自定義指標

### 完整的追蹤數據
- **分散式追蹤**: 跨服務的完整請求路徑
- **服務依賴**: 自動發現服務間依賴關係
- **錯誤追蹤**: 詳細的錯誤信息和堆棧追蹤
- **性能分析**: 延遲分析和瓶頸識別

### 結構化日誌
- **統一格式**: 所有服務使用一致的日誌格式
- **上下文信息**: 包含 trace ID 和 span ID
- **多級別**: DEBUG, INFO, WARN, ERROR
- **結構化**: JSON 格式便於解析

## 🏗️ 新架構優勢

### 1. 多語言支持
- **15 個微服務** 使用 **8 種不同語言**
- 真實反映現代微服務架構的複雜性
- 每種語言的最佳實踐和觀測性模式

### 2. 真實的業務場景
- **電商平台**: 完整的購物流程
- **功能標誌**: 動態功能控制
- **事件驅動**: Kafka 事件流
- **緩存策略**: Valkey 緩存層

### 3. 觀測性最佳實踐
- **三大支柱**: Metrics, Logs, Traces 完整集成
- **自動儀表**: OpenTelemetry 自動儀表化
- **採樣策略**: 智能採樣減少開銷
- **關聯分析**: 跨數據源的關聯分析

## 📈 RCA 評估改進

### 更真實的故障場景
```bash
# CPU 壓力測試
./chaos/cpu/inject_cpu_stress.sh checkoutservice

# 記憶體壓力測試  
./chaos/memory/inject_memory_stress.sh cartservice

# 網路延遲注入
./chaos/network/inject_network_delay.sh frontend

# 磁碟 I/O 壓力
./chaos/disk/inject_disk_stress.sh productcatalogservice
```

### 豐富的數據集
- **15 個微服務** 的完整觀測性數據
- **多種故障模式** 的數據集
- **時間序列數據** 用於趨勢分析
- **關聯數據** 用於根因分析

## 🔍 故障排除

### 常見問題

1. **服務啟動失敗**
   ```bash
   # 檢查 Docker 資源
   docker system df
   docker system prune  # 清理未使用的資源
   ```

2. **端口衝突**
   ```bash
   # 檢查端口使用情況
   lsof -i :8080
   lsof -i :9090
   ```

3. **記憶體不足**
   ```bash
   # 增加 Docker 記憶體限制
   # Docker Desktop -> Settings -> Resources -> Memory
   ```

### 日誌檢查
```bash
# 檢查特定服務日誌
docker-compose -f docker-compose.otel-demo-v2.yml logs frontend
docker-compose -f docker-compose.otel-demo-v2.yml logs otelcol

# 檢查所有服務狀態
docker-compose -f docker-compose.otel-demo-v2.yml ps
```

## 📚 相關資源

- [OpenTelemetry Demo 官方文檔](https://opentelemetry.io/docs/demo/)
- [架構圖](https://opentelemetry.io/docs/demo/architecture/)
- [GitHub 倉庫](https://github.com/open-telemetry/opentelemetry-demo)
- [發布說明](https://github.com/open-telemetry/opentelemetry-demo/releases)

## 🎯 下一步

1. **啟動新版本**: 使用 `start-otel-demo-v2.sh`
2. **生成流量**: 訪問 http://localhost:8080 進行購物
3. **收集數據**: 使用新的數據收集器
4. **故障注入**: 使用 chaos engineering 工具
5. **RCA 評估**: 使用收集的數據進行根因分析

升級完成後，你將擁有一個功能完整、真實的微服務觀測性環境，可以生成高質量的 RE2 格式數據集用於 RCA 評估！