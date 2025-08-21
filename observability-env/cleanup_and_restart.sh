#!/bin/bash

echo "🧹 清理舊容器..."

# 停止並移除舊的容器
docker-compose -f docker-compose.yml -f docker-compose.otel-demo.yml down --remove-orphans

echo "🚀 重新啟動服務..."

# 重新啟動
./scripts/start-otel-demo.sh

echo "✅ 完成！"