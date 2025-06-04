# GNN-KAN GPU 執行指南

## 1. 快速啟動 (GPU 版本)

以下步驟將幫助你使用 GPU 加速執行 GNN-KAN 方法並與其他方法進行比較：

```bash
# 1. 進入 docker_gnnkan 目錄
cd /Users/user/RCAEval/docker_gnnkan

# 2. 啟動 GPU 支持的容器
docker compose up -d

# 3. 進入容器
docker exec -it rcaeval-gnnkan bash

# 4. 啟動 GNN-KAN 環境
source env_gnnkan/bin/activate

# 5. 確認 GPU 可用
python -c "import torch; print('GPU available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# 6. 執行 GNN-KAN 方法 (完整測試)
python main.py --method gnn_kan --dataset re2-tt
```

## 2. 與其他方法比較

在同一 GPU 容器內可以執行以下命令進行方法比較：

```bash
# PC-PageRank 方法
python main.py --method pc_pagerank --dataset re2-tt

# TraceRCA 方法
python main.py --method tracerca --dataset re2-tt

# CausalRCA 方法
python main.py --method causalrca --dataset re2-tt
```

## 3. 結果分析

結果將保存在 `/app/output/results/` 目錄下，可以使用以下命令查看：

```bash
# 查看結果文件
ls -la /app/output/results/

# 執行比較腳本 (如果已實現)
python experiments/compare_gnn_kan.py
```

## 4. 常見問題

- **GPU 未識別**: 確保主機已安裝 NVIDIA 驅動和 nvidia-docker2
- **內存不足**: 嘗試減小 batch size 或使用漸進式加載
- **結果不一致**: 設置隨機種子確保可重複性

## 5. 進階設置

如需修改 GPU 相關配置，請編輯 `docker_gnnkan/docker-compose.yml` 中的以下部分：

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1  # 可以調整使用的 GPU 數量
          capabilities: [gpu]
```