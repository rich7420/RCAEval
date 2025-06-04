# GNN-KAN RCA 方法實現計劃

## 專案概述
在 RCAEval 專案中實現一個新的根因分析方法，使用 KAN (Kolmogorov-Arnold Networks) 取代 GNN 中的 MLP 層。

## 專案結構分析
```
RCAEval/
├── e2e/                    # 主要的 RCA 方法實現
│   ├── causalrca.py       # 使用 VAE + GNN 的方法
│   ├── run.py             # 使用 DLinear + 圖注意力的方法
│   └── gnn_kan.py         # 我們要實現的新方法
├── classes/
│   ├── data.py            # 數據處理類
│   └── graph.py           # 圖處理類
├── io/
│   └── time_series.py     # 時間序列處理工具
├── benchmark/
│   ├── evaluation.py      # 評估指標
│   └── metrics.py         # 度量計算
└── utility/
    └── visualization.py   # 可視化工具
```

## 目標功能
1. **特徵提取模組**：
   - Sliding window 對齊 timestamp
   - TF-IDF 向量化提取 Log 特徵 (可選 DLA)
   - STL 分解 metrics 成向量
   - 特徵處理 (KLL algorithm)
   - 拓樸特徵計算
   - 錯誤特徵提取
   - 融合特徵

2. **GNN-KAN 模組**：
   - 使用 KAN 取代 GNN 中的 MLP 層
   - 取代傳統的 σ (激活函數) 與矩陣 W
   - GPU 加速計算

3. **根因分析**：
   - 圖構建
   - 因果推理
   - 排序輸出

## 實現步驟

### 第一步：建立基礎架構
1. 建立 `gnn_kan.py` 主方法檔案
2. 建立 KAN 網路層實現
3. 建立特徵提取模組

### 第二步：特徵提取實現
1. **時間窗口對齊**：
   ```python
   def sliding_window_alignment(data, window_size, step_size):
       # 實現滑動窗口對齊邏輯
   ```

2. **TF-IDF 日誌特徵**：
   ```python
   def extract_log_features(log_data, use_dla=False):
       # TF-IDF 向量化 + 可選 DLA
   ```

3. **STL 分解**：
   ```python
   def stl_decomposition(metrics_data):
       # STL 分解趨勢、季節性、殘差
   ```

4. **KLL 算法特徵處理**：
   ```python
   def kll_feature_processing(features):
       # KLL 算法處理特徵
   ```

5. **拓樸特徵**：
   ```python
   def compute_topology_features(graph):
       # 計算圖的拓樸特徵
   ```

### 第三步：KAN 網路實現
```python
class KANLayer(nn.Module):
    """KAN 層實現，取代傳統 MLP"""
    def __init__(self, input_dim, output_dim, grid_size=5):
        # 初始化 B-spline 基函數
        # 取代傳統的 W 矩陣和 σ 激活函數
    
    def forward(self, x):
        # KAN 前向傳播
        # 使用可學習的 spline 函數

class GNNKANEncoder(nn.Module):
    """使用 KAN 的 GNN 編碼器"""
    def __init__(self, feature_dim, hidden_dim, output_dim):
        # 替換傳統 MLP 為 KAN 層
    
    def forward(self, node_features, edge_index):
        # GNN + KAN 前向傳播
```

### 第四步：主方法實現
```python
def gnn_kan_rca(data, inject_time=None, dataset=None, with_bg=False, **kwargs):
    """
    主要的 GNN-KAN RCA 方法
    
    Args:
        data: 輸入數據 (multimodal 或 單一模態)
        inject_time: 注入時間點
        dataset: 數據集名稱
        with_bg: 是否包含背景數據
        **kwargs: 其他參數
    
    Returns:
        dict: 包含 adj, node_names, ranks 的結果
    """
```

### 第五步：集成到框架
1. 在 `RCAEval/e2e/__init__.py` 中註冊新方法
2. 修改 `main.py` 支持新方法
3. 添加配置參數

### 第六步：評估與比較
1. 使用現有的評估框架
2. 與 `causalrca.py` 等方法比較
3. 生成性能報告

---

## 技術細節

### KAN vs 傳統 MLP
- **傳統 MLP**: `y = σ(Wx + b)`
- **KAN**: `y = Σ φ(x_i)` (使用可學習的 spline 函數)

### GPU 優化
- 使用 PyTorch CUDA 支持
- 批次處理優化
- 內存管理

### 特徵融合策略
```python
def feature_fusion(log_features, metric_features, topology_features, error_features):
    """
    多模態特徵融合
    - 注意力機制
    - 特徵權重學習
    - 維度對齊
    """
```

## 執行與測試

### 訓練命令
```bash
python main.py --method gnn_kan --dataset online-boutique --data_path data/online-boutique/cartservice_mem/1/
```

### 評估命令
```bash
python -m RCAEval.benchmark.evaluation --method gnn_kan --baseline causalrca
```

### 比較腳本
```python
# compare_methods.py
methods = ['gnn_kan', 'causalrca', 'run', 'microcause']
results = benchmark_methods(methods, datasets, metrics=['precision', 'recall', 'f1'])
```

## 預期成果
1. 新的 GNN-KAN RCA 方法
2. 完整的特徵提取流水線
3. GPU 加速實現
4. 與現有方法的性能比較
5. 詳細的實驗報告

## 依賴項
- PyTorch (CUDA 支持)
- NetworkX
- scikit-learn
- pandas, numpy
- statsmodels (STL 分解)
- 自定義 KAN 實現庫

## 文件組織
```
/Users/user/RCAEval/
├── RCAEval/e2e/gnn_kan.py          # 主實現文件
├── RCAEval/kan/                     # KAN 相關模組
│   ├── __init__.py
│   ├── kan_layer.py                # KAN 層實現
│   └── feature_extraction.py      # 特徵提取模組
├── tests/test_gnn_kan.py           # 測試文件
├── experiments/                     # 實驗腳本
│   ├── run_gnn_kan.py
│   └── compare_methods.py
└── gnn_kan.md                      # 本計劃文件
```

# GNN-KAN 使用說明與測試指南

本文件說明如何在本專案中安裝、測試、執行 GNN-KAN 方法，並與其他 RCA 方法進行比較。

## 1. 安裝依賴

請先安裝 Python 3.10 及相關依賴：

```bash
python3.10 -m venv venv_gnn_kan
source venv_gnn_kan/bin/activate
pip install -r requirements.txt
```

如需 GPU 支援，請確保已安裝對應的 CUDA 版本與 PyTorch。

## 2. 測試 GNN-KAN 方法

可先用 smoke test 測試 pipeline 是否可跑通：

```bash
python main.py --method gnn_kan --dataset online-boutique --test
```

正式測試 RE2-TT 數據集：

```bash
python main.py --method gnn_kan --dataset re2-tt
```

執行結果會輸出於 `output/results/` 目錄。

## 3. 與其他方法比較

你可以將 `--method` 參數換成其他方法名稱，例如：

- pc_pagerank：
  ```bash
  python main.py --method pc_pagerank --dataset re2-tt
  ```
- tracerca：
  ```bash
  python main.py --method tracerca --dataset re2-tt
  ```
- causalrca：
  ```bash
  python main.py --method causalrca --dataset re2-tt
  ```

所有方法的結果都會輸出於 `output/results/`，可用於後續比較與評估。

## 4. 常見問題

- 若遇到 CUDA/torch 相關錯誤，請確認安裝的 PyTorch 版本與硬體環境相符。
- 若遇到 ImportError，請確認 Python 版本與 requirements.txt 依賴已正確安裝。
- 若要自訂參數，可於 main.py 增加對應 method 的 kwargs。

## 5. 參考

- GNN-KAN 實現：`RCAEval/e2e/gnn_kan.py`
- 其他方法實現：
  - `RCAEval/e2e/pc_pagerank.py`
  - `RCAEval/e2e/tracerca.py`
  - `RCAEval/e2e/causalrca.py`

如有問題，請參考 README.md 或聯絡專案維護者。

---

## 6. Docker 環境執行說明

### 6.1 單純 Dockerfile 執行

1. 在專案根目錄下建置映像：
   ```bash
   docker build -t rcaeval .
   ```
2. 啟動容器並進入 bash：
   ```bash
   docker run -it --rm -v $PWD:/app rcaeval
   ```
3. 進入容器後，啟動 GNN-KAN 或其他方法：
   ```bash
   python main.py --method gnn_kan --dataset re2-tt
   python main.py --method pc_pagerank --dataset re2-tt
   python main.py --method tracerca --dataset re2-tt
   python main.py --method causalrca --dataset re2-tt
   ```

### 6.2 使用 docker compose（推薦自動化/多服務）

1. 在專案根目錄下建立 `docker_gnnkan/` 資料夾，並複製 Dockerfile 至其中。
2. 建立 `docker_gnnkan/docker-compose.yml`，內容如下：
   ```yaml
   version: '3.8'
   services:
     gnnkan:
       build:
         context: ..
         dockerfile: docker_gnnkan/Dockerfile
       image: rcaeval-gnnkan:latest
       container_name: gnnkan
       working_dir: /app
       volumes:
         - ../:/app
       tty: true
   ```
3. 在 `docker_gnnkan/` 下執行：
   ```bash
   docker compose up -d
   docker exec -it gnnkan bash
   ```
4. 進入容器後，執行 GNN-KAN 或其他方法：
   ```bash
   python main.py --method gnn_kan --dataset re2-tt
   # 或比較
   python main.py --method pc_pagerank --dataset re2-tt
   python main.py --method tracerca --dataset re2-tt
   python main.py --method causalrca --dataset re2-tt
   ```

---

如遇到權限、依賴、路徑等問題，請確認 docker volume 掛載正確，且 requirements.txt、setup.py、資料集皆已同步到容器內。

如需 GPU 支援，請參考 [NVIDIA 官方說明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 並於 compose.yml 加入 `runtime: nvidia` 與相關設置。

## 7. 使用 GPU 執行 GNN-KAN

為了獲得最佳性能，GNN-KAN 可以使用 GPU 加速計算。以下是在 GPU 環境中執行 GNN-KAN 的詳細說明。

### 7.1 GPU 環境要求

- NVIDIA GPU 顯卡
- NVIDIA 驅動程序
- NVIDIA Container Toolkit (nvidia-docker2)
- Docker 和 Docker Compose

### 7.2 使用專用 GPU Docker 環境

我們提供了專門的 GPU Docker 環境，位於 `docker_gnnkan/` 目錄下：

1. **檢查 NVIDIA Docker 支持**：
   ```bash
   nvidia-smi
   ```
   確保您的系統已安裝 NVIDIA 驅動並可以正常識別 GPU。

2. **啟動 GPU 容器**：
   ```bash
   cd docker_gnnkan
   docker compose up -d
   ```

3. **進入容器**：
   ```bash
   docker exec -it rcaeval-gnnkan bash
   ```

4. **執行 GNN-KAN（使用 GPU）**：
   ```bash
   source env_gnnkan/bin/activate
   python main.py --method gnn_kan --dataset re2-tt
   ```

5. **驗證 GPU 使用情況**：
   ```bash
   # 在容器內執行
   python -c "import torch; print('GPU available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
   ```

### 7.3 比較不同方法（使用 GPU）

與標準環境相同，您可以通過切換 `--method` 參數來比較不同的方法：

```bash
# GNN-KAN (GPU 加速)
python main.py --method gnn_kan --dataset re2-tt

# 其他方法
python main.py --method pc_pagerank --dataset re2-tt
python main.py --method tracerca --dataset re2-tt
python main.py --method causalrca --dataset re2-tt
```

### 7.4 Jupyter Notebook 支持

我們還提供了帶有 Jupyter 的 GPU 環境：

1. **啟動 Jupyter 服務**：
   ```bash
   # 已經在 docker-compose.yml 中配置
   # 啟動後訪問 http://localhost:8889
   ```

2. **或者在主容器中啟動**：
   ```bash
   docker exec -it rcaeval-gnnkan bash
   source env_gnnkan/bin/activate
   jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
   ```

### 7.5 GPU Docker 環境說明

我們的 GPU Docker 環境基於 NVIDIA CUDA 12.1.1 和 cuDNN 8，並安裝了以下關鍵組件：

- PyTorch (CUDA 12.1 版本)
- PyTorch Geometric (GPU 支持)
- scikit-learn、pandas、networkx 等數據科學庫
- Jupyter 筆記本支持

主要配置文件：
- `docker_gnnkan/Dockerfile`: GPU 版本的 Docker 環境
- `docker_gnnkan/docker-compose.yml`: 啟動 GPU 容器的配置

### 7.6 GPU 加速效果比較

在大型數據集上，GPU 加速可以顯著提高 GNN-KAN 的性能：

| 數據集 | CPU 執行時間 | GPU 執行時間 | 加速比 |
|--------|-------------|-------------|-------|
| re2-tt | ~X分鐘     | ~Y秒        | ~Z倍  |

*注：實際性能取決於您的硬件配置*

### 7.7 GPU 相關故障排除

- **問題**：`RuntimeError: CUDA error: no kernel image is available for execution on the device`
  **解決方案**：檢查 PyTorch 版本與 CUDA 版本是否匹配

- **問題**：`ImportError: cannot import name xxx from torch_geometric`
  **解決方案**：確保已安裝正確版本的 PyG: `pip install torch-geometric -f https://data.pyg.org/whl/torch-<版本>+cu<CUDA版本>.html`

- **問題**：無法訪問 GPU
  **解決方案**：確保 docker-compose.yml 中的 `deploy.resources.reservations.devices` 設置正確