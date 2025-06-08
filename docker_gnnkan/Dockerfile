FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
#docker compose build --no-cache
# 使用 NVIDIA CUDA 11.8 基礎映像，包含 GPU 支援

# 設定環境變數
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Taipei
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# 設定 NumPy API 環境變數以解決 tigramite 編譯問題
ENV NPY_DISABLE_CPU_FEATURES="AVX512_SKX"
ENV CFLAGS="-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"

# 安裝系統依賴
RUN apt-get update -y && \
    apt-get install -y \
        build-essential \
        libxml2 \
        libxml2-dev \
        zlib1g-dev \
        python3-tk \
        graphviz \
        software-properties-common \
        git \
        wget \
        curl \
        openjdk-11-jre-headless \
        pkg-config \
        libcairo2-dev \
        libgirepository1.0-dev \
        vim \
        htop \
        # 新增編譯依賴以解決 tigramite 和 scikit-network 問題
        libblas-dev \
        liblapack-dev \
        libatlas-base-dev \
        gfortran \
        libgomp1 \
        && rm -rf /var/lib/apt/lists/*

# 安裝 Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3.10-distutils \
        && rm -rf /var/lib/apt/lists/*

# 安裝 pip
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# 設定 Python 3.10 為預設
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# 設定工作目錄
WORKDIR /app

# 複製專案檔案
COPY . .

# 建立 GNN-KAN 環境 (Python 3.10)
RUN python3.10 -m venv env_gnnkan && \
    . env_gnnkan/bin/activate && \
    pip install --upgrade pip && \
    pip install wheel && \
    pip install numpy==1.23.5 && \
    # 安裝基礎依賴
    pip install --default-timeout=100 --retries=10 --no-cache-dir -r requirements.txt && \
    # 安裝 GPU 版本的 PyTorch (使用 CUDA 11.8)
    pip install --default-timeout=100 --retries=10 --no-cache-dir torch==1.12.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116 && \
    # 安裝 PyG (PyTorch Geometric) GPU 版本 (使用 CUDA 11.6 相容 11.8)
    pip install --default-timeout=100 --retries=10 --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.1+cu116.html && \
    # 安裝 Jupyter
    pip install jupyter notebook && \
    # 確保 tigramite 和 scikit-network 正確安裝
    pip install --default-timeout=100 --retries=10 --no-cache-dir tigramite==4.2.2.1 scikit-network==0.31.0

# 執行連結腳本來連結自訂的庫檔案
RUN chmod +x script/link.sh && \
    bash script/link.sh || true

# 建立資料目錄
RUN mkdir -p data results

# 設定權限
RUN chmod -R 755 /app

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python --version || exit 1

# 暴露常用連接埠
EXPOSE 8080 8888

# 設定標籤
LABEL maintainer="RCAEval GNN-KAN Project"
LABEL description="Docker environment for RCAEval GNN-KAN with GPU support"
LABEL version="1.0.0"

# 預設指令：啟動 GNN-KAN 環境的 bash
CMD ["/bin/bash", "-c", "source env_gnnkan/bin/activate && echo 'RCAEval GNN-KAN environment with GPU support ready.' && /bin/bash"]