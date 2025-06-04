FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
#docker compose build --no-cache
# 使用 NVIDIA CUDA 基礎映像，包含 GPU 支持
# FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# FROM ubuntu:22.04
# 使用 Ubuntu 22.04 作為CPU基礎映像
# 設置環境變數
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Taipei
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# 設置 NumPy API 環境變量以解決 tigramite 編譯問題
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
        # 添加編譯依賴以解決 tigramite 和 scikit-network 問題
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

# 設置 Python 3.10 為預設
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# 設置工作目錄
WORKDIR /app

# 複製專案文件
COPY . .

# 創建 GNN-KAN 環境 (Python 3.10)
RUN python3.10 -m venv env_gnnkan && \
    . env_gnnkan/bin/activate && \
    pip install --upgrade pip && \
    pip install wheel && \
    pip install numpy==1.23.5 && \
    # 安裝基礎依賴
    pip install --default-timeout=100 --retries=10 --no-cache-dir -r requirements.txt && \
    # 安裝 GPU 版本的 PyTorch
    pip install --default-timeout=100 --retries=10 --no-cache-dir torch==1.12.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    # 安裝 PyG (PyTorch Geometric) GPU 版本
    pip install --default-timeout=100 --retries=10 --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.1+cu121.html && \
    # 安裝 Jupyter
    pip install jupyter notebook && \
    # 確保 tigramite 和 scikit-network 正確安裝
    pip install --default-timeout=100 --retries=10 --no-cache-dir tigramite==4.2.2.1 scikit-network==0.31.0

# 執行連結腳本來連結自定義的庫文件
RUN chmod +x script/link.sh && \
    bash script/link.sh || true

# 創建數據目錄
RUN mkdir -p data results

# 設置權限
RUN chmod -R 755 /app

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python --version || exit 1

# 暴露常用端口
EXPOSE 8080 8888

# 設置標籤
LABEL maintainer="RCAEval GNN-KAN Project"
LABEL description="Docker environment for RCAEval GNN-KAN with GPU support"
LABEL version="1.0.0"

# 預設命令：啟動 GNN-KAN 環境的 bash
CMD ["/bin/bash", "-c", "source env_gnnkan/bin/activate && echo 'RCAEval GNN-KAN environment with GPU support ready.' && /bin/bash"]