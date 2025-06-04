#!/bin/bash
# GNN-KAN RCA 安裝和執行腳本

echo "🚀 GNN-KAN RCA Setup and Execution Script"
echo "=========================================="

# 檢查 Python 版本
echo "📋 Checking Python version..."
python --version

# 安裝必要的依賴
echo "📦 Installing dependencies..."

# 基本依賴
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install scikit-network
pip install statsmodels
pip install networkx
pip install scipy
pip install sklearn
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn

echo "✅ Dependencies installed!"

# 運行測試
echo "🧪 Running GNN-KAN tests..."
cd /Users/user/RCAEval
python tests/test_gnn_kan.py

echo "📊 Running comparison with other RCA methods..."
python experiments/compare_gnn_kan.py

echo "🎉 GNN-KAN RCA setup completed!"
echo ""
echo "📖 Usage examples:"
echo "-------------------"
echo "# 在 Python 中使用："
echo "from RCAEval.e2e.gnn_kan import gnn_kan_rca"
echo "result = gnn_kan_rca(data, inject_time=inject_time, dataset='your_dataset')"
echo ""
echo "# 或在 RCAEval 框架中："
echo "from RCAEval.e2e import gnn_kan_rca"
echo "result = gnn_kan_rca(data, inject_time=inject_time)"
echo ""
echo "📈 結果格式："
echo "{'adj': adjacency_matrix, 'node_names': [node_list], 'ranks': [ranked_nodes]}"