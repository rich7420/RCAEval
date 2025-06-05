#!/bin/bash

# GNN-KAN RCA 執行與比較腳本
# 使用方法: ./run_gnn_kan_comparison.sh

echo "======================================"
echo "GNN-KAN RCA 執行與比較腳本"
echo "======================================"

# 檢查 GPU 環境
echo "檢查 GPU 環境..."
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'GPU 數量: {torch.cuda.device_count()}') if torch.cuda.is_available() else None"

echo ""
echo "1. 執行 GNN-KAN 方法 (測試模式)"
echo "--------------------------------------"
python main.py --dataset online-boutique --method gnn_kan --test

echo ""
echo "2. 執行其他方法進行比較 (測試模式)"
echo "--------------------------------------"

# 比較方法列表
methods=("causalrca" "nsigma" "microcause" "pc_pagerank" "dummy")

for method in "${methods[@]}"; do
    echo "執行 $method..."
    python main.py --dataset online-boutique --method $method --test
    echo "$method 完成"
    echo ""
done

echo ""
echo "3. 使用專用比較工具"
echo "--------------------------------------"
python experiments/compare_gnn_kan.py

echo ""
echo "4. 執行完整評估 (如果需要)"
echo "--------------------------------------"
echo "如果想要完整評估，請執行以下命令:"
echo "python main.py --dataset online-boutique --method gnn_kan"
echo "python main.py --dataset online-boutique --method causalrca"

echo ""
echo "比較完成！結果保存在 output/ 目錄中"