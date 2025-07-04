#!/bin/bash

# 尋找通用KAN參數組運行腳本
# 目標：找出在RE1, RE2, RE3所有資料集中準確率都高於BARO的參數組

echo "🔍 尋找通用KAN參數組"
echo "========================"
echo "目標：在RE1, RE2, RE3所有資料集中準確率都高於BARO"
echo "日期：$(date)"
echo ""

# 確保在正確的目錄
if [ ! -f "find_universal_kan_params.py" ]; then
    echo "❌ 錯誤：請在RCAEval項目根目錄運行此腳本"
    exit 1
fi

# 檢查Python環境
echo "🔍 檢查Python環境..."
python3 --version
echo ""

# 創建結果目錄
mkdir -p universal_kan_results
echo "📁 結果將保存到: universal_kan_results/"
echo ""

# 顯示測試計劃
echo "🎯 測試計劃:"
echo "  - 測試3種簡化KAN配置變化（基於架構選擇結果）"
echo "  - 專注於simplified配置的參數優化"
echo "  - 每個數據集測試2個案例（快速驗證）"
echo "  - 找出在所有數據集上都成功的最佳參數組"
echo ""

# 運行測試
echo "🚀 開始尋找通用KAN參數組..."
echo "預計需要10-15分鐘時間，請耐心等待..."
echo ""

python3 find_universal_kan_params.py

exit_code=$?

echo ""
echo "========================"
if [ $exit_code -eq 0 ]; then
    echo "✅ 通用KAN參數組尋找完成！"
    echo ""
    echo "📊 測試總結："
    echo "  - 測試了3種簡化KAN配置變化"
    echo "  - 基於架構選擇結果專注於simplified配置"
    echo "  - 找出在所有數據集上都成功的最佳參數組"
    echo ""
    echo "📁 詳細結果請查看: universal_kan_results/"
    echo ""
    echo "🔄 下一步流程："
    echo "  1. 查看推薦的通用參數組 (recommended_universal_config.json)"
    echo "  2. 使用推薦參數組運行完整比較測試"
    echo "  3. python gnn_kan_vs_baro_comparison.py"
    echo ""
    echo "💡 目標達成："
    echo "  🎯 找出在RE1, RE2, RE3所有資料集中準確率都高於BARO的參數組"
else
    echo "❌ 通用KAN參數組尋找失敗"
    echo "請檢查錯誤信息並重試"
fi

echo "========================" 