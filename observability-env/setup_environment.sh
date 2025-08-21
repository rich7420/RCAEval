#!/bin/bash

# 設置觀測性環境的腳本

set -e

echo "🚀 設置觀測性數據收集環境"
echo "=" * 50

# 檢查 Python 版本
echo "🐍 檢查 Python 版本..."
python3 --version

# 檢查是否在虛擬環境中
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ 已在虛擬環境中: $VIRTUAL_ENV"
else
    echo "⚠️ 建議在虛擬環境中運行"
    echo "   創建虛擬環境: python3 -m venv venv"
    echo "   激活虛擬環境: source venv/bin/activate"
fi

# 升級 pip
echo "📦 升級 pip..."
python3 -m pip install --upgrade pip

# 安裝依賴
echo "📦 安裝依賴包..."
python3 -m pip install -r requirements.txt

# 檢查安裝
echo "🔍 檢查關鍵依賴..."
python3 -c "import requests; print('✅ requests:', requests.__version__)"
python3 -c "import pandas; print('✅ pandas:', pandas.__version__)"
python3 -c "import numpy; print('✅ numpy:', numpy.__version__)"

# 創建必要的目錄
echo "📁 創建必要的目錄..."
mkdir -p collectors/output
mkdir -p collectors/logs
mkdir -p collectors/temp
mkdir -p re2_format_data

# 設置權限
echo "🔐 設置權限..."
chmod +x collectors/*.py
chmod +x scripts/*.sh

echo "✅ 環境設置完成！"
echo ""
echo "💡 後續操作:"
echo "  1. 啟動觀測性服務: ./scripts/start-demo.sh"
echo "  2. 運行數據收集: cd collectors && python3 collect_available_data.py"
echo "  3. 檢查診斷: cd collectors && python3 debug_collection.py"