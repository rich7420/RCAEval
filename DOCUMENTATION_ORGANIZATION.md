# 📚 RCAEval 文檔與測試文件組織說明

## 📋 整理總結 (2024年6月)

### ✅ 保留的核心文件

#### 🧪 測試文件 (5個)
1. **`run_modularized_tests.py`** - 主要測試套件
   - 用途：完整的GNN-KAN模組化測試
   - 狀態：✅ 最新，功能完整
   - 使用：`python run_modularized_tests.py`

2. **`test_advanced_metrics.py`** - 高級指標測試
   - 用途：測試參數效率、可解釋性、計算效率指標
   - 狀態：✅ 最新，與比較系統集成
   - 使用：`python test_advanced_metrics.py`

3. **`test_comparison_quick.py`** - 快速比較測試
   - 用途：快速驗證GNN-KAN vs BARO比較功能
   - 狀態：✅ 最新，簡化測試流程
   - 使用：`python test_comparison_quick.py`

4. **`gpu_comprehensive_test.py`** - GPU綜合測試
   - 用途：GPU可用性檢查、KAN層性能基準測試
   - 狀態：✅ 最新，整合了所有GPU測試功能
   - 使用：`python gpu_comprehensive_test.py`

5. **`gpu_usage_check.py`** - GPU使用檢查
   - 用途：詳細的GPU診斷和使用情況檢查
   - 狀態：✅ 保留，提供詳細診斷信息
   - 使用：`python gpu_usage_check.py`

#### 📖 說明文檔 (6個核心)
1. **`README.md`** - 主要項目說明
   - 用途：項目總體介紹和使用指南
   - 狀態：✅ 保留

2. **`README_COMPARISON.md`** - 比較功能說明
   - 用途：GNN-KAN vs BARO比較系統使用指南
   - 狀態：✅ 最新，包含高級指標說明
   - 內容：基本指標、高級指標、使用方法

3. **`TEST_FILES_ORGANIZATION.md`** - 測試文件組織
   - 用途：測試文件清理和組織記錄
   - 狀態：✅ 保留，記錄清理過程

4. **`GPU_QUICKSTART.md`** - GPU快速開始
   - 用途：GPU環境快速設置指南
   - 狀態：✅ 保留，Docker GPU設置

5. **`docker.md`** - Docker說明
   - 用途：Docker環境配置和使用
   - 狀態：✅ 保留

6. **`GNN_KAN_COMPLETE_TECHNICAL_GUIDE.md`** - 完整技術指南
   - 用途：GNN-KAN技術細節和實現說明
   - 狀態：✅ 保留，技術參考文檔
   - 內容：2849行完整技術文檔

### 🗂️ 建議歸檔的文件 (3個)

#### 📁 移至 `docs/archive/` 目錄
1. **`GNN_KAN_Implementation_Guide.md`** - 實現指南
   - 原因：與完整技術指南重複，內容較舊
   - 建議：歸檔到 `docs/archive/`

2. **`MODULARIZATION_COMPLETE_REPORT.md`** - 模組化完成報告
   - 原因：模組化已完成，報告可歸檔
   - 建議：歸檔到 `docs/archive/`

3. **`gnn_kan.md`** - 早期GNN-KAN計劃
   - 原因：早期實現計劃，已被完整技術指南取代
   - 建議：歸檔到 `docs/archive/`

### 📊 文件統計

#### 整理前
- 測試文件：5個 (全部保留)
- 說明文檔：9個 → 6個核心 + 3個歸檔

#### 整理後
- **核心文件**：11個 (5測試 + 6文檔)
- **歸檔文件**：3個
- **減少比例**：25% (9→6 核心文檔)

### 🎯 使用指南

#### 新用戶快速開始
1. 閱讀 `README.md` - 項目總覽
2. 查看 `GPU_QUICKSTART.md` - 環境設置
3. 運行 `test_comparison_quick.py` - 功能驗證

#### 開發者深入了解
1. 閱讀 `GNN_KAN_COMPLETE_TECHNICAL_GUIDE.md` - 技術細節
2. 查看 `README_COMPARISON.md` - 比較系統
3. 運行 `run_modularized_tests.py` - 完整測試

#### 性能測試
1. 運行 `gpu_comprehensive_test.py` - GPU性能
2. 運行 `test_advanced_metrics.py` - 高級指標
3. 查看 `gpu_usage_check.py` - 詳細診斷

### 🔧 維護建議

#### 定期更新 (每月)
- 檢查測試文件是否與最新代碼同步
- 更新README中的使用示例
- 驗證GPU測試在不同環境下的兼容性

#### 版本控制
- 重要文檔變更記錄在git commit中
- 歸檔文件保留git歷史記錄
- 定期備份測試結果和性能基準

### ✅ 整理完成狀態

**當前狀態**：文檔和測試文件已完成整理
- ✅ 核心功能文件保留並更新
- ✅ 重複和過時文件識別
- ✅ 歸檔建議明確
- ✅ 使用指南完整

**下一步**：執行歸檔操作，創建 `docs/archive/` 目錄並移動相應文件。 