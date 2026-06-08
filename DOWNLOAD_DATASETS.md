# 數據集下載指南

## 快速開始

使用 `download_all_datasets.py` 腳本可以自動下載所有實驗所需的數據集。

### 基本用法

```bash
# 使用 Python 3.10 環境（推薦）
source .venv310/bin/activate
python download_all_datasets.py

# 或直接使用虛擬環境中的 Python
.venv310/bin/python download_all_datasets.py
```

### 常用命令

```bash
# 1. 下載所有數據集（跳過已存在的）
python download_all_datasets.py

# 2. 強制重新下載所有數據集
python download_all_datasets.py --no-skip-existing

# 3. 只下載特定數據集
python download_all_datasets.py --datasets re1,re2,re3

# 4. 下載基礎數據集和 RE 系列
python download_all_datasets.py --datasets online-boutique,sock-shop-1,re1,re2

# 5. 只下載 RE 系列（用於完整實驗）
python download_all_datasets.py --datasets re1,re2,re3
```

## 可用數據集

### 基礎數據集
- `online-boutique` - Online Boutique 微服務系統 (~100MB)
- `sock-shop-1` - Sock Shop v1 微服務系統 (~100MB)
- `sock-shop-2` - Sock Shop v2 微服務系統 (~100MB)
- `train-ticket` - Train Ticket 微服務系統 (~100MB)

### RE 系列數據集
- `re1` - RE1 數據集，包含 RE1-OB, RE1-SS, RE1-TT (~390MB)
- `re2` - RE2 數據集，包含 RE2-OB, RE2-SS, RE2-TT (~4.2GB)
- `re3` - RE3 數據集，包含 RE3-OB, RE3-SS, RE3-TT (~534MB)

### 合成數據集
- `syn_circa` - CIRCA 合成數據集 (~50MB)
- `syn_rcd` - RCD 合成數據集 (~50MB)
- `syn_causil` - CauSIL 合成數據集 (~50MB)

### 多源數據集
- `multi-source` - 多源遙測數據樣本 (~10MB)

## 數據集大小估算

| 數據集類型 | 大小 | 下載時間（取決於網路） |
|-----------|------|---------------------|
| 基礎數據集 | ~400MB | 5-10 分鐘 |
| RE1 系列 | ~390MB | 5-10 分鐘 |
| RE2 系列 | ~4.2GB | 30-60 分鐘 |
| RE3 系列 | ~534MB | 10-15 分鐘 |
| 合成數據集 | ~150MB | 3-5 分鐘 |
| **總計** | **~5.7GB** | **1-2 小時** |

## 使用建議

### 快速測試
如果只是想快速測試功能，建議只下載基礎數據集：
```bash
python download_all_datasets.py --datasets online-boutique,sock-shop-1
```

### 標準實驗
對於標準的實驗，建議下載 RE1 系列：
```bash
python download_all_datasets.py --datasets re1
```

### 完整實驗
對於完整的實驗和論文重現，需要下載所有 RE 系列：
```bash
python download_all_datasets.py --datasets re1,re2,re3
```

### 方法比較
如果需要比較不同方法，建議下載合成數據集：
```bash
python download_all_datasets.py --datasets syn_circa,syn_rcd,syn_causil
```

## 故障排除

### 下載失敗
如果某個數據集下載失敗：
1. 檢查網路連接
2. 確認磁碟空間充足（至少需要 10GB 可用空間）
3. 單獨重新下載失敗的數據集：
   ```bash
   python download_all_datasets.py --datasets <失敗的數據集名稱>
   ```

### 驗證失敗
如果下載完成但驗證失敗：
1. 檢查 `data/` 目錄權限
2. 手動檢查數據集目錄是否為空
3. 使用 `--no-skip-existing` 強制重新下載

### 網路問題
如果遇到網路超時：
- 腳本會自動重試，但可能需要多次執行
- 可以分批下載，先下載小數據集，再下載大數據集

## 數據集位置

所有數據集會下載到 `data/` 目錄下：
```
data/
├── online-boutique/
├── sock-shop-1/
├── sock-shop-2/
├── train-ticket/
├── RE1/
│   ├── RE1-OB/
│   ├── RE1-SS/
│   └── RE1-TT/
├── RE2/
│   ├── RE2-OB/
│   ├── RE2-SS/
│   └── RE2-TT/
├── RE3/
│   ├── RE3-OB/
│   ├── RE3-SS/
│   └── RE3-TT/
├── syn_circa/
├── syn_rcd/
├── syn_causil/
└── multi-source-data/
```

## 驗證下載

下載完成後，可以運行以下命令驗證數據集：
```bash
# 檢查數據集目錄
ls -lh data/

# 檢查特定數據集
ls -lh data/RE1/
ls -lh data/RE2/
```

## 注意事項

1. **磁碟空間**：確保至少有 10GB 可用空間
2. **網路穩定**：RE2 系列較大（~4.2GB），建議在網路穩定的環境下下載
3. **時間**：完整下載所有數據集可能需要 1-2 小時
4. **虛擬環境**：建議在虛擬環境中運行，確保依賴已正確安裝

## 相關文檔

- 項目 README: [README.md](README.md)
- 設置指南: [docs/SETUP.md](docs/SETUP.md)
- 數據集詳細信息: 參見 README.md 中的 "Available Datasets" 部分

