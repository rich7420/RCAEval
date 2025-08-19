#!/usr/bin/env python3
"""
智能數據集管理器
自動檢查、下載和驗證所有數據集
- 已存在的數據集跳過下載
- 缺失的數據集自動下載
- 完整的驗證和報告
"""

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

# 添加項目路徑
sys.path.insert(0, '.')

class SmartDatasetManager:
    """智能數據集管理器"""
    
    def __init__(self):
        self.all_datasets = {
            # 基礎數據集
            "online-boutique": {
                "path": "data/online-boutique",
                "download_func": "download_online_boutique_dataset",
                "description": "Online Boutique微服務系統",
                "series": "basic"
            },
            "sock-shop-1": {
                "path": "data/sock-shop-1",
                "download_func": "download_sock_shop_1_dataset", 
                "description": "Sock Shop v1 微服務系統",
                "series": "basic"
            },
            "sock-shop-2": {
                "path": "data/sock-shop-2",
                "download_func": "download_sock_shop_2_dataset",
                "description": "Sock Shop v2 微服務系統", 
                "series": "basic"
            },
            "train-ticket": {
                "path": "data/train-ticket",
                "download_func": "download_train_ticket_dataset",
                "description": "Train Ticket微服務系統",
                "series": "basic"
            },
            
            # RE1 系列數據集
            "re1-ob": {
                "path": "data/RE1/RE1-OB",
                "download_func": "download_re1_dataset",
                "description": "RE1 Online Boutique數據集",
                "series": "RE1"
            },
            "re1-ss": {
                "path": "data/RE1/RE1-SS", 
                "download_func": "download_re1_dataset",
                "description": "RE1 Sock Shop數據集",
                "series": "RE1"
            },
            "re1-tt": {
                "path": "data/RE1/RE1-TT",
                "download_func": "download_re1_dataset", 
                "description": "RE1 Train Ticket數據集",
                "series": "RE1"
            },
            
            # RE2 系列數據集
            "re2-ob": {
                "path": "data/RE2/RE2-OB",
                "download_func": "download_re2_dataset",
                "description": "RE2 Online Boutique數據集",
                "series": "RE2"
            },
            "re2-ss": {
                "path": "data/RE2/RE2-SS",
                "download_func": "download_re2_dataset",
                "description": "RE2 Sock Shop數據集", 
                "series": "RE2"
            },
            "re2-tt": {
                "path": "data/RE2/RE2-TT",
                "download_func": "download_re2_dataset",
                "description": "RE2 Train Ticket數據集",
                "series": "RE2"
            },
            
            # RE3 系列數據集
            "re3-ob": {
                "path": "data/RE3/RE3-OB",
                "download_func": "download_re3_dataset",
                "description": "RE3 Online Boutique數據集",
                "series": "RE3"
            },
            "re3-ss": {
                "path": "data/RE3/RE3-SS",
                "download_func": "download_re3_dataset",
                "description": "RE3 Sock Shop數據集",
                "series": "RE3"
            },
            "re3-tt": {
                "path": "data/RE3/RE3-TT", 
                "download_func": "download_re3_dataset",
                "description": "RE3 Train Ticket數據集",
                "series": "RE3"
            },
            
            # 多模態數據集
            "multi-source": {
                "path": "data/multi-source-data",
                "download_func": "download_multi_source_sample",
                "description": "多源遙測數據樣本",
                "series": "multimodal"
            },
            
            # 合成數據集 - CIRCA 系列
            "circa10": {
                "path": "data/syn_circa/10",
                "download_func": "download_syn_circa_dataset",
                "description": "CIRCA10 合成數據集",
                "series": "synthetic"
            },
            "circa50": {
                "path": "data/syn_circa/50", 
                "download_func": "download_syn_circa_dataset",
                "description": "CIRCA50 合成數據集",
                "series": "synthetic"
            },
            
            # 合成數據集 - RCD 系列
            "rcd10": {
                "path": "data/syn_rcd/10",
                "download_func": "download_syn_rcd_dataset",
                "description": "RCD10 合成數據集",
                "series": "synthetic"
            },
            "rcd50": {
                "path": "data/syn_rcd/50",
                "download_func": "download_syn_rcd_dataset", 
                "description": "RCD50 合成數據集",
                "series": "synthetic"
            },
            
            # 合成數據集 - CauSIL 系列
            "causil": {
                "path": "data/syn_causil",
                "download_func": "download_syn_causil_dataset",
                "description": "CauSIL 合成數據集",
                "series": "synthetic"
            }
        }
        
        self.download_functions = {}
        self.existing_datasets = []
        self.missing_datasets = []
        
    def check_dataset_status(self) -> Tuple[List[str], List[str]]:
        """檢查所有數據集的存在狀態"""
        print("🔍 檢查所有數據集狀態...")
        
        existing = []
        missing = []
        
        for name, config in self.all_datasets.items():
            path = config["path"]
            if os.path.exists(path) and os.listdir(path):  # 檢查路徑存在且非空
                existing.append(name)
                print(f"✅ {name}: {path}")
            else:
                missing.append(name)
                print(f"❌ {name}: {path} (缺失或為空)")
        
        self.existing_datasets = existing
        self.missing_datasets = missing
        
        print(f"\n📊 數據集狀態統計:")
        print(f"   總數據集: {len(self.all_datasets)}")
        print(f"   已存在: {len(existing)}")
        print(f"   需下載: {len(missing)}")
        
        return existing, missing
    
    def load_download_functions(self) -> bool:
        """載入所有下載函數"""
        print("\n🔧 載入下載函數...")
        
        try:
            from RCAEval.utility import (
                download_online_boutique_dataset,
                download_sock_shop_1_dataset,
                download_sock_shop_2_dataset, 
                download_train_ticket_dataset,
                download_re1_dataset,
                download_re2_dataset,
                download_re3_dataset,
                download_syn_circa_dataset,
                download_syn_rcd_dataset,
                download_syn_causil_dataset
            )
            
            # 嘗試載入多源數據集函數
            try:
                from RCAEval.utility import download_multi_source_sample
            except ImportError:
                print("⚠️ download_multi_source_sample 不可用，將跳過")
                download_multi_source_sample = None
            
            # 建立函數映射
            self.download_functions = {
                "download_online_boutique_dataset": download_online_boutique_dataset,
                "download_sock_shop_1_dataset": download_sock_shop_1_dataset,
                "download_sock_shop_2_dataset": download_sock_shop_2_dataset,
                "download_train_ticket_dataset": download_train_ticket_dataset,
                "download_re1_dataset": download_re1_dataset,
                "download_re2_dataset": download_re2_dataset,
                "download_re3_dataset": download_re3_dataset,
                "download_syn_circa_dataset": download_syn_circa_dataset,
                "download_syn_rcd_dataset": download_syn_rcd_dataset,
                "download_syn_causil_dataset": download_syn_causil_dataset,
            }
            
            if download_multi_source_sample:
                self.download_functions["download_multi_source_sample"] = download_multi_source_sample
            
            print(f"✅ 成功載入 {len(self.download_functions)} 個下載函數")
            return True
            
        except ImportError as e:
            print(f"❌ 下載函數載入失敗: {e}")
            return False
    
    def group_downloads_by_function(self) -> Dict[str, List[str]]:
        """按下載函數分組數據集（避免重複下載）"""
        download_groups = {}
        
        for dataset_name in self.missing_datasets:
            config = self.all_datasets[dataset_name]
            func_name = config["download_func"]
            
            if func_name not in download_groups:
                download_groups[func_name] = []
            download_groups[func_name].append(dataset_name)
        
        return download_groups
    
    def download_missing_datasets(self) -> Tuple[List[str], List[str]]:
        """下載所有缺失的數據集"""
        if not self.missing_datasets:
            print("🎉 所有數據集都已存在，無需下載！")
            return [], []
        
        print(f"\n📦 開始下載 {len(self.missing_datasets)} 個缺失數據集...")
        
        download_groups = self.group_downloads_by_function()
        print(f"📋 需要執行 {len(download_groups)} 個下載任務:")
        
        for func_name, datasets in download_groups.items():
            print(f"   {func_name}: {datasets}")
        
        successful_downloads = []
        failed_downloads = []
        
        for func_name, datasets in download_groups.items():
            print(f"\n🔄 執行下載: {func_name}")
            print(f"   涵蓋數據集: {datasets}")
            
            if func_name not in self.download_functions:
                print(f"❌ 下載函數 {func_name} 不可用")
                failed_downloads.extend(datasets)
                continue
            
            try:
                start_time = time.time()
                
                # 執行下載
                download_func = self.download_functions[func_name]
                download_func()
                
                download_time = time.time() - start_time
                print(f"✅ 下載完成 ({download_time:.1f}秒)")
                
                # 驗證下載結果
                verified_datasets = []
                for dataset in datasets:
                    path = self.all_datasets[dataset]["path"]
                    if os.path.exists(path) and os.listdir(path):
                        verified_datasets.append(dataset)
                        print(f"   ✅ {dataset} 驗證成功")
                    else:
                        print(f"   ❌ {dataset} 驗證失敗")
                
                successful_downloads.extend(verified_datasets)
                failed_datasets = [d for d in datasets if d not in verified_datasets]
                failed_downloads.extend(failed_datasets)
                
            except Exception as e:
                print(f"❌ 下載失敗: {e}")
                failed_downloads.extend(datasets)
                traceback.print_exc()
        
        return successful_downloads, failed_downloads
    
    def verify_all_datasets(self) -> Dict[str, bool]:
        """驗證所有數據集的完整性"""
        print("\n🔍 驗證所有數據集完整性...")
        
        verification_results = {}
        
        for name, config in self.all_datasets.items():
            path = config["path"]
            
            if os.path.exists(path):
                # 檢查目錄是否非空
                if os.path.isdir(path) and os.listdir(path):
                    verification_results[name] = True
                    print(f"✅ {name}: 完整")
                else:
                    verification_results[name] = False
                    print(f"❌ {name}: 目錄為空")
            else:
                verification_results[name] = False
                print(f"❌ {name}: 不存在")
        
        verified_count = sum(verification_results.values())
        total_count = len(verification_results)
        
        print(f"\n📊 驗證結果統計:")
        print(f"   驗證通過: {verified_count}/{total_count}")
        print(f"   完整性: {verified_count/total_count*100:.1f}%")
        
        return verification_results
    
    def generate_comprehensive_report(self, successful_downloads: List[str], 
                                    failed_downloads: List[str],
                                    verification_results: Dict[str, bool]):
        """生成完整的數據集管理報告"""
        print("\n📋 生成完整報告...")
        
        # 按系列分組統計
        series_stats = {}
        for name, config in self.all_datasets.items():
            series = config["series"]
            if series not in series_stats:
                series_stats[series] = {"total": 0, "available": 0, "datasets": []}
            
            series_stats[series]["total"] += 1
            series_stats[series]["datasets"].append(name)
            
            if verification_results.get(name, False):
                series_stats[series]["available"] += 1
        
        # 生成報告
        report = f"""# 智能數據集管理完整報告

## 執行摘要
- 執行時間: {time.strftime('%Y-%m-%d %H:%M:%S')}
- 總數據集: {len(self.all_datasets)}
- 原有數據集: {len(self.existing_datasets)}
- 需要下載: {len(self.missing_datasets)}
- 成功下載: {len(successful_downloads)}
- 下載失敗: {len(failed_downloads)}
- 最終可用: {sum(verification_results.values())}

## 數據集系列統計
"""
        
        for series, stats in series_stats.items():
            completion_rate = stats["available"] / stats["total"] * 100
            report += f"""
### {series.upper()} 系列
- 完整性: {stats["available"]}/{stats["total"]} ({completion_rate:.1f}%)
- 數據集: {', '.join(stats["datasets"])}
"""
        
        # 可用數據集列表
        available_datasets = [name for name, verified in verification_results.items() if verified]
        if available_datasets:
            report += f"""
## 可用數據集列表 ({len(available_datasets)} 個)
"""
            for series in ["basic", "RE1", "RE2", "RE3", "multimodal", "synthetic"]:
                series_datasets = [name for name in available_datasets 
                                 if self.all_datasets[name]["series"] == series]
                if series_datasets:
                    report += f"""
### {series.upper()} 系列
"""
                    for dataset in series_datasets:
                        desc = self.all_datasets[dataset]["description"]
                        report += f"- ✅ **{dataset}**: {desc}\n"
        
        # 本次下載結果
        if successful_downloads or failed_downloads:
            report += f"""
## 本次下載結果
"""
            if successful_downloads:
                report += f"""
### 成功下載 ({len(successful_downloads)} 個)
"""
                for dataset in successful_downloads:
                    report += f"- ✅ {dataset}\n"
            
            if failed_downloads:
                report += f"""
### 下載失敗 ({len(failed_downloads)} 個)
"""
                for dataset in failed_downloads:
                    report += f"- ❌ {dataset}\n"
        
        # 測試建議
        if available_datasets:
            basic_datasets = [d for d in available_datasets if self.all_datasets[d]["series"] == "basic"]
            re_datasets = [d for d in available_datasets if d.startswith("re")]
            synthetic_datasets = [d for d in available_datasets if self.all_datasets[d]["series"] == "synthetic"]
            
            report += f"""
## 測試建議

### 基礎功能測試
```bash
# 使用基礎數據集測試
python comparison.py --methods gnn_kan baro --datasets {' '.join(basic_datasets[:2])} --limit 3
```

### RE 系列測試
```bash
# 測試 RE 系列數據集
python comparison.py --methods gnn_kan baro --datasets {' '.join(re_datasets[:3])} --limit 2
```
"""
            
            if synthetic_datasets:
                report += f"""
### 合成數據集測試
```bash
# 測試合成數據集
python comparison.py --methods gnn_kan baro circa rcd --datasets {' '.join(synthetic_datasets[:2])} --limit 2
```
"""
            
            report += f"""
### 完整性能測試
```bash
# 大規模測試
python comparison.py --methods gnn_kan baro --datasets {' '.join(available_datasets[:5])} --limit 5
```
"""
        
        # 下一步建議
        missing_datasets = [name for name, verified in verification_results.items() if not verified]
        if missing_datasets:
            report += f"""
## 下一步建議

### 仍需處理的數據集
"""
            for dataset in missing_datasets:
                report += f"- ❌ {dataset}: 需要手動檢查或重新下載\n"
            
            report += f"""
### 手動下載命令
```python
# 在 Python 中手動下載
from RCAEval.utility import *

# 例如重新下載失敗的數據集
# download_re3_dataset()  # 如果 RE3 系列失敗
# download_syn_circa_dataset()  # 如果 CIRCA 系列失敗
```
"""
        
        report += f"""
## 數據集使用指南

### 數據集選擇建議
1. **快速測試**: 使用 `online-boutique`, `sock-shop-1`
2. **標準測試**: 使用 RE1 系列 (`re1-ob`, `re1-ss`, `re1-tt`)
3. **大規模測試**: 使用 RE2/RE3 系列
4. **方法比較**: 使用合成數據集 (`circa10`, `rcd10`)
5. **多模態測試**: 使用 `multi-source`

### 性能考量
- 小型數據集 (< 1GB): `circa10`, `rcd10`, `sock-shop-1`
- 中型數據集 (1-5GB): RE1 系列, `online-boutique`
- 大型數據集 (> 5GB): RE2/RE3 系列

### 故障排除
如果某個數據集下載失敗：
1. 檢查網路連接
2. 確認磁碟空間充足
3. 手動執行對應的下載函數
4. 檢查 Zenodo 服務狀態
"""
        
        # 保存報告
        with open("comprehensive_dataset_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        print("✅ 完整報告已保存到 comprehensive_dataset_report.md")

def main():
    """主函數"""
    print("=" * 70)
    print("🚀 智能數據集管理器 - 全自動檢查、下載、驗證")
    print("=" * 70)
    
    manager = SmartDatasetManager()
    
    try:
        # 步驟 1: 檢查現有數據集狀態
        print("\n📋 步驟 1: 檢查數據集狀態")
        existing, missing = manager.check_dataset_status()
        
        # 步驟 2: 載入下載函數
        print("\n📋 步驟 2: 載入下載函數")
        if not manager.load_download_functions():
            print("❌ 下載函數載入失敗，無法繼續")
            return False
        
        # 步驟 3: 下載缺失數據集
        print("\n📋 步驟 3: 下載缺失數據集")
        successful_downloads, failed_downloads = manager.download_missing_datasets()
        
        # 步驟 4: 驗證所有數據集
        print("\n📋 步驟 4: 驗證所有數據集")
        verification_results = manager.verify_all_datasets()
        
        # 步驟 5: 生成完整報告
        print("\n📋 步驟 5: 生成完整報告")
        manager.generate_comprehensive_report(
            successful_downloads, failed_downloads, verification_results
        )
        
        # 最終結果
        total_available = sum(verification_results.values())
        total_datasets = len(manager.all_datasets)
        completion_rate = total_available / total_datasets * 100
        
        print("\n" + "=" * 70)
        print("🎯 數據集管理完成")
        print("=" * 70)
        print(f"📊 最終狀態:")
        print(f"   總數據集: {total_datasets}")
        print(f"   可用數據集: {total_available}")
        print(f"   完整性: {completion_rate:.1f}%")
        
        if successful_downloads:
            print(f"   本次下載成功: {len(successful_downloads)} 個")
        
        if failed_downloads:
            print(f"   本次下載失敗: {len(failed_downloads)} 個")
        
        if completion_rate >= 80:
            print("\n🎉 數據集準備就緒！可以開始 GNN-KAN 測試")
            print("\n🚀 建議測試命令:")
            available_datasets = [name for name, verified in verification_results.items() if verified]
            if available_datasets:
                print(f"python comparison.py --methods gnn_kan baro --datasets {' '.join(available_datasets[:3])} --limit 3")
        else:
            print("\n⚠️ 數據集完整性較低，建議檢查網路連接後重新執行")
        
        return completion_rate >= 50  # 至少50%完整性才算成功
        
    except Exception as e:
        print(f"❌ 執行過程中出現錯誤: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)