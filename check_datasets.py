#!/usr/bin/env python3
"""
检查所有数据集的状态
"""

import os
from pathlib import Path

# 定义所有数据集及其路径
ALL_DATASETS = {
    # 基础数据集
    "online-boutique": {
        "path": "data/online-boutique",
        "description": "Online Boutique 微服務系統",
    },
    "sock-shop-1": {
        "path": "data/sock-shop-1",
        "description": "Sock Shop v1 微服務系統",
    },
    "sock-shop-2": {
        "path": "data/sock-shop-2",
        "description": "Sock Shop v2 微服務系統",
    },
    "train-ticket": {
        "path": "data/train-ticket",
        "description": "Train Ticket 微服務系統",
    },
    
    # RE1 系列
    "re1": {
        "path": "data/RE1",
        "description": "RE1 數據集（包含 RE1-OB, RE1-SS, RE1-TT）",
        "sub_datasets": ["RE1-OB", "RE1-SS", "RE1-TT"]
    },
    
    # RE2 系列
    "re2": {
        "path": "data/RE2",
        "description": "RE2 數據集（包含 RE2-OB, RE2-SS, RE2-TT）",
        "sub_datasets": ["RE2-OB", "RE2-SS", "RE2-TT"]
    },
    
    # RE3 系列
    "re3": {
        "path": "data/RE3",
        "description": "RE3 數據集（包含 RE3-OB, RE3-SS, RE3-TT）",
        "sub_datasets": ["RE3-OB", "RE3-SS", "RE3-TT"]
    },
    
    # 合成數據集
    "syn_circa": {
        "path": "data/syn_circa",
        "description": "CIRCA 合成數據集",
    },
    "syn_rcd": {
        "path": "data/syn_rcd",
        "description": "RCD 合成數據集",
    },
    "syn_causil": {
        "path": "data/syn_causil",
        "description": "CauSIL 合成數據集",
    },
    
    # 多源數據集
    "multi-source": {
        "path": "data/multi-source-data",
        "description": "多源遙測數據樣本",
    },
}


def check_dataset_exists(dataset_path: str) -> tuple[bool, int]:
    """檢查數據集是否已存在並返回文件數量"""
    path = Path(dataset_path)
    if not path.exists():
        return False, 0
    
    # 檢查目錄是否非空
    if path.is_dir():
        file_count = len(list(path.rglob('*')))
        return file_count > 0, file_count
    
    return path.is_file(), 1 if path.is_file() else 0


def check_sub_datasets(base_path: str, sub_datasets: list) -> dict:
    """檢查子數據集的狀態"""
    results = {}
    for sub in sub_datasets:
        sub_path = os.path.join(base_path, sub)
        exists, count = check_dataset_exists(sub_path)
        results[sub] = {"exists": exists, "count": count}
    return results


def main():
    """主函數"""
    print("="*70)
    print("📊 數據集狀態檢查")
    print("="*70)
    
    existing = []
    missing = []
    incomplete = []
    
    for name, config in ALL_DATASETS.items():
        path = config['path']
        exists, file_count = check_dataset_exists(path)
        
        # 如果有子數據集，檢查每個子數據集
        if "sub_datasets" in config:
            sub_results = check_sub_datasets(path, config['sub_datasets'])
            all_exist = all(r["exists"] for r in sub_results.values())
            
            if all_exist:
                total_count = sum(r["count"] for r in sub_results.values())
                print(f"\n✅ {name}: 完整")
                print(f"   路徑: {path}")
                print(f"   描述: {config['description']}")
                print(f"   文件數: {total_count}")
                for sub, result in sub_results.items():
                    print(f"     - {sub}: {result['count']} 個文件/目錄")
                existing.append(name)
            else:
                print(f"\n⚠️  {name}: 不完整")
                print(f"   路徑: {path}")
                print(f"   描述: {config['description']}")
                for sub, result in sub_results.items():
                    status = "✅" if result["exists"] else "❌"
                    print(f"     {status} {sub}: {result['count']} 個文件/目錄" if result["exists"] else f"     {status} {sub}: 缺失")
                incomplete.append(name)
        else:
            if exists:
                print(f"\n✅ {name}: 完整")
                print(f"   路徑: {path}")
                print(f"   描述: {config['description']}")
                print(f"   文件數: {file_count}")
                existing.append(name)
            else:
                print(f"\n❌ {name}: 缺失")
                print(f"   路徑: {path}")
                print(f"   描述: {config['description']}")
                missing.append(name)
    
    # 總結
    print("\n" + "="*70)
    print("📋 總結")
    print("="*70)
    print(f"✅ 完整: {len(existing)} 個")
    if existing:
        for name in existing:
            print(f"   - {name}")
    
    print(f"\n⚠️  不完整: {len(incomplete)} 個")
    if incomplete:
        for name in incomplete:
            print(f"   - {name}")
    
    print(f"\n❌ 缺失: {len(missing)} 個")
    if missing:
        for name in missing:
            print(f"   - {name}")
    
    print("="*70)
    
    if missing or incomplete:
        print("\n💡 下載建議:")
        if missing:
            print(f"   下載缺失的數據集:")
            print(f"   python download_all_datasets.py --datasets {','.join(missing)}")
        if incomplete:
            print(f"   重新下載不完整的數據集:")
            print(f"   python download_all_datasets.py --datasets {','.join(incomplete)} --no-skip-existing")
    else:
        print("\n🎉 所有數據集都已完整下載！")
    
    return len(missing) == 0 and len(incomplete) == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

