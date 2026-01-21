#!/usr/bin/env python3
"""
下載所有 RCAEval 實驗所需的數據集

此腳本會自動下載以下數據集：
- 基礎數據集：online-boutique, sock-shop-1, sock-shop-2, train-ticket
- RE1 系列：RE1-OB, RE1-SS, RE1-TT
- RE2 系列：RE2-OB, RE2-SS, RE2-TT
- RE3 系列：RE3-OB, RE3-SS, RE3-TT
- 合成數據集：syn_circa, syn_rcd, syn_causil
- 多源數據集：multi-source-data

使用方法：
    python download_all_datasets.py [--skip-existing] [--datasets DATASET1,DATASET2,...]
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Optional

# 添加項目路徑
sys.path.insert(0, str(Path(__file__).parent))

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
        download_syn_causil_dataset,
        download_multi_source_sample,
    )
except ImportError as e:
    print(f"❌ 無法導入下載函數: {e}")
    print("請確保已安裝 RCAEval 包：pip install -e .")
    sys.exit(1)


# 定義所有數據集及其下載函數
ALL_DATASETS = {
    # 基礎數據集
    "online-boutique": {
        "func": download_online_boutique_dataset,
        "path": "data/online-boutique",
        "description": "Online Boutique 微服務系統",
        "size": "~100MB",
    },
    "sock-shop-1": {
        "func": download_sock_shop_1_dataset,
        "path": "data/sock-shop-1",
        "description": "Sock Shop v1 微服務系統",
        "size": "~100MB",
    },
    "sock-shop-2": {
        "func": download_sock_shop_2_dataset,
        "path": "data/sock-shop-2",
        "description": "Sock Shop v2 微服務系統",
        "size": "~100MB",
    },
    "train-ticket": {
        "func": download_train_ticket_dataset,
        "path": "data/train-ticket",
        "description": "Train Ticket 微服務系統",
        "size": "~100MB",
    },
    
    # RE1 系列（包含 3 個子數據集：OB, SS, TT）
    "re1": {
        "func": download_re1_dataset,
        "path": "data/RE1",
        "description": "RE1 數據集（包含 RE1-OB, RE1-SS, RE1-TT）",
        "size": "~390MB",
    },
    
    # RE2 系列（包含 3 個子數據集：OB, SS, TT）
    "re2": {
        "func": download_re2_dataset,
        "path": "data/RE2",
        "description": "RE2 數據集（包含 RE2-OB, RE2-SS, RE2-TT）",
        "size": "~4.2GB",
    },
    
    # RE3 系列（包含 3 個子數據集：OB, SS, TT）
    "re3": {
        "func": download_re3_dataset,
        "path": "data/RE3",
        "description": "RE3 數據集（包含 RE3-OB, RE3-SS, RE3-TT）",
        "size": "~534MB",
    },
    
    # 合成數據集
    "syn_circa": {
        "func": download_syn_circa_dataset,
        "path": "data/syn_circa",
        "description": "CIRCA 合成數據集",
        "size": "~50MB",
    },
    "syn_rcd": {
        "func": download_syn_rcd_dataset,
        "path": "data/syn_rcd",
        "description": "RCD 合成數據集",
        "size": "~50MB",
    },
    "syn_causil": {
        "func": download_syn_causil_dataset,
        "path": "data/syn_causil",
        "description": "CauSIL 合成數據集",
        "size": "~50MB",
    },
    
    # 多源數據集
    "multi-source": {
        "func": download_multi_source_sample,
        "path": "data/multi-source-data",
        "description": "多源遙測數據樣本",
        "size": "~10MB",
    },
}


def check_dataset_exists(dataset_path: str) -> bool:
    """檢查數據集是否已存在"""
    path = Path(dataset_path)
    if not path.exists():
        return False
    
    # 檢查目錄是否非空
    if path.is_dir():
        return len(list(path.iterdir())) > 0
    
    return path.is_file()


def download_dataset(name: str, config: dict, skip_existing: bool = True) -> bool:
    """下載單個數據集"""
    print(f"\n{'='*70}")
    print(f"📦 下載數據集: {name}")
    print(f"   描述: {config['description']}")
    print(f"   大小: {config['size']}")
    print(f"   路徑: {config['path']}")
    print(f"{'='*70}")
    
    # 檢查是否已存在
    if skip_existing and check_dataset_exists(config['path']):
        print(f"✅ 數據集已存在，跳過下載")
        return True
    
    try:
        start_time = time.time()
        print(f"⏳ 開始下載...")
        
        # 執行下載
        config['func']()
        
        download_time = time.time() - start_time
        
        # 驗證下載結果
        if check_dataset_exists(config['path']):
            print(f"✅ 下載成功！耗時: {download_time:.1f} 秒")
            return True
        else:
            print(f"❌ 下載完成但驗證失敗，請檢查路徑: {config['path']}")
            return False
            
    except Exception as e:
        print(f"❌ 下載失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="下載所有 RCAEval 實驗所需的數據集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 下載所有數據集（跳過已存在的）
  python download_all_datasets.py
  
  # 強制重新下載所有數據集
  python download_all_datasets.py --no-skip-existing
  
  # 只下載特定數據集
  python download_all_datasets.py --datasets re1,re2,re3
  
  # 下載基礎數據集和 RE 系列
  python download_all_datasets.py --datasets online-boutique,sock-shop-1,re1,re2
        """
    )
    
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="跳過已存在的數據集（默認：True）"
    )
    
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="強制重新下載所有數據集"
    )
    
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="指定要下載的數據集（逗號分隔），例如：re1,re2,re3。如果不指定，則下載所有數據集"
    )
    
    args = parser.parse_args()
    
    # 確定要下載的數據集列表
    if args.datasets:
        dataset_names = [name.strip() for name in args.datasets.split(",")]
        # 驗證數據集名稱
        invalid_names = [name for name in dataset_names if name not in ALL_DATASETS]
        if invalid_names:
            print(f"❌ 無效的數據集名稱: {', '.join(invalid_names)}")
            print(f"可用的數據集: {', '.join(ALL_DATASETS.keys())}")
            sys.exit(1)
    else:
        dataset_names = list(ALL_DATASETS.keys())
    
    # 打印標題
    print("="*70)
    print("🚀 RCAEval 數據集下載工具")
    print("="*70)
    print(f"📋 將下載 {len(dataset_names)} 個數據集")
    print(f"⚙️  跳過已存在: {args.skip_existing}")
    print(f"📦 數據集列表: {', '.join(dataset_names)}")
    print("="*70)
    
    # 統計信息
    total_datasets = len(dataset_names)
    successful = []
    failed = []
    skipped = []
    
    # 下載每個數據集
    for i, name in enumerate(dataset_names, 1):
        config = ALL_DATASETS[name]
        
        # 檢查是否已存在
        if args.skip_existing and check_dataset_exists(config['path']):
            print(f"\n[{i}/{total_datasets}] ✅ {name}: 已存在，跳過")
            skipped.append(name)
            continue
        
        print(f"\n[{i}/{total_datasets}] 正在處理: {name}")
        
        if download_dataset(name, config, args.skip_existing):
            successful.append(name)
        else:
            failed.append(name)
    
    # 打印總結
    print("\n" + "="*70)
    print("📊 下載總結")
    print("="*70)
    print(f"✅ 成功: {len(successful)} 個")
    if successful:
        for name in successful:
            print(f"   - {name}")
    
    print(f"\n⏭️  跳過: {len(skipped)} 個")
    if skipped:
        for name in skipped:
            print(f"   - {name}")
    
    print(f"\n❌ 失敗: {len(failed)} 個")
    if failed:
        for name in failed:
            print(f"   - {name}")
        print("\n💡 提示: 失敗的數據集可以稍後單獨重新下載")
    
    print("="*70)
    
    # 計算總大小估算（只顯示成功下載的）
    downloaded_sizes = [
        config['size'] for name, config in ALL_DATASETS.items()
        if name in successful
    ]
    if downloaded_sizes:
        print(f"\n💾 本次下載的數據集大小: {', '.join(downloaded_sizes)}")
    
    # 使用建議
    if successful or skipped:
        available = successful + skipped
        print(f"\n🎉 可用數據集 ({len(available)} 個): {', '.join(available)}")
        print("\n💡 使用建議:")
        print("   1. 快速測試: 使用 online-boutique, sock-shop-1")
        print("   2. 標準測試: 使用 re1 系列")
        print("   3. 完整測試: 使用 re1, re2, re3 系列")
        print("\n📝 示例命令:")
        print("   python main.py --method baro --dataset re2-tt")
        print("   python main.py --method baro --dataset online-boutique")
    
    # 返回狀態碼
    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

