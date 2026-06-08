#!/usr/bin/env python3
"""
改进的 RE2 数据集下载脚本
支持重试机制和断点续传
"""

import os
import sys
import time
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

def download_with_retry(url, local_path, max_retries=3, chunk_size=8192, timeout=300):
    """
    带重试机制的下载函数
    """
    for attempt in range(max_retries):
        try:
            print(f"\n尝试 {attempt + 1}/{max_retries}: 下载 {os.path.basename(local_path)}")
            
            # 检查是否已有部分下载的文件
            resume_pos = 0
            if os.path.exists(local_path):
                resume_pos = os.path.getsize(local_path)
                if resume_pos > 0:
                    print(f"  发现已存在的文件 ({resume_pos / 1024 / 1024:.1f} MB)，尝试续传...")
            
            headers = {}
            if resume_pos > 0:
                headers['Range'] = f'bytes={resume_pos}-'
            
            response = requests.get(url, stream=True, headers=headers, timeout=timeout)
            
            # 检查服务器是否支持断点续传
            if resume_pos > 0 and response.status_code == 206:
                mode = 'ab'
                total_size = int(response.headers.get('content-range', '').split('/')[-1])
                initial_pos = resume_pos
            elif resume_pos > 0 and response.status_code == 200:
                print("  服务器不支持断点续传，重新下载...")
                os.remove(local_path)
                mode = 'wb'
                total_size = int(response.headers.get('content-length', 0))
                initial_pos = 0
            else:
                mode = 'wb'
                total_size = int(response.headers.get('content-length', 0))
                initial_pos = 0
            
            if total_size == 0:
                print("  警告: 无法获取文件大小")
            
            with open(local_path, mode) as f:
                with tqdm(
                    desc=f"下载 {os.path.basename(local_path)}",
                    total=total_size,
                    initial=initial_pos,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # 验证下载的文件大小
            downloaded_size = os.path.getsize(local_path)
            if total_size > 0 and downloaded_size != total_size:
                print(f"  警告: 文件大小不匹配 (下载: {downloaded_size}, 预期: {total_size})")
                if attempt < max_retries - 1:
                    print("  将重试...")
                    time.sleep(5)
                    continue
                else:
                    raise Exception(f"下载不完整: {downloaded_size}/{total_size} bytes")
            
            print(f"  ✅ 下载成功: {downloaded_size / 1024 / 1024:.1f} MB")
            return True
            
        except requests.exceptions.Timeout:
            print(f"  ❌ 超时错误 (尝试 {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10
                print(f"  等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                raise
                
        except requests.exceptions.RequestException as e:
            print(f"  ❌ 网络错误: {e} (尝试 {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10
                print(f"  等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                raise
                
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10
                print(f"  等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                raise
    
    return False


def verify_zip_file(zip_path):
    """验证 ZIP 文件是否完整"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 尝试读取文件列表
            file_list = zip_ref.namelist()
            if len(file_list) == 0:
                return False
            # 尝试读取第一个文件的第一个字节
            if len(file_list) > 0:
                zip_ref.read(file_list[0], 1)
            return True
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as e:
        print(f"  ❌ ZIP 文件损坏: {e}")
        return False
    except Exception as e:
        print(f"  ❌ 验证失败: {e}")
        return False


def download_and_extract_re2_dataset():
    """下载并解压 RE2 数据集"""
    base_url = "https://zenodo.org/records/14590730/files"
    datasets = {
        "RE2-OB": f"{base_url}/RE2-OB.zip?download=1",
        "RE2-SS": f"{base_url}/RE2-SS.zip?download=1",
        "RE2-TT": f"{base_url}/RE2-TT.zip?download=1",
    }
    
    local_path = "data/RE2"
    os.makedirs(local_path, exist_ok=True)
    
    print("="*70)
    print("🚀 改进的 RE2 数据集下载工具")
    print("="*70)
    print(f"📁 目标目录: {local_path}")
    print(f"📦 需要下载: {len(datasets)} 个文件")
    print("="*70)
    
    for dataset_name, url in datasets.items():
        dataset_path = os.path.join(local_path, dataset_name)
        
        # 检查是否已存在
        if os.path.exists(dataset_path) and os.listdir(dataset_path):
            print(f"\n✅ {dataset_name}: 已存在，跳过")
            continue
        
        zip_file = f"{dataset_name}.zip"
        print(f"\n{'='*70}")
        print(f"📦 处理: {dataset_name}")
        print(f"{'='*70}")
        
        try:
            # 下载
            if not download_with_retry(url, zip_file, max_retries=5, timeout=600):
                print(f"❌ {dataset_name} 下载失败")
                continue
            
            # 验证 ZIP 文件
            print(f"🔍 验证 ZIP 文件...")
            if not verify_zip_file(zip_file):
                print(f"❌ {dataset_name}.zip 文件损坏，删除并重试...")
                os.remove(zip_file)
                if not download_with_retry(url, zip_file, max_retries=3, timeout=600):
                    print(f"❌ {dataset_name} 重新下载失败")
                    continue
            
            # 解压
            print(f"📂 解压 {zip_file}...")
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(local_path)
                print(f"✅ {dataset_name} 解压成功")
            except Exception as e:
                print(f"❌ 解压失败: {e}")
                continue
            
            # 清理 ZIP 文件
            os.remove(zip_file)
            print(f"🗑️  已删除临时文件: {zip_file}")
            
            # 验证解压结果
            if os.path.exists(dataset_path) and os.listdir(dataset_path):
                print(f"✅ {dataset_name} 验证成功")
            else:
                print(f"❌ {dataset_name} 解压后验证失败")
                
        except Exception as e:
            print(f"❌ {dataset_name} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            # 清理可能损坏的文件
            if os.path.exists(zip_file):
                os.remove(zip_file)
    
    # 最终验证
    print("\n" + "="*70)
    print("📊 最终验证")
    print("="*70)
    
    all_complete = True
    for dataset_name in datasets.keys():
        dataset_path = os.path.join(local_path, dataset_name)
        if os.path.exists(dataset_path) and os.listdir(dataset_path):
            file_count = len(list(Path(dataset_path).rglob('*')))
            print(f"✅ {dataset_name}: 完整 ({file_count} 个文件/目录)")
        else:
            print(f"❌ {dataset_name}: 缺失或不完整")
            all_complete = False
    
    if all_complete:
        print("\n🎉 所有 RE2 数据集下载完成！")
    else:
        print("\n⚠️  部分数据集未完成，请检查网络连接后重试")
    
    return all_complete


if __name__ == "__main__":
    try:
        success = download_and_extract_re2_dataset()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断下载")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

