#!/usr/bin/env python3
"""
專門用於從 start-demo.sh 啟動的微服務收集觀測性數據的腳本
支持 OpenTelemetry Demo 和 Online Boutique 應用
"""

import sys
import os
import time
import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# 添加 collectors 模組到路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_collector import ObservabilityDataCollector

def check_services():
    """檢查觀測性服務狀態"""
    services = {
        'Prometheus': 'http://localhost:9090/api/v1/query?query=up',
        'Loki': 'http://localhost:3100/ready',
        'Jaeger': 'http://localhost:16686/api/services'
    }
    
    print("🔍 檢查觀測性服務狀態...")
    all_ok = True
    
    for service_name, url in services.items():
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"✅ {service_name}: 運行正常")
            else:
                print(f"❌ {service_name}: HTTP {response.status_code}")
                all_ok = False
        except Exception as e:
            print(f"❌ {service_name}: 連接失敗 - {e}")
            all_ok = False
    
    return all_ok

def check_microservices():
    """檢查微服務應用狀態"""
    apps = {
        'OpenTelemetry Demo': 'http://localhost:8080',
        'Online Boutique': 'http://localhost:8084'
    }
    
    print("\n🚀 檢查微服務應用狀態...")
    running_apps = []
    
    for app_name, url in apps.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {app_name}: 運行正常")
                running_apps.append(app_name)
            else:
                print(f"⚠️ {app_name}: HTTP {response.status_code}")
        except Exception as e:
            print(f"⚠️ {app_name}: 未運行")
    
    return running_apps

def generate_traffic(running_apps, duration_minutes=5):
    """使用 Task 4 的流量生成系統產生觀測性數據"""
    print(f"\n🔄 使用 Task 4 流量生成系統生成 {duration_minutes} 分鐘的流量...")
    
    import subprocess
    import os
    
    # 獲取當前目錄
    current_dir = os.getcwd()
    traffic_dir = os.path.join(os.path.dirname(current_dir), 'traffic')
    
    processes = []
    
    for app_name in running_apps:
        try:
            if 'OpenTelemetry' in app_name:
                print(f"  🎯 為 {app_name} 啟動 Locust 流量生成...")
                
                # 使用 Locust 生成流量
                locust_dir = os.path.join(traffic_dir, 'locust')
                if os.path.exists(locust_dir):
                    cmd = [
                        'python3', '-m', 'locust',
                        '-f', 'otel_demo_users.py',
                        '--host=http://localhost:8080',
                        '--users=10',
                        '--spawn-rate=2',
                        '--headless',
                        f'--run-time={duration_minutes}m'
                    ]
                    
                    process = subprocess.Popen(
                        cmd,
                        cwd=locust_dir,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    processes.append(process)
                    print(f"    ✅ Locust 進程已啟動 (PID: {process.pid})")
                
                # 同時使用行為模擬器
                behavior_dir = os.path.join(traffic_dir, 'behavior')
                if os.path.exists(behavior_dir):
                    cmd = [
                        'python3', 'run_behavior_simulation.py',
                        '--profile', 'otel_demo_realistic',
                        '--users', '5',
                        '--duration', str(duration_minutes * 60),
                        '--spawn-rate', '1'
                    ]
                    
                    process = subprocess.Popen(
                        cmd,
                        cwd=behavior_dir,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    processes.append(process)
                    print(f"    ✅ 行為模擬器已啟動 (PID: {process.pid})")
                    
            elif 'Boutique' in app_name:
                print(f"  🎯 為 {app_name} 啟動 Locust 流量生成...")
                
                locust_dir = os.path.join(traffic_dir, 'locust')
                if os.path.exists(locust_dir):
                    cmd = [
                        'python3', '-m', 'locust',
                        '-f', 'online_boutique_users.py',
                        '--host=http://localhost:8084',
                        '--users=8',
                        '--spawn-rate=2',
                        '--headless',
                        f'--run-time={duration_minutes}m'
                    ]
                    
                    process = subprocess.Popen(
                        cmd,
                        cwd=locust_dir,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    processes.append(process)
                    print(f"    ✅ Locust 進程已啟動 (PID: {process.pid})")
                    
        except Exception as e:
            print(f"    ⚠️ 啟動 {app_name} 流量生成失敗: {e}")
            continue
    
    if processes:
        print(f"⏳ 等待流量生成完成 ({duration_minutes} 分鐘)...")
        print("    💡 流量生成器正在後台運行，同時進行數據收集...")
        
        # 等待一段時間讓流量開始生成
        time.sleep(30)
        print("    ✅ 流量生成已開始，繼續數據收集...")
        
        # 返回進程列表以便後續清理
        return processes
    else:
        print("⚠️ 未能啟動任何流量生成器")
        return []

def cleanup_traffic_processes(processes):
    """清理流量生成進程"""
    if not processes:
        return
        
    print("\n🧹 清理流量生成進程...")
    for process in processes:
        try:
            if process.poll() is None:  # 進程仍在運行
                process.terminate()
                process.wait(timeout=5)
                print(f"    ✅ 進程 {process.pid} 已終止")
        except Exception as e:
            print(f"    ⚠️ 清理進程 {process.pid} 失敗: {e}")
            try:
                process.kill()
            except:
                pass

def collect_data(duration_minutes=10, output_dir="microservices_data", generate_traffic_flag=True):
    """收集觀測性數據"""
    print(f"\n📊 開始收集 {duration_minutes} 分鐘的觀測性數據...")
    
    # 初始化收集器
    collector = ObservabilityDataCollector()
    
    # 設置時間範圍
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=duration_minutes)
    
    print(f"⏰ 收集時間範圍: {start_time.strftime('%Y-%m-%d %H:%M:%S')} - {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 收集數據
        print("🔄 正在收集數據...")
        data = collector.collect_all_data(start_time, end_time)
        
        # 顯示收集結果
        print("\n📈 收集結果:")
        total_records = 0
        for data_type, df in data.items():
            if not df.empty:
                print(f"  ✅ {data_type.capitalize()}: {len(df)} 條記錄")
                total_records += len(df)
                
                # 顯示服務信息
                if 'service' in df.columns:
                    services = df['service'].unique()
                    print(f"     服務: {', '.join(services[:5])}" + ("..." if len(services) > 5 else ""))
            else:
                print(f"  ⚠️ {data_type.capitalize()}: 無數據")
        
        print(f"\n📊 總計收集: {total_records} 條記錄")
        
        if total_records == 0:
            print("⚠️ 未收集到任何數據，可能需要:")
            print("   1. 等待更長時間讓應用生成數據")
            print("   2. 訪問應用 URL 生成流量")
            print("   3. 檢查觀測性服務配置")
            return False
        
        # 生成分析
        print("\n🔍 生成數據分析...")
        analysis = collector.generate_comprehensive_analysis(data)
        
        # 創建輸出目錄
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 導出數據
        print("\n📤 導出數據...")
        experiment_name = f"microservices_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exported_files = collector.export_data_for_re2(data, output_path, experiment_name)
        
        print("✅ 數據導出完成:")
        total_size = 0
        for file_type, file_path in exported_files.items():
            if Path(file_path).exists():
                file_size = Path(file_path).stat().st_size
                total_size += file_size
                print(f"  📄 {file_type}: {Path(file_path).name} ({file_size:,} bytes)")
        
        # 保存分析報告
        analysis_file = output_path / f"{experiment_name}_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        
        analysis_size = analysis_file.stat().st_size
        total_size += analysis_size
        print(f"  📊 分析報告: {analysis_file.name} ({analysis_size:,} bytes)")
        
        print(f"\n💾 總文件大小: {total_size:,} bytes")
        print(f"📁 輸出目錄: {output_path.absolute()}")
        
        # 顯示關鍵分析結果
        if analysis:
            print("\n📈 關鍵分析結果:")
            
            if analysis.get('metrics_analysis'):
                ma = analysis['metrics_analysis']
                print(f"  📊 Metrics: {ma.get('unique_services', 0)} 個服務, "
                      f"{ma.get('unique_metrics', 0)} 種指標, "
                      f"質量分數: {ma.get('quality_score', 0):.2f}")
            
            if analysis.get('logs_analysis'):
                la = analysis['logs_analysis']
                print(f"  📝 Logs: {la.get('unique_services', 0)} 個服務, "
                      f"{la.get('templates_count', 0)} 個模板, "
                      f"{la.get('clusters_count', 0)} 個聚類")
            
            if analysis.get('traces_analysis'):
                ta = analysis['traces_analysis']
                print(f"  🔍 Traces: {ta.get('total_traces', 0)} 個追蹤, "
                      f"{ta.get('total_spans', 0)} 個 span, "
                      f"錯誤率: {ta.get('error_rate', 0):.2%}")
            
            if analysis.get('cross_data_analysis', {}).get('service_coverage'):
                sc = analysis['cross_data_analysis']['service_coverage']
                print(f"  🔗 服務覆蓋: {sc.get('total_services', 0)} 個服務, "
                      f"{sc.get('in_all_sources', 0)} 個完整覆蓋")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 數據收集失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='從微服務收集觀測性數據')
    parser.add_argument('--duration', type=int, default=10, help='數據收集時間長度 (分鐘)')
    parser.add_argument('--output', type=str, default='microservices_data', help='輸出目錄')
    parser.add_argument('--traffic', action='store_true', help='生成流量')
    parser.add_argument('--traffic-duration', type=int, default=5, help='流量生成時間 (分鐘)')
    parser.add_argument('--skip-checks', action='store_true', help='跳過服務檢查')
    
    args = parser.parse_args()
    
    print("🚀 Task 6 微服務數據收集器")
    print("=" * 50)
    
    # 檢查服務狀態
    if not args.skip_checks:
        if not check_services():
            print("\n❌ 觀測性服務未正常運行")
            print("💡 請先運行: ./scripts/start-demo.sh")
            return False
        
        running_apps = check_microservices()
        if not running_apps:
            print("\n❌ 未檢測到運行中的微服務應用")
            print("💡 請先運行: ./scripts/start-demo.sh")
            return False
        
        print(f"\n✅ 檢測到運行中的應用: {', '.join(running_apps)}")
    
    # 生成流量 (如果需要)
    traffic_processes = []
    if args.traffic:
        if not args.skip_checks:
            traffic_processes = generate_traffic(running_apps, args.traffic_duration)
        else:
            print(f"⏳ 等待 {args.traffic_duration} 分鐘以生成數據...")
            time.sleep(args.traffic_duration * 60)
    
    # 收集數據
    try:
        success = collect_data(args.duration, args.output, args.traffic)
        
        if success:
            print("\n🎉 數據收集完成！")
            print("\n💡 後續操作:")
            print("  📊 查看 Grafana: http://localhost:3000")
            print("  📈 查看 Prometheus: http://localhost:9090")
            print("  🔍 查看 Jaeger: http://localhost:16686")
            print("  🔄 重新收集: python3 collect_microservices_data.py")
            return True
        else:
            print("\n❌ 數據收集失敗")
            print("\n💡 故障排除:")
            print("  1. 檢查服務狀態: docker-compose ps")
            print("  2. 查看服務日誌: docker-compose logs")
            print("  3. 訪問應用生成數據: http://localhost:8080 或 http://localhost:8084")
            print("  4. 等待更長時間: --duration 20")
            return False
    finally:
        # 清理流量生成進程
        cleanup_traffic_processes(traffic_processes)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)