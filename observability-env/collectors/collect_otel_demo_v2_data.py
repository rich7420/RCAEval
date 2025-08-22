#!/usr/bin/env python3
"""
OpenTelemetry Demo v2.0.0+ 數據收集器
收集完整的微服務觀測性數據並轉換為 RE2 格式
"""

import sys
import os
import time
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# OpenTelemetry Demo v2.0.0+ 微服務列表
OTEL_DEMO_SERVICES = [
    'accountingservice',
    'adservice', 
    'cartservice',
    'checkoutservice',
    'currencyservice',
    'emailservice',
    'featureflagservice',
    'frontend',
    'loadgenerator',
    'paymentservice',
    'productcatalogservice',
    'quoteservice',
    'recommendationservice',
    'shippingservice',
    'frontendproxy'
]

def check_services_health():
    """檢查所有服務的健康狀態"""
    print("🔍 檢查服務健康狀態...")
    
    health_checks = {
        'Frontend': 'http://localhost:8080',
        'Prometheus': 'http://localhost:9090/-/healthy',
        'Jaeger': 'http://localhost:16686',
        'OpenTelemetry Collector': 'http://localhost:13133'
    }
    
    healthy_services = []
    
    for service_name, url in health_checks.items():
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"  ✅ {service_name}: 健康")
                healthy_services.append(service_name)
            else:
                print(f"  ⚠️ {service_name}: HTTP {response.status_code}")
        except Exception as e:
            print(f"  ❌ {service_name}: {e}")
    
    return healthy_services

def collect_prometheus_metrics(start_time: datetime, end_time: datetime) -> pd.DataFrame:
    """收集 Prometheus 指標數據"""
    print("📊 收集 Prometheus 指標數據...")
    
    # 針對 OpenTelemetry Demo v2.0.0+ 的指標查詢
    queries = {
        'http_requests_total': 'rate(http_requests_total[1m])',
        'http_request_duration': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[1m]))',
        'process_cpu_seconds': 'rate(process_cpu_seconds_total[1m])',
        'process_memory_bytes': 'process_resident_memory_bytes',
        'go_memstats_alloc_bytes': 'go_memstats_alloc_bytes',
        'go_goroutines': 'go_goroutines',
        'service_up': 'up'
    }
    
    all_metrics = []
    start_ts = int(start_time.timestamp())
    end_ts = int(end_time.timestamp())
    
    for metric_name, query in queries.items():
        try:
            print(f"  🔍 查詢指標: {metric_name}")
            
            params = {
                'query': query,
                'start': start_ts,
                'end': end_ts,
                'step': '15s'
            }
            
            response = requests.get('http://localhost:9090/api/v1/query_range', 
                                  params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if data['status'] == 'success' and data['data']['result']:
                    print(f"    ✅ 找到 {len(data['data']['result'])} 個時間序列")
                    
                    for result in data['data']['result']:
                        metric_labels = result['metric']
                        values = result['values']
                        
                        service = (metric_labels.get('job') or 
                                 metric_labels.get('service_name') or
                                 metric_labels.get('instance', '').split(':')[0] or
                                 'unknown')
                        
                        for timestamp, value in values:
                            try:
                                all_metrics.append({
                                    'timestamp': pd.to_datetime(float(timestamp), unit='s'),
                                    'service': service,
                                    'metric_name': metric_name,
                                    'value': float(value),
                                    'labels': json.dumps(metric_labels, sort_keys=True)
                                })
                            except (ValueError, TypeError):
                                continue
                                
                else:
                    print(f"    ⚠️ {metric_name}: 無數據")
                    
            else:
                print(f"    ❌ {metric_name}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"    ❌ {metric_name}: {e}")
            continue
    
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        print(f"✅ 收集到 {len(df)} 條指標記錄")
        return df
    else:
        print("⚠️ 未收集到任何指標")
        return pd.DataFrame()

def convert_to_re2_format(metrics_df: pd.DataFrame, output_dir: str):
    """轉換為 RE2 格式"""
    print(f"📤 轉換為 RE2 格式並導出到 {output_dir}...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    exported_files = {}
    
    # 轉換 metrics 為 RE2 寬表格式
    if not metrics_df.empty:
        print("  📊 轉換 metrics...")
        
        metrics_df['time'] = metrics_df['timestamp'].astype(int) // 10**9
        metrics_df['metric_column'] = metrics_df['service'] + '_' + metrics_df['metric_name']
        
        metrics_wide = metrics_df.pivot_table(
            index='time',
            columns='metric_column',
            values='value',
            aggfunc='mean'
        ).fillna(0).reset_index()
        
        metrics_path = output_path / 'metrics.csv'
        metrics_wide.to_csv(metrics_path, index=False)
        exported_files['metrics'] = str(metrics_path)
        print(f"    ✅ metrics.csv: {len(metrics_wide)} 行, {len(metrics_wide.columns)-1} 個指標")
    
    # 創建空的其他文件以符合 RE2 格式
    empty_logs = pd.DataFrame(columns=['time', 'timestamp', 'container_name', 'message', 'level', 'req_path', 'error', 'cluster_id', 'log_template'])
    logs_path = output_path / 'logs.csv'
    empty_logs.to_csv(logs_path, index=False)
    exported_files['logs'] = str(logs_path)
    
    empty_logts = pd.DataFrame(columns=['time'])
    logts_path = output_path / 'logts.csv'
    empty_logts.to_csv(logts_path, index=False)
    exported_files['logts'] = str(logts_path)
    
    empty_traces = pd.DataFrame(columns=['trace_id', 'span_id', 'parent_span_id', 'service', 'operation', 'start_time', 'duration_ms', 'error'])
    traces_path = output_path / 'traces.csv'
    empty_traces.to_csv(traces_path, index=False)
    exported_files['traces'] = str(traces_path)
    
    empty_err = pd.DataFrame(columns=['time'])
    tracets_err_path = output_path / 'tracets_err.csv'
    empty_err.to_csv(tracets_err_path, index=False)
    exported_files['tracets_err'] = str(tracets_err_path)
    
    empty_lat = pd.DataFrame(columns=['time'])
    tracets_lat_path = output_path / 'tracets_lat.csv'
    empty_lat.to_csv(tracets_lat_path, index=False)
    exported_files['tracets_lat'] = str(tracets_lat_path)
    
    # 創建 cluster_info.json
    all_services = list(metrics_df['service'].unique()) if not metrics_df.empty else []
    
    cluster_info = {
        "services": sorted(all_services),
        "microservices": [s for s in all_services if s in OTEL_DEMO_SERVICES],
        "metrics_count": len(metrics_df) if not metrics_df.empty else 0,
        "logs_count": 0,
        "traces_count": 0,
        "collection_time": datetime.now().isoformat(),
        "demo_version": "2.0.0+",
        "format": "RE2-compatible"
    }
    
    cluster_info_path = output_path / 'cluster_info.json'
    with open(cluster_info_path, 'w') as f:
        json.dump(cluster_info, f, indent=2)
    exported_files['cluster_info'] = str(cluster_info_path)
    
    return exported_files

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='OpenTelemetry Demo v2.0.0+ 數據收集器')
    parser.add_argument('--duration', type=int, default=5, help='數據收集時間長度 (分鐘)')
    parser.add_argument('--output', type=str, default='otel_demo_v2_data', help='輸出目錄')
    
    args = parser.parse_args()
    
    print("🚀 OpenTelemetry Demo v2.0.0+ 數據收集器")
    print("=" * 60)
    
    # 健康檢查
    healthy_services = check_services_health()
    
    # 設置時間範圍
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=args.duration)
    
    print(f"⏰ 收集時間範圍: {start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')}")
    
    # 收集數據
    metrics_df = collect_prometheus_metrics(start_time, end_time)
    
    # 轉換為 RE2 格式
    exported_files = convert_to_re2_format(metrics_df, args.output)
    
    print(f"🎉 數據收集和轉換完成！")
    print(f"📁 輸出目錄: {Path(args.output).absolute()}")
    
    total_size = 0
    for file_type, file_path in exported_files.items():
        if Path(file_path).exists():
            file_size = Path(file_path).stat().st_size
            total_size += file_size
            print(f"  {file_type}: {Path(file_path).name} ({file_size:,} bytes)")
    
    print(f"💾 總文件大小: {total_size:,} bytes")

if __name__ == "__main__":
    main()