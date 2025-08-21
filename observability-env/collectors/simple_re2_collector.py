#!/usr/bin/env python3
"""
簡化的 RE2 格式數據收集器
直接生成符合 RE2 標準的 CSV 文件
"""

import pandas as pd
import numpy as np
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import time

def collect_prometheus_data(duration_minutes=5):
    """收集 Prometheus 數據並轉換為 RE2 格式"""
    print("📊 收集 Prometheus 數據...")
    
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=duration_minutes)
    
    # 時間戳轉換
    start_ts = int(start_time.timestamp())
    end_ts = int(end_time.timestamp())
    
    # 擴展查詢以包含微服務指標
    queries = {
        # 基礎進程指標
        'process_cpu_seconds_total': 'rate(process_cpu_seconds_total[1m])',
        'process_resident_memory_bytes': 'process_resident_memory_bytes',
        'process_open_fds': 'process_open_fds',
        'go_memstats_alloc_bytes': 'go_memstats_alloc_bytes',
        'go_goroutines': 'go_goroutines',
        'up': 'up',
        
        # HTTP 指標 (微服務)
        'http_requests_total': 'rate(http_requests_total[1m])',
        'http_request_duration_seconds': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[1m]))',
        'http_requests_in_flight': 'http_requests_in_flight',
        
        # gRPC 指標 (微服務)
        'grpc_server_handled_total': 'rate(grpc_server_handled_total[1m])',
        'grpc_server_handling_seconds': 'histogram_quantile(0.95, rate(grpc_server_handling_seconds_bucket[1m]))',
        
        # 容器指標 (如果可用)
        'container_cpu_usage_seconds_total': 'rate(container_cpu_usage_seconds_total[1m])',
        'container_memory_usage_bytes': 'container_memory_usage_bytes',
        'container_network_receive_bytes_total': 'rate(container_network_receive_bytes_total[1m])',
        'container_network_transmit_bytes_total': 'rate(container_network_transmit_bytes_total[1m])'
    }
    
    all_data = []
    
    for metric_name, query in queries.items():
        try:
            params = {
                'query': query,
                'start': start_ts,
                'end': end_ts,
                'step': '15s'
            }
            
            response = requests.get('http://localhost:9090/api/v1/query_range', 
                                  params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data['status'] == 'success' and data['data']['result']:
                    for result in data['data']['result']:
                        labels = result['metric']
                        values = result['values']
                        
                        # 提取服務名稱
                        service = labels.get('job', labels.get('instance', 'unknown'))
                        
                        for timestamp, value in values:
                            try:
                                all_data.append({
                                    'time': int(float(timestamp)),
                                    'service': service,
                                    'metric': metric_name,
                                    'value': float(value)
                                })
                            except (ValueError, TypeError):
                                continue
                                
        except Exception as e:
            print(f"  ⚠️ {metric_name}: {e}")
            continue
    
    if all_data:
        df = pd.DataFrame(all_data)
        print(f"  ✅ 收集到 {len(df)} 條指標記錄")
        return df
    else:
        print("  ⚠️ 未收集到指標數據")
        return pd.DataFrame()

def collect_loki_data(duration_minutes=5):
    """收集 Loki 數據並轉換為 RE2 格式"""
    print("📝 收集 Loki 數據...")
    
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=duration_minutes)
    
    start_ns = int(start_time.timestamp() * 1000000000)
    end_ns = int(end_time.timestamp() * 1000000000)
    
    try:
        params = {
            'query': '{job=~".+"}',
            'start': start_ns,
            'end': end_ns,
            'limit': 500
        }
        
        response = requests.get('http://localhost:3100/loki/api/v1/query_range', 
                              params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data['status'] == 'success' and data['data']['result']:
                all_logs = []
                
                for stream in data['data']['result']:
                    labels = stream['stream']
                    values = stream['values']
                    
                    service = labels.get('service_name', labels.get('job', 'unknown'))
                    
                    for entry in values:
                        timestamp_ns, message = entry
                        
                        # 解析日誌級別
                        level = 'info'
                        if any(word in message.lower() for word in ['error', 'err']):
                            level = 'error'
                        elif any(word in message.lower() for word in ['warn', 'warning']):
                            level = 'warn'
                        elif 'debug' in message.lower():
                            level = 'debug'
                        
                        timestamp_dt = pd.to_datetime(int(timestamp_ns), unit='ns')
                        
                        all_logs.append({
                            'time': timestamp_dt.strftime('%H:%M'),
                            'timestamp': int(timestamp_ns) // 1000,  # 微秒
                            'container_name': service,
                            'message': message[:200],  # 限制長度
                            'level': level,
                            'req_path': '',
                            'error': '',
                            'cluster_id': 1,
                            'log_template': f"{service} {message[:50]}"
                        })
                
                if all_logs:
                    df = pd.DataFrame(all_logs)
                    print(f"  ✅ 收集到 {len(df)} 條日誌記錄")
                    return df
                    
        print("  ⚠️ 未收集到日誌數據")
        
    except Exception as e:
        print(f"  ❌ 日誌收集失敗: {e}")
    
    return pd.DataFrame()

def create_mock_traces(duration_minutes=5):
    """創建模擬追蹤數據"""
    print("🔍 創建模擬追蹤數據...")
    
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=duration_minutes)
    
    # OpenTelemetry Demo 微服務 + 觀測性基礎設施
    services = [
        'frontend', 'adservice', 'cartservice', 'checkoutservice', 
        'currencyservice', 'emailservice', 'paymentservice', 
        'productcatalogservice', 'recommendationservice', 'shippingservice',
        'loadgenerator', 'redis', 'prometheus', 'jaeger', 'loki', 'otel-collector'
    ]
    operations = [
        'GetAds', 'GetCart', 'AddItem', 'PlaceOrder', 'Convert', 
        'SendEmail', 'Charge', 'GetProduct', 'ListRecommendations', 
        'GetQuote', 'GenerateLoad', 'query', 'insert', 'health_check', 'export'
    ]
    
    traces = []
    trace_id = 1
    
    # 每30秒創建一批追蹤
    current_time = start_time
    while current_time < end_time:
        for service in services:
            for operation in operations:
                if trace_id % 2 == 0:  # 50% 的追蹤
                    traces.append({
                        'trace_id': f'trace_{trace_id:06d}',
                        'span_id': f'span_{trace_id:06d}',
                        'parent_span_id': '',
                        'service': service,
                        'operation': operation,
                        'start_time': current_time.isoformat(),
                        'duration_ms': np.random.randint(5, 200),
                        'error': trace_id % 15 == 0  # ~6.7% 錯誤率
                    })
                trace_id += 1
        
        current_time += timedelta(seconds=30)
    
    if traces:
        df = pd.DataFrame(traces)
        print(f"  ✅ 創建了 {len(df)} 條追蹤記錄")
        return df
    
    return pd.DataFrame()

def convert_to_re2_format(metrics_df, logs_df, traces_df, output_dir):
    """轉換為 RE2 格式並保存"""
    print(f"📤 轉換為 RE2 格式...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 1. metrics.csv - 寬表格式
    if not metrics_df.empty:
        print("  📊 處理 metrics...")
        
        # 創建寬表格式
        metrics_pivot = metrics_df.pivot_table(
            index='time',
            columns=['service', 'metric'],
            values='value',
            aggfunc='mean'
        ).fillna(0)
        
        # 扁平化列名
        metrics_pivot.columns = [f"{service}_{metric}" for service, metric in metrics_pivot.columns]
        metrics_pivot = metrics_pivot.reset_index()
        
        metrics_path = output_path / 'metrics.csv'
        metrics_pivot.to_csv(metrics_path, index=False)
        print(f"    ✅ metrics.csv: {len(metrics_pivot)} 行, {len(metrics_pivot.columns)-1} 列")
    else:
        # 空文件
        pd.DataFrame(columns=['time']).to_csv(output_path / 'metrics.csv', index=False)
    
    # 2. logs.csv
    if not logs_df.empty:
        print("  📝 處理 logs...")
        logs_path = output_path / 'logs.csv'
        logs_df.to_csv(logs_path, index=False)
        print(f"    ✅ logs.csv: {len(logs_df)} 行")
        
        # 3. logts.csv - 時間序列格式
        logs_df['time_unix'] = pd.to_datetime(logs_df['timestamp'], unit='us').astype(int) // 10**9
        logs_df['service_cluster'] = logs_df['container_name'] + '_1'
        
        logts_counts = logs_df.groupby(['time_unix', 'service_cluster']).size().reset_index(name='count')
        logts_pivot = logts_counts.pivot(index='time_unix', columns='service_cluster', values='count').fillna(0)
        logts_pivot = logts_pivot.reset_index().rename(columns={'time_unix': 'time'})
        
        logts_path = output_path / 'logts.csv'
        logts_pivot.to_csv(logts_path, index=False)
        print(f"    ✅ logts.csv: {len(logts_pivot)} 行, {len(logts_pivot.columns)-1} 列")
    else:
        # 空文件
        pd.DataFrame(columns=['time', 'timestamp', 'container_name', 'message', 'level', 'req_path', 'error', 'cluster_id', 'log_template']).to_csv(output_path / 'logs.csv', index=False)
        pd.DataFrame(columns=['time']).to_csv(output_path / 'logts.csv', index=False)
    
    # 4. traces.csv
    if not traces_df.empty:
        print("  🔍 處理 traces...")
        traces_path = output_path / 'traces.csv'
        traces_df.to_csv(traces_path, index=False)
        print(f"    ✅ traces.csv: {len(traces_df)} 行")
        
        # 5. tracets_err.csv - 錯誤追蹤時間序列
        traces_df['time'] = pd.to_datetime(traces_df['start_time']).astype(int) // 10**9
        traces_df['service_operation'] = traces_df['service'] + '_' + traces_df['operation']
        
        error_traces = traces_df[traces_df['error'] == True]
        if not error_traces.empty:
            error_counts = error_traces.groupby(['time', 'service_operation']).size().reset_index(name='count')
            error_pivot = error_counts.pivot(index='time', columns='service_operation', values='count').fillna(0.0)
        else:
            # 創建空的時間序列
            unique_ops = traces_df['service_operation'].unique()
            time_points = traces_df['time'].unique()
            error_pivot = pd.DataFrame(index=time_points, columns=unique_ops).fillna(0.0)
        
        error_pivot = error_pivot.reset_index()
        tracets_err_path = output_path / 'tracets_err.csv'
        error_pivot.to_csv(tracets_err_path, index=False)
        print(f"    ✅ tracets_err.csv: {len(error_pivot)} 行")
        
        # 6. tracets_lat.csv - 高延遲追蹤時間序列
        latency_threshold = traces_df['duration_ms'].quantile(0.95)
        high_latency = traces_df[traces_df['duration_ms'] >= latency_threshold]
        
        if not high_latency.empty:
            lat_counts = high_latency.groupby(['time', 'service_operation']).size().reset_index(name='count')
            lat_pivot = lat_counts.pivot(index='time', columns='service_operation', values='count').fillna(0)
        else:
            unique_ops = traces_df['service_operation'].unique()
            time_points = traces_df['time'].unique()
            lat_pivot = pd.DataFrame(index=time_points, columns=unique_ops).fillna(0)
        
        lat_pivot = lat_pivot.reset_index()
        tracets_lat_path = output_path / 'tracets_lat.csv'
        lat_pivot.to_csv(tracets_lat_path, index=False)
        print(f"    ✅ tracets_lat.csv: {len(lat_pivot)} 行")
    else:
        # 空文件
        empty_traces = pd.DataFrame(columns=['trace_id', 'span_id', 'parent_span_id', 'service', 'operation', 'start_time', 'duration_ms', 'error'])
        empty_traces.to_csv(output_path / 'traces.csv', index=False)
        pd.DataFrame(columns=['time']).to_csv(output_path / 'tracets_err.csv', index=False)
        pd.DataFrame(columns=['time']).to_csv(output_path / 'tracets_lat.csv', index=False)
    
    # 7. cluster_info.json
    services_list = []
    if not metrics_df.empty:
        services_list.extend(metrics_df['service'].unique().tolist())
    if not logs_df.empty:
        services_list.extend(logs_df['container_name'].unique().tolist())
    if not traces_df.empty:
        services_list.extend(traces_df['service'].unique().tolist())
    
    cluster_info = {
        "services": list(set(services_list)),
        "metrics_count": len(metrics_df) if not metrics_df.empty else 0,
        "logs_count": len(logs_df) if not logs_df.empty else 0,
        "traces_count": len(traces_df) if not traces_df.empty else 0,
        "collection_time": datetime.now().isoformat(),
        "format": "RE2-compatible"
    }
    
    cluster_info_path = output_path / 'cluster_info.json'
    with open(cluster_info_path, 'w') as f:
        json.dump(cluster_info, f, indent=2)
    print(f"    ✅ cluster_info.json")
    
    return {
        'metrics': str(output_path / 'metrics.csv'),
        'logs': str(output_path / 'logs.csv'),
        'logts': str(output_path / 'logts.csv'),
        'traces': str(output_path / 'traces.csv'),
        'tracets_err': str(output_path / 'tracets_err.csv'),
        'tracets_lat': str(output_path / 'tracets_lat.csv'),
        'cluster_info': str(cluster_info_path)
    }

def main():
    parser = argparse.ArgumentParser(description='簡化的 RE2 格式數據收集器')
    parser.add_argument('--duration', type=int, default=5, help='數據收集時間 (分鐘)')
    parser.add_argument('--output', type=str, default='re2_data', help='輸出目錄')
    parser.add_argument('--mock-only', action='store_true', help='只生成模擬數據')
    
    args = parser.parse_args()
    
    print("🚀 簡化 RE2 格式數據收集器")
    print("=" * 50)
    
    if args.mock_only:
        print("🎭 只生成模擬數據模式")
        metrics_df = pd.DataFrame()
        logs_df = pd.DataFrame()
        traces_df = create_mock_traces(args.duration)
    else:
        # 收集真實數據
        metrics_df = collect_prometheus_data(args.duration)
        logs_df = collect_loki_data(args.duration)
        traces_df = create_mock_traces(args.duration)
    
    # 轉換並保存
    exported_files = convert_to_re2_format(metrics_df, logs_df, traces_df, args.output)
    
    print(f"\n🎉 數據收集完成！")
    print(f"📁 輸出目錄: {Path(args.output).absolute()}")
    
    total_size = 0
    for file_type, file_path in exported_files.items():
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            total_size += size
            print(f"  📄 {file_type}: {Path(file_path).name} ({size:,} bytes)")
    
    print(f"\n💾 總大小: {total_size:,} bytes")
    print("\n💡 使用方式:")
    print("  1. 檢查生成的文件格式")
    print("  2. 用於 RCA 評估系統")
    print("  3. 與標準 RE2 數據集比較")

if __name__ == "__main__":
    main()