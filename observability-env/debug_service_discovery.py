#!/usr/bin/env python3
"""
Debug script to see what services are actually discovered from the observability backends.
"""

import sys
import os
import logging

# Add config to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'config'))

from config import ServiceTargetingSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def debug_service_discovery():
    """Debug what services are actually discovered."""
    print("🔍 Debugging Service Discovery")
    print("=" * 50)
    
    targeting_system = ServiceTargetingSystem()
    
    # Get discovered services
    services = targeting_system._get_cached_services()
    
    print(f"\n📊 Discovered {len(services)} services:")
    print("-" * 50)
    
    for service in services:
        print(f"🔹 Service: {service.name}")
        print(f"   Status: {service.status.value}")
        print(f"   Endpoints: {service.endpoints}")
        print(f"   Metrics: {service.metrics}")
        print(f"   Source: {service.metadata.get('source', 'unknown')}")
        print()
    
    # Show what Docker containers are running
    print("🐳 Docker Containers Running:")
    print("-" * 50)
    
    import subprocess
    try:
        result = subprocess.run(['docker', 'ps', '--format', 'table {{.Names}}\t{{.Image}}\t{{.Status}}'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("Failed to get docker containers")
    except Exception as e:
        print(f"Error getting docker containers: {e}")
    
    # Test specific service queries
    print("🔍 Testing Prometheus Queries:")
    print("-" * 50)
    
    import requests
    
    prometheus_url = "http://localhost:9090"
    test_queries = [
        'up',
        '{__name__=~".+"}',
        '{job=~".+"}',
        '{service=~".+"}',
        '{container=~".+"}',
        'container_cpu_usage_seconds_total',
        'process_cpu_seconds_total'
    ]
    
    for query in test_queries:
        try:
            response = requests.get(f"{prometheus_url}/api/v1/query", 
                                  params={'query': query}, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success':
                    results = data['data']['result']
                    print(f"✅ {query}: {len(results)} results")
                    
                    # Show first few results
                    for i, result in enumerate(results[:3]):
                        metric = result['metric']
                        print(f"   {i+1}. {metric}")
                else:
                    print(f"❌ {query}: {data.get('error', 'Unknown error')}")
            else:
                print(f"❌ {query}: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ {query}: {e}")
    
    return services


if __name__ == "__main__":
    debug_service_discovery()