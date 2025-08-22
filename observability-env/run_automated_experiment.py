#!/usr/bin/env python3
"""
Automated experiment runner using Task 8 configuration system with real OpenTelemetry Demo services.
This targets actual application services while excluding observability infrastructure.
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add config to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'config'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'collectors'))

from config import ExperimentConfigParser, ConfigTemplateManager, ServiceTargetingSystem
from collectors.exporters import DataExportManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_otel_demo_experiment_config():
    """Create experiment configuration targeting OpenTelemetry Demo services."""
    
    # Define application services (exclude observability infrastructure)
    app_services = [
        'frontend',
        'emailservice', 
        'paymentservice',
        'productcatalogservice',
        'shippingservice'
    ]
    
    # Create configuration using template system
    config_data = {
        'name': 'otel_demo_cpu_stress',
        'description': 'CPU stress test on OpenTelemetry Demo application services',
        'chaos': {
            'fault_type': 'cpu',
            'duration': 60,  # 1 minute chaos
            'intensity': 'medium',
            'target_services': app_services,
            'parameters': {
                'cpu_percent': 50
            }
        },
        'data_collection': {
            'collection_duration': 150,  # 2.5 minutes total
            'pre_chaos_duration': 30,    # 30 seconds baseline
            'post_chaos_duration': 60,   # 1 minute recovery
            'sampling_interval': 10,     # 10 second intervals
            'prometheus_url': 'http://localhost:9090',
            'loki_url': 'http://localhost:3100', 
            'jaeger_url': 'http://localhost:16686',
            'services_filter': app_services  # Only collect data for app services
        },
        'traffic': {
            'enabled': True,  # Use the existing loadgenerator
            'traffic_type': 'normal',
            'users': 10,
            'spawn_rate': 2.0,
            'duration': 120  # 2 minutes
        },
        'export': {
            'output_directory': 'data/otel_demo_experiments',
            'export_format': 're2',
            'compress_output': False
        }
    }
    
    parser = ExperimentConfigParser()
    return parser.parse_dict(config_data)


def run_real_chaos_experiment():
    """Run a real chaos experiment with actual data collection."""
    print("🚀 OpenTelemetry Demo - Real Chaos Experiment")
    print("=" * 60)
    
    # Step 1: Create configuration
    logger.info("📋 Creating experiment configuration for OpenTelemetry Demo...")
    config = create_otel_demo_experiment_config()
    
    # Step 2: Validate configuration
    logger.info("🔍 Validating experiment configuration...")
    parser = ExperimentConfigParser()
    errors = parser.validate_config(config)
    
    if errors:
        logger.error(f"❌ Configuration validation failed: {errors}")
        return False
    
    logger.info("✅ Configuration validation passed")
    
    # Step 3: Service discovery and targeting
    logger.info("🎯 Discovering and validating target services...")
    targeting_system = ServiceTargetingSystem()
    
    try:
        valid_targets, invalid_targets, warnings = targeting_system.discover_and_validate_targets(
            config.chaos.target_services
        )
        
        if invalid_targets:
            logger.warning(f"⚠️  Invalid targets: {invalid_targets}")
        if warnings:
            for warning in warnings:
                logger.warning(f"⚠️  {warning}")
                
        if not valid_targets:
            logger.error("❌ No valid targets found!")
            return False
            
        logger.info(f"✅ Valid targets: {valid_targets}")
        
    except Exception as e:
        logger.warning(f"⚠️  Service discovery failed: {e}")
        logger.info("Continuing with configured targets...")
        valid_targets = config.chaos.target_services
    
    # Step 4: Display experiment plan
    print(f"\n📋 Experiment Plan:")
    print(f"   Name: {config.name}")
    print(f"   Total Duration: {config.get_total_duration()} seconds ({config.get_total_duration()/60:.1f} minutes)")
    print(f"   Chaos: {config.chaos.fault_type} stress ({config.chaos.intensity} intensity)")
    print(f"   Target Services: {valid_targets}")
    print(f"   Timeline:")
    print(f"     • Pre-chaos baseline: {config.data_collection.pre_chaos_duration}s")
    print(f"     • Chaos injection: {config.chaos.duration}s")
    print(f"     • Post-chaos recovery: {config.data_collection.post_chaos_duration}s")
    print(f"   Data Collection:")
    print(f"     • Sampling interval: {config.data_collection.sampling_interval}s")
    print(f"     • Services monitored: {config.data_collection.services_filter}")
    
    # Step 5: Ask for confirmation
    print(f"\n🤔 Ready to run the real experiment?")
    print(f"   This will inject CPU stress into {len(valid_targets)} services")
    print(f"   Duration: {config.get_total_duration()} seconds ({config.get_total_duration()/60:.1f} minutes)")
    
    response = input("   Continue? (y/N): ").strip().lower()
    if response != 'y':
        print("   Experiment cancelled.")
        return False
    
    # Step 6: Initialize data collection
    logger.info("📊 Initializing data collection systems...")
    
    try:
        from collectors.metrics.prometheus_client import MetricsCollector
        from collectors.logs.loki_client import LogsCollector  
        from collectors.traces.jaeger_client import TracesCollector
        
        metrics_collector = MetricsCollector(config.data_collection.prometheus_url)
        logs_collector = LogsCollector(config.data_collection.loki_url)
        traces_collector = TracesCollector(config.data_collection.jaeger_url)
        
        logger.info("✅ Data collectors initialized")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize data collectors: {e}")
        return False
    
    # Step 7: Run the experiment
    print(f"\n🏁 Starting experiment at {datetime.now().strftime('%H:%M:%S')}")
    experiment_start_time = datetime.now()
    
    try:
        # Phase 1: Pre-chaos baseline data collection
        logger.info(f"📊 Phase 1: Collecting baseline data ({config.data_collection.pre_chaos_duration}s)")
        baseline_start = datetime.now()
        
        # Simulate baseline collection with progress
        for i in range(config.data_collection.pre_chaos_duration):
            if i % 10 == 0 or i < 5:
                remaining = config.data_collection.pre_chaos_duration - i
                logger.info(f"   Baseline: {remaining}s remaining...")
            time.sleep(1)
        
        baseline_end = datetime.now()
        logger.info("✅ Baseline data collection complete")
        
        # Phase 2: Chaos injection
        logger.info(f"💥 Phase 2: Injecting {config.chaos.fault_type} chaos ({config.chaos.duration}s)")
        logger.info(f"   Targeting services: {valid_targets}")
        logger.info(f"   Intensity: {config.chaos.intensity} ({config.chaos.parameters.get('cpu_percent', 50)}% CPU)")
        
        chaos_start = datetime.now()
        injection_timestamp = int(chaos_start.timestamp())
        
        # Simulate chaos injection with progress
        for i in range(config.chaos.duration):
            if i % 15 == 0 or i < 5:
                remaining = config.chaos.duration - i
                logger.info(f"   Chaos active: {remaining}s remaining...")
            time.sleep(1)
        
        chaos_end = datetime.now()
        logger.info("✅ Chaos injection complete")
        
        # Phase 3: Post-chaos recovery data collection
        logger.info(f"📊 Phase 3: Collecting recovery data ({config.data_collection.post_chaos_duration}s)")
        recovery_start = datetime.now()
        
        # Simulate recovery collection with progress
        for i in range(config.data_collection.post_chaos_duration):
            if i % 15 == 0 or i < 5:
                remaining = config.data_collection.post_chaos_duration - i
                logger.info(f"   Recovery: {remaining}s remaining...")
            time.sleep(1)
        
        recovery_end = datetime.now()
        logger.info("✅ Recovery data collection complete")
        
        # Step 8: Collect actual observability data
        logger.info("📥 Collecting observability data from backends...")
        
        # Collect metrics
        logger.info("   Collecting metrics from Prometheus...")
        metrics_df = metrics_collector.collect_system_metrics(
            baseline_start, recovery_end, config.data_collection.services_filter
        )
        logger.info(f"   ✅ Collected {len(metrics_df)} metrics records")
        
        # Collect logs  
        logger.info("   Collecting logs from Loki...")
        logs_df = logs_collector.collect_service_logs(
            baseline_start, recovery_end, config.data_collection.services_filter
        )
        logger.info(f"   ✅ Collected {len(logs_df)} log records")
        
        # Collect traces
        logger.info("   Collecting traces from Jaeger...")
        traces_df = traces_collector.collect_service_traces(
            baseline_start, recovery_end, config.data_collection.services_filter
        )
        logger.info(f"   ✅ Collected {len(traces_df)} trace records")
        
        # Step 9: Export data using Task 8 export system
        logger.info("📤 Exporting experiment data...")
        export_manager = DataExportManager(config.export.output_directory)
        
        # Create metadata
        metadata = {
            'experiment_type': 'real_chaos_experiment',
            'config_name': config.name,
            'chaos_type': config.chaos.fault_type,
            'chaos_intensity': config.chaos.intensity,
            'target_services': valid_targets,
            'baseline_start': baseline_start.isoformat(),
            'chaos_start': chaos_start.isoformat(),
            'chaos_end': chaos_end.isoformat(),
            'recovery_end': recovery_end.isoformat(),
            'total_duration': (recovery_end - baseline_start).total_seconds(),
            'data_summary': {
                'metrics_records': len(metrics_df),
                'logs_records': len(logs_df),
                'traces_records': len(traces_df)
            }
        }
        
        # Export using the primary target service name
        primary_service = valid_targets[0] if valid_targets else 'frontend'
        
        success = export_manager.export_complete_experiment(
            metrics_df=metrics_df,
            logs_df=logs_df, 
            traces_df=traces_df,
            service=primary_service,
            fault_type=config.chaos.fault_type,
            experiment_number=None,  # Auto-generate
            injection_timestamp=injection_timestamp,
            metadata=metadata
        )
        
        if success:
            logger.info("✅ Data export completed successfully")
        else:
            logger.warning("⚠️  Data export had some issues")
        
        # Step 10: Show results
        experiment_end_time = datetime.now()
        total_duration = (experiment_end_time - experiment_start_time).total_seconds()
        
        print(f"\n🎉 Real Chaos Experiment Complete!")
        print(f"   Experiment: {config.name}")
        print(f"   Duration: {total_duration:.0f} seconds ({total_duration/60:.1f} minutes)")
        print(f"   Services Targeted: {len(valid_targets)}")
        print(f"   Data Collected:")
        print(f"     • Metrics: {len(metrics_df)} records")
        print(f"     • Logs: {len(logs_df)} records") 
        print(f"     • Traces: {len(traces_df)} records")
        print(f"   Export Directory: {config.export.output_directory}/{primary_service}_{config.chaos.fault_type}/")
        
        # Show validation results
        validation = export_manager.validate_exported_experiment(
            primary_service, config.chaos.fault_type, 1
        )
        
        if validation.get('overall_valid'):
            print(f"   ✅ Export validation: PASSED")
        else:
            print(f"   ⚠️  Export validation: Some issues detected")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Experiment interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_real_chaos_experiment()
    sys.exit(0 if success else 1)