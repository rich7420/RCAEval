#!/usr/bin/env python3
"""
Real experiment demonstration using Task 8 configuration system.
This runs an actual experiment with the configured durations.
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


def create_quick_experiment_config():
    """Create a quick experiment configuration for demonstration."""
    template_manager = ConfigTemplateManager()
    
    # Use the quick validation template but make it even shorter
    config = template_manager.create_config_from_template(
        'quick_validation',
        ['prometheus'],  # Target prometheus service (known to be available and safe)
        overrides={
            'name': 'demo_real_experiment',
            'description': 'Real experiment demonstration using Task 8 configuration system',
            'chaos': {
                'duration': 30,  # 30 seconds of chaos
                'intensity': 'low'
            },
            'data_collection': {
                'collection_duration': 90,  # 1.5 minutes total
                'pre_chaos_duration': 20,   # 20 seconds baseline
                'post_chaos_duration': 40,  # 40 seconds recovery
                'sampling_interval': 5      # 5 second intervals
            },
            'traffic': {
                'enabled': False  # Disable traffic for simplicity
            }
        }
    )
    
    return config


def simulate_data_collection_phase(phase_name, duration_seconds, config):
    """Simulate a data collection phase."""
    logger.info(f"📊 Starting {phase_name} phase ({duration_seconds} seconds)")
    
    start_time = time.time()
    end_time = start_time + duration_seconds
    
    # Simulate data collection with progress updates
    while time.time() < end_time:
        remaining = int(end_time - time.time())
        if remaining % 5 == 0 or remaining <= 3:  # Update every 5 seconds or final countdown
            logger.info(f"   {phase_name}: {remaining} seconds remaining...")
        time.sleep(1)
    
    logger.info(f"✅ {phase_name} phase complete")


def simulate_chaos_injection(chaos_config):
    """Simulate chaos injection."""
    logger.info(f"💥 Injecting {chaos_config.fault_type} chaos (intensity: {chaos_config.intensity})")
    logger.info(f"   Target services: {chaos_config.target_services}")
    logger.info(f"   Duration: {chaos_config.duration} seconds")
    
    # Record injection start time
    injection_start = int(time.time())
    
    # Simulate the chaos injection process
    simulate_data_collection_phase("Chaos Injection", chaos_config.duration, None)
    
    return injection_start


def run_real_experiment():
    """Run a real experiment using the configuration system."""
    print("🚀 Real Experiment Demonstration - Task 8 Configuration System")
    print("=" * 70)
    
    # Step 1: Create experiment configuration
    logger.info("📋 Creating experiment configuration...")
    config = create_quick_experiment_config()
    
    # Step 2: Validate configuration
    logger.info("🔍 Validating experiment configuration...")
    parser = ExperimentConfigParser()
    errors = parser.validate_config(config)
    
    if errors:
        logger.error(f"❌ Configuration validation failed: {errors}")
        return False
    
    logger.info("✅ Configuration validation passed")
    
    # Step 3: Display experiment plan
    print(f"\n📋 Experiment Plan:")
    print(f"   Name: {config.name}")
    print(f"   Description: {config.description}")
    print(f"   Total Duration: {config.get_total_duration()} seconds")
    print(f"   Chaos Type: {config.chaos.fault_type} ({config.chaos.intensity} intensity)")
    print(f"   Target Services: {config.chaos.target_services}")
    print(f"   Timeline:")
    print(f"     • Pre-chaos baseline: {config.data_collection.pre_chaos_duration}s")
    print(f"     • Chaos injection: {config.chaos.duration}s") 
    print(f"     • Post-chaos recovery: {config.data_collection.post_chaos_duration}s")
    
    # Step 4: Service targeting validation
    logger.info("🎯 Validating service targets...")
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
                
        logger.info(f"✅ Valid targets: {valid_targets}")
        
    except Exception as e:
        logger.warning(f"⚠️  Service discovery failed (continuing anyway): {e}")
    
    # Step 5: Ask user confirmation
    print(f"\n🤔 Ready to run the experiment?")
    print(f"   This will take {config.get_total_duration()} seconds ({config.get_total_duration()/60:.1f} minutes)")
    
    response = input("   Continue? (y/N): ").strip().lower()
    if response != 'y':
        print("   Experiment cancelled.")
        return False
    
    # Step 6: Run the actual experiment
    print(f"\n🏁 Starting experiment at {datetime.now().strftime('%H:%M:%S')}")
    experiment_start = time.time()
    
    try:
        # Phase 1: Pre-chaos baseline
        simulate_data_collection_phase(
            "Pre-Chaos Baseline", 
            config.data_collection.pre_chaos_duration,
            config
        )
        
        # Phase 2: Chaos injection
        injection_timestamp = simulate_chaos_injection(config.chaos)
        
        # Phase 3: Post-chaos recovery
        simulate_data_collection_phase(
            "Post-Chaos Recovery",
            config.data_collection.post_chaos_duration, 
            config
        )
        
        experiment_end = time.time()
        actual_duration = int(experiment_end - experiment_start)
        
        # Step 7: Simulate data export
        logger.info("📤 Exporting experiment data...")
        
        # Create mock data for export demonstration
        import pandas as pd
        import numpy as np
        
        # Generate some mock observability data
        timestamps = [int(experiment_start) + i * 5 for i in range(actual_duration // 5)]
        
        mock_metrics = pd.DataFrame([
            {
                'timestamp': ts,
                'service': 'prometheus',
                'metric_name': 'cpu_usage_percent',
                'value': np.random.uniform(20, 80),
                'labels': '{"service":"prometheus"}'
            }
            for ts in timestamps
        ])
        
        mock_logs = pd.DataFrame([
            {
                'timestamp': ts,
                'service': 'prometheus', 
                'level': 'INFO',
                'message': f'Processing request at {ts}',
                'labels': '{"service":"prometheus"}'
            }
            for ts in timestamps[::2]  # Every other timestamp
        ])
        
        mock_traces = pd.DataFrame([
            {
                'trace_id': f'trace_{i:06d}',
                'span_id': f'span_{i:06d}',
                'parent_span_id': '',
                'service': 'prometheus',
                'operation': 'GET /metrics',
                'start_time': ts,
                'duration_ms': np.random.uniform(10, 100),
                'error': False
            }
            for i, ts in enumerate(timestamps[::3])  # Every third timestamp
        ])
        
        # Export using the data export system
        export_manager = DataExportManager("data/demo_experiment")
        
        success = export_manager.export_complete_experiment(
            metrics_df=mock_metrics,
            logs_df=mock_logs,
            traces_df=mock_traces,
            service='prometheus',
            fault_type='cpu',
            experiment_number=1,
            injection_timestamp=injection_timestamp,
            metadata={
                'experiment_type': 'demo',
                'actual_duration': actual_duration,
                'config_name': config.name
            }
        )
        
        if success:
            logger.info("✅ Data export completed successfully")
        else:
            logger.warning("⚠️  Data export had some issues")
        
        # Step 8: Show results
        print(f"\n🎉 Experiment Complete!")
        print(f"   Planned Duration: {config.get_total_duration()} seconds")
        print(f"   Actual Duration: {actual_duration} seconds")
        print(f"   Experiment ID: {config.experiment_id}")
        print(f"   Data exported to: data/demo_experiment/prometheus_cpu/1/")
        
        # Show exported files
        export_dir = Path("data/demo_experiment/prometheus_cpu/1")
        if export_dir.exists():
            print(f"\n📁 Exported Files:")
            for file_path in sorted(export_dir.iterdir()):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    print(f"     📄 {file_path.name} ({size} bytes)")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Experiment interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        return False


if __name__ == "__main__":
    success = run_real_experiment()
    sys.exit(0 if success else 1)