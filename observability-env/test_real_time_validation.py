#!/usr/bin/env python3
"""
Test script for the Real-Time OpenTelemetry Demo Validation System.
This runs a quick validation test to ensure everything is working.
"""

import sys
import os
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_prerequisites():
    """Check if all prerequisites are met."""
    logger.info("🔍 Checking prerequisites...")
    
    # Check if we're in the right directory
    if not Path('real_time_otel_validation.py').exists():
        logger.error("❌ real_time_otel_validation.py not found. Run from observability-env directory.")
        return False
    
    # Check if Docker is running
    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, timeout=10)
        if result.returncode != 0:
            logger.error("❌ Docker is not running or accessible")
            return False
        logger.info("✅ Docker is accessible")
    except Exception as e:
        logger.error(f"❌ Docker check failed: {e}")
        return False
    
    # Check if OpenTelemetry Demo containers are running
    try:
        result = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            containers = result.stdout.strip().split('\n')
            otel_containers = [c for c in containers if any(svc in c for svc in 
                             ['frontend', 'emailservice', 'paymentservice', 'productcatalogservice'])]
            
            if otel_containers:
                logger.info(f"✅ Found OpenTelemetry Demo containers: {otel_containers}")
            else:
                logger.warning("⚠️  No OpenTelemetry Demo containers found")
                logger.info("💡 Start the demo with: docker-compose -f docker-compose.otel-demo.yml up -d")
                return False
    except Exception as e:
        logger.error(f"❌ Container check failed: {e}")
        return False
    
    # Check observability endpoints
    import requests
    
    endpoints = {
        'Prometheus': 'http://localhost:9090',
        'Loki': 'http://localhost:3100', 
        'Jaeger': 'http://localhost:16686',
        'Grafana': 'http://localhost:3000'
    }
    
    for name, url in endpoints.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code in [200, 404]:  # 404 is OK for some endpoints
                logger.info(f"✅ {name} accessible at {url}")
            else:
                logger.warning(f"⚠️  {name} returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️  {name} not accessible: {e}")
    
    return True


def run_quick_validation():
    """Run a quick validation test."""
    logger.info("🚀 Running quick validation test...")
    
    try:
        # Run the validation with short duration
        cmd = [sys.executable, 'real_time_otel_validation.py', '--chaos-type', 'cpu', '--duration', '30']
        
        logger.info(f"🏃 Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, timeout=120)  # 2 minute timeout
        
        if result.returncode == 0:
            logger.info("✅ Quick validation test PASSED")
            return True
        else:
            logger.error("❌ Quick validation test FAILED")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Validation test timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Validation test failed: {e}")
        return False


def run_dry_run():
    """Run a dry run without actual chaos injection."""
    logger.info("🧪 Running dry run test...")
    
    try:
        # Import and test the validator class directly
        sys.path.insert(0, '.')
        from real_time_otel_validation import RealTimeOTelValidator
        
        validator = RealTimeOTelValidator()
        
        # Test environment validation
        if not validator.validate_environment():
            logger.error("❌ Environment validation failed")
            return False
        
        # Test collector initialization
        if not validator.initialize_collectors():
            logger.error("❌ Collector initialization failed")
            return False
        
        # Test service discovery
        services = validator.discover_target_services()
        if not services:
            logger.error("❌ No services discovered")
            return False
        
        logger.info(f"✅ Dry run completed successfully. Found services: {services}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dry run failed: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 Real-Time OpenTelemetry Demo Validation - Test Suite")
    print("=" * 65)
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        logger.error("❌ Prerequisites check failed")
        return False
    
    # Step 2: Run dry run
    if not run_dry_run():
        logger.error("❌ Dry run failed")
        return False
    
    # Step 3: Ask user if they want to run full validation
    print("\n🤔 Prerequisites and dry run passed!")
    print("   Ready to run the full validation test with actual chaos injection?")
    print("   This will inject CPU stress for 30 seconds.")
    
    response = input("   Continue with full test? (y/N): ").strip().lower()
    
    if response == 'y':
        success = run_quick_validation()
        if success:
            print("\n🎉 All tests PASSED! The real-time validation system is working correctly.")
            print("💡 You can now run full experiments with:")
            print("   python real_time_otel_validation.py --chaos-type cpu --duration 60")
        else:
            print("\n❌ Full validation test failed. Check the logs for details.")
        return success
    else:
        print("\n✅ Test suite completed successfully (dry run only)")
        print("💡 Run the full validation when ready with:")
        print("   python real_time_otel_validation.py --chaos-type cpu --duration 60")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)