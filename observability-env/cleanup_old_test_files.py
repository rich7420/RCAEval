#!/usr/bin/env python3
"""
Cleanup script to organize old test files after implementing the consolidated
Real-Time OpenTelemetry Demo Validation System.

This script moves old test files to an archive directory to keep the workspace clean
while preserving them for reference.
"""

import os
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def cleanup_old_files():
    """Move old test files to archive directory."""
    
    # Create archive directory
    archive_dir = Path('archive_old_tests')
    archive_dir.mkdir(exist_ok=True)
    
    # Files to archive (these are now replaced by real_time_otel_validation.py)
    files_to_archive = [
        'demo_real_experiment.py',
        'run_automated_experiment.py', 
        'debug_service_discovery.py'
    ]
    
    logger.info("🗂️  Archiving old test files...")
    
    archived_count = 0
    for file_path in files_to_archive:
        if Path(file_path).exists():
            try:
                shutil.move(file_path, archive_dir / file_path)
                logger.info(f"✅ Archived: {file_path}")
                archived_count += 1
            except Exception as e:
                logger.error(f"❌ Failed to archive {file_path}: {e}")
        else:
            logger.info(f"⏭️  Skipped: {file_path} (not found)")
    
    # Create README in archive
    archive_readme = archive_dir / 'README.md'
    with open(archive_readme, 'w') as f:
        f.write("""# Archived Test Files

These files have been replaced by the consolidated Real-Time OpenTelemetry Demo Validation System.

## Replacement

All functionality from these files is now available in:
- `real_time_otel_validation.py` - Main validation system
- `test_real_time_validation.py` - Test suite

## Original Files

- `demo_real_experiment.py` - Basic experiment demonstration
- `run_automated_experiment.py` - Automated experiment runner
- `debug_service_discovery.py` - Service discovery debugging

## Migration

The new system provides:
- Better error handling and safety features
- Real-time monitoring and validation
- Comprehensive reporting
- Single command operation
- Integration with Task 8 configuration system

See `REAL_TIME_VALIDATION_README.md` for usage instructions.
""")
    
    logger.info(f"📄 Created archive README: {archive_readme}")
    
    if archived_count > 0:
        logger.info(f"✅ Successfully archived {archived_count} files to {archive_dir}")
        logger.info("💡 The new consolidated system is ready to use:")
        logger.info("   python real_time_otel_validation.py --chaos-type cpu --duration 60")
    else:
        logger.info("ℹ️  No files needed archiving")
    
    return archived_count


def show_new_system_info():
    """Show information about the new consolidated system."""
    
    print("\n🚀 Task 8 Test Files Consolidated!")
    print("=" * 50)
    print()
    print("📁 New Consolidated System:")
    print("   • real_time_otel_validation.py     - Main validation system")
    print("   • test_real_time_validation.py     - Test suite")
    print("   • REAL_TIME_VALIDATION_README.md   - Documentation")
    print()
    print("🎯 Key Features:")
    print("   • Single command validation")
    print("   • Real-time chaos injection and monitoring")
    print("   • Comprehensive data quality validation")
    print("   • Safety-first design with automatic cleanup")
    print("   • Integration with Task 8 configuration system")
    print()
    print("🚀 Quick Start:")
    print("   1. Test the system:")
    print("      python test_real_time_validation.py")
    print()
    print("   2. Run validation:")
    print("      python real_time_otel_validation.py --chaos-type cpu --duration 60")
    print()
    print("📖 For detailed usage, see: REAL_TIME_VALIDATION_README.md")


if __name__ == "__main__":
    print("🧹 Cleaning up old Task 8 test files...")
    
    archived_count = cleanup_old_files()
    show_new_system_info()
    
    print(f"\n✅ Cleanup complete! Archived {archived_count} files.")