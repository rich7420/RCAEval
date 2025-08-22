"""
Directory structure organizer for RE2-compatible data organization.
Implements requirements 5.1, 5.2, 5.3, 5.4 for directory structure organization.
"""

import os
import shutil
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DirectoryOrganizer:
    """Organize collected data in RE2-compatible directory structure."""
    
    def __init__(self, base_output_dir: str = "data/collected"):
        """
        Initialize directory organizer.
        
        Args:
            base_output_dir: Base directory for organized data
        """
        self.base_output_dir = Path(base_output_dir)
        self.fault_types = ['cpu', 'mem', 'disk', 'delay', 'loss', 'socket']
        
    def create_re2_structure(self, service: str, fault_type: str, 
                           experiment_number: int) -> str:
        """
        Create RE2-compatible directory structure.
        
        Directory format: service_faulttype/experiment_number/
        
        Args:
            service: Service name
            fault_type: Fault type (cpu, mem, disk, delay, loss, socket)
            experiment_number: Experiment number (1, 2, 3, etc.)
            
        Returns:
            Path to created experiment directory
        """
        try:
            # Validate fault type
            if fault_type not in self.fault_types:
                logger.warning(f"Unknown fault type: {fault_type}. Using as-is.")
                
            # Create directory structure
            service_fault_dir = f"{service}_{fault_type}"
            experiment_dir = self.base_output_dir / service_fault_dir / str(experiment_number)
            
            # Create directories
            experiment_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Created RE2 directory structure: {experiment_dir}")
            return str(experiment_dir)
            
        except Exception as e:
            logger.error(f"Failed to create RE2 directory structure: {e}")
            raise
            
    def organize_experiment_data(self, source_dir: str, service: str, 
                               fault_type: str, experiment_number: int,
                               file_mapping: Optional[Dict[str, str]] = None) -> bool:
        """
        Organize experiment data into RE2 structure.
        
        Args:
            source_dir: Source directory containing data files
            service: Service name
            fault_type: Fault type
            experiment_number: Experiment number
            file_mapping: Optional mapping of source files to target names
            
        Returns:
            True if organization successful, False otherwise
        """
        try:
            # Create target directory
            target_dir = self.create_re2_structure(service, fault_type, experiment_number)
            
            # Default file mapping for RE2 format
            if file_mapping is None:
                file_mapping = {
                    'metrics.csv': 'metrics.csv',
                    'logs.csv': 'logs.csv',
                    'logts.csv': 'logts.csv',
                    'traces.csv': 'traces.csv',
                    'tracets_err.csv': 'tracets_err.csv',
                    'tracets_lat.csv': 'tracets_lat.csv',
                    'cluster_info.json': 'cluster_info.json',
                    'inject_time.txt': 'inject_time.txt'
                }
                
            # Copy files to target directory
            source_path = Path(source_dir)
            target_path = Path(target_dir)
            
            copied_files = []
            missing_files = []
            
            for source_file, target_file in file_mapping.items():
                source_file_path = source_path / source_file
                target_file_path = target_path / target_file
                
                if source_file_path.exists():
                    shutil.copy2(source_file_path, target_file_path)
                    copied_files.append(target_file)
                    logger.debug(f"Copied {source_file} -> {target_file}")
                else:
                    missing_files.append(source_file)
                    logger.warning(f"Source file not found: {source_file}")
                    
            # Log summary
            logger.info(f"Organized experiment data for {service}_{fault_type}/{experiment_number}")
            logger.info(f"Copied files: {copied_files}")
            if missing_files:
                logger.warning(f"Missing files: {missing_files}")
                
            return len(copied_files) > 0
            
        except Exception as e:
            logger.error(f"Failed to organize experiment data: {e}")
            return False
            
    def get_next_experiment_number(self, service: str, fault_type: str) -> int:
        """
        Get the next available experiment number for a service-fault combination.
        
        Args:
            service: Service name
            fault_type: Fault type
            
        Returns:
            Next experiment number
        """
        try:
            service_fault_dir = self.base_output_dir / f"{service}_{fault_type}"
            
            if not service_fault_dir.exists():
                return 1
                
            # Find existing experiment numbers
            existing_numbers = []
            for item in service_fault_dir.iterdir():
                if item.is_dir() and item.name.isdigit():
                    existing_numbers.append(int(item.name))
                    
            if not existing_numbers:
                return 1
                
            return max(existing_numbers) + 1
            
        except Exception as e:
            logger.error(f"Failed to get next experiment number: {e}")
            return 1
            
    def list_experiments(self, service: Optional[str] = None, 
                        fault_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List existing experiments in the directory structure.
        
        Args:
            service: Optional service filter
            fault_type: Optional fault type filter
            
        Returns:
            List of experiment information
        """
        experiments = []
        
        try:
            if not self.base_output_dir.exists():
                return experiments
                
            for service_fault_dir in self.base_output_dir.iterdir():
                if not service_fault_dir.is_dir():
                    continue
                    
                # Parse service_faulttype directory name
                dir_name = service_fault_dir.name
                if '_' not in dir_name:
                    continue
                    
                parts = dir_name.rsplit('_', 1)
                if len(parts) != 2:
                    continue
                    
                dir_service, dir_fault_type = parts
                
                # Apply filters
                if service and dir_service != service:
                    continue
                if fault_type and dir_fault_type != fault_type:
                    continue
                    
                # Find experiment numbers
                for exp_dir in service_fault_dir.iterdir():
                    if not exp_dir.is_dir() or not exp_dir.name.isdigit():
                        continue
                        
                    exp_number = int(exp_dir.name)
                    
                    # Get experiment info
                    exp_info = {
                        'service': dir_service,
                        'fault_type': dir_fault_type,
                        'experiment_number': exp_number,
                        'path': str(exp_dir),
                        'files': [f.name for f in exp_dir.iterdir() if f.is_file()]
                    }
                    
                    # Add file sizes
                    exp_info['file_sizes'] = {}
                    for file_path in exp_dir.iterdir():
                        if file_path.is_file():
                            exp_info['file_sizes'][file_path.name] = file_path.stat().st_size
                            
                    experiments.append(exp_info)
                    
            # Sort by service, fault_type, experiment_number
            experiments.sort(key=lambda x: (x['service'], x['fault_type'], x['experiment_number']))
            
            return experiments
            
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return []
            
    def validate_experiment_structure(self, service: str, fault_type: str, 
                                    experiment_number: int) -> Dict[str, Any]:
        """
        Validate experiment directory structure and files.
        
        Args:
            service: Service name
            fault_type: Fault type
            experiment_number: Experiment number
            
        Returns:
            Dict with validation results
        """
        try:
            service_fault_dir = f"{service}_{fault_type}"
            experiment_dir = self.base_output_dir / service_fault_dir / str(experiment_number)
            
            validation = {
                'valid': False,
                'directory_exists': False,
                'required_files': {},
                'optional_files': {},
                'extra_files': [],
                'total_size_bytes': 0
            }
            
            if not experiment_dir.exists():
                validation['error'] = f"Experiment directory does not exist: {experiment_dir}"
                return validation
                
            validation['directory_exists'] = True
            
            # Check required files
            required_files = ['metrics.csv', 'logs.csv', 'traces.csv', 'cluster_info.json']
            optional_files = ['logts.csv', 'tracets_err.csv', 'tracets_lat.csv', 'inject_time.txt']
            
            all_files = set()
            total_size = 0
            
            for file_path in experiment_dir.iterdir():
                if file_path.is_file():
                    file_name = file_path.name
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    all_files.add(file_name)
                    
                    if file_name in required_files:
                        validation['required_files'][file_name] = {
                            'exists': True,
                            'size_bytes': file_size,
                            'empty': file_size == 0
                        }
                    elif file_name in optional_files:
                        validation['optional_files'][file_name] = {
                            'exists': True,
                            'size_bytes': file_size,
                            'empty': file_size == 0
                        }
                    else:
                        validation['extra_files'].append(file_name)
                        
            # Check missing required files
            for req_file in required_files:
                if req_file not in all_files:
                    validation['required_files'][req_file] = {
                        'exists': False,
                        'size_bytes': 0,
                        'empty': True
                    }
                    
            # Check missing optional files
            for opt_file in optional_files:
                if opt_file not in all_files:
                    validation['optional_files'][opt_file] = {
                        'exists': False,
                        'size_bytes': 0,
                        'empty': True
                    }
                    
            validation['total_size_bytes'] = total_size
            
            # Determine if valid (all required files exist and are non-empty)
            required_valid = all(
                info['exists'] and not info['empty'] 
                for info in validation['required_files'].values()
            )
            
            validation['valid'] = required_valid
            
            return validation
            
        except Exception as e:
            logger.error(f"Failed to validate experiment structure: {e}")
            return {'valid': False, 'error': str(e)}
            
    def create_experiment_metadata(self, service: str, fault_type: str, 
                                 experiment_number: int, metadata: Dict[str, Any]) -> bool:
        """
        Create metadata file for experiment.
        
        Args:
            service: Service name
            fault_type: Fault type
            experiment_number: Experiment number
            metadata: Metadata dictionary
            
        Returns:
            True if metadata created successfully, False otherwise
        """
        try:
            service_fault_dir = f"{service}_{fault_type}"
            experiment_dir = self.base_output_dir / service_fault_dir / str(experiment_number)
            
            if not experiment_dir.exists():
                logger.error(f"Experiment directory does not exist: {experiment_dir}")
                return False
                
            # Add standard metadata
            full_metadata = {
                'service': service,
                'fault_type': fault_type,
                'experiment_number': experiment_number,
                'created_at': datetime.now().isoformat(),
                'directory_path': str(experiment_dir),
                **metadata
            }
            
            # Write metadata file
            metadata_path = experiment_dir / 'experiment_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(full_metadata, f, indent=2)
                
            logger.info(f"Created experiment metadata: {metadata_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create experiment metadata: {e}")
            return False
            
    def cleanup_experiment(self, service: str, fault_type: str, 
                         experiment_number: int) -> bool:
        """
        Clean up experiment directory.
        
        Args:
            service: Service name
            fault_type: Fault type
            experiment_number: Experiment number
            
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            service_fault_dir = f"{service}_{fault_type}"
            experiment_dir = self.base_output_dir / service_fault_dir / str(experiment_number)
            
            if not experiment_dir.exists():
                logger.warning(f"Experiment directory does not exist: {experiment_dir}")
                return True
                
            # Remove experiment directory
            shutil.rmtree(experiment_dir)
            logger.info(f"Cleaned up experiment directory: {experiment_dir}")
            
            # Remove parent directory if empty
            parent_dir = experiment_dir.parent
            if parent_dir.exists() and not any(parent_dir.iterdir()):
                parent_dir.rmdir()
                logger.info(f"Removed empty parent directory: {parent_dir}")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup experiment: {e}")
            return False
            
    def get_directory_summary(self) -> Dict[str, Any]:
        """
        Get summary of organized directory structure.
        
        Returns:
            Dict with directory summary
        """
        try:
            summary = {
                'base_directory': str(self.base_output_dir),
                'total_experiments': 0,
                'services': set(),
                'fault_types': set(),
                'service_fault_combinations': {},
                'total_size_bytes': 0
            }
            
            experiments = self.list_experiments()
            
            for exp in experiments:
                summary['total_experiments'] += 1
                summary['services'].add(exp['service'])
                summary['fault_types'].add(exp['fault_type'])
                
                # Count combinations
                combo_key = f"{exp['service']}_{exp['fault_type']}"
                if combo_key not in summary['service_fault_combinations']:
                    summary['service_fault_combinations'][combo_key] = 0
                summary['service_fault_combinations'][combo_key] += 1
                
                # Add file sizes
                for file_size in exp['file_sizes'].values():
                    summary['total_size_bytes'] += file_size
                    
            # Convert sets to sorted lists
            summary['services'] = sorted(list(summary['services']))
            summary['fault_types'] = sorted(list(summary['fault_types']))
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get directory summary: {e}")
            return {}
            
    def migrate_data_to_re2_structure(self, source_base_dir: str) -> bool:
        """
        Migrate existing data to RE2 directory structure.
        
        Args:
            source_base_dir: Base directory containing existing data
            
        Returns:
            True if migration successful, False otherwise
        """
        try:
            source_path = Path(source_base_dir)
            
            if not source_path.exists():
                logger.error(f"Source directory does not exist: {source_base_dir}")
                return False
                
            migrated_count = 0
            
            # Look for data directories that might need migration
            for item in source_path.iterdir():
                if not item.is_dir():
                    continue
                    
                # Try to parse directory name for service and fault type
                dir_name = item.name
                
                # Check if already in RE2 format
                if '_' in dir_name and any(item.iterdir()):
                    parts = dir_name.rsplit('_', 1)
                    if len(parts) == 2 and parts[1] in self.fault_types:
                        # Already in RE2 format, check for numbered subdirectories
                        service, fault_type = parts
                        
                        for subitem in item.iterdir():
                            if subitem.is_dir() and subitem.name.isdigit():
                                # Already properly organized
                                continue
                            elif subitem.is_file():
                                # Files in service_fault directory, need to move to numbered subdir
                                exp_num = self.get_next_experiment_number(service, fault_type)
                                target_dir = self.create_re2_structure(service, fault_type, exp_num)
                                
                                # Move files
                                for file_item in item.iterdir():
                                    if file_item.is_file():
                                        shutil.move(str(file_item), os.path.join(target_dir, file_item.name))
                                        
                                migrated_count += 1
                                break
                                
            logger.info(f"Migrated {migrated_count} data directories to RE2 structure")
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate data to RE2 structure: {e}")
            return False