"""
Injection timestamp recorder for chaos engineering timing capture.
Implements requirements 3.7 and 4.5 for chaos injection timing correlation.
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class InjectionTimestampRecorder:
    """Record and manage chaos injection timestamps for data correlation."""
    
    def __init__(self, output_dir: str = "data/collected"):
        """
        Initialize injection timestamp recorder.
        
        Args:
            output_dir: Base directory for timestamp files
        """
        self.output_dir = Path(output_dir)
        self.injection_log = []
        
    def record_injection_start(self, service: str, fault_type: str, 
                             experiment_number: int, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Record the start of chaos injection.
        
        Args:
            service: Target service name
            fault_type: Type of fault being injected
            experiment_number: Experiment number
            metadata: Optional metadata about the injection
            
        Returns:
            Unix timestamp of injection start
        """
        try:
            timestamp = int(time.time())
            
            injection_record = {
                'service': service,
                'fault_type': fault_type,
                'experiment_number': experiment_number,
                'injection_start': timestamp,
                'injection_start_iso': datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                'metadata': metadata or {}
            }
            
            self.injection_log.append(injection_record)
            
            logger.info(f"Recorded injection start for {service}_{fault_type}/{experiment_number} at {timestamp}")
            return timestamp
            
        except Exception as e:
            logger.error(f"Failed to record injection start: {e}")
            return int(time.time())
            
    def record_injection_end(self, service: str, fault_type: str, 
                           experiment_number: int) -> int:
        """
        Record the end of chaos injection.
        
        Args:
            service: Target service name
            fault_type: Type of fault being injected
            experiment_number: Experiment number
            
        Returns:
            Unix timestamp of injection end
        """
        try:
            timestamp = int(time.time())
            
            # Find matching injection record
            for record in reversed(self.injection_log):
                if (record['service'] == service and 
                    record['fault_type'] == fault_type and 
                    record['experiment_number'] == experiment_number and
                    'injection_end' not in record):
                    
                    record['injection_end'] = timestamp
                    record['injection_end_iso'] = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
                    record['injection_duration'] = timestamp - record['injection_start']
                    
                    logger.info(f"Recorded injection end for {service}_{fault_type}/{experiment_number} at {timestamp}")
                    return timestamp
                    
            # If no matching record found, create a new one
            logger.warning(f"No matching injection start found for {service}_{fault_type}/{experiment_number}")
            injection_record = {
                'service': service,
                'fault_type': fault_type,
                'experiment_number': experiment_number,
                'injection_end': timestamp,
                'injection_end_iso': datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                'metadata': {}
            }
            self.injection_log.append(injection_record)
            
            return timestamp
            
        except Exception as e:
            logger.error(f"Failed to record injection end: {e}")
            return int(time.time())
            
    def create_inject_time_file(self, service: str, fault_type: str, 
                              experiment_number: int) -> bool:
        """
        Create inject_time.txt file for specific experiment.
        
        Args:
            service: Service name
            fault_type: Fault type
            experiment_number: Experiment number
            
        Returns:
            True if file created successfully, False otherwise
        """
        try:
            # Find injection record
            injection_record = None
            for record in self.injection_log:
                if (record['service'] == service and 
                    record['fault_type'] == fault_type and 
                    record['experiment_number'] == experiment_number):
                    injection_record = record
                    break
                    
            if not injection_record:
                logger.error(f"No injection record found for {service}_{fault_type}/{experiment_number}")
                return False
                
            # Create experiment directory path
            service_fault_dir = f"{service}_{fault_type}"
            experiment_dir = self.output_dir / service_fault_dir / str(experiment_number)
            experiment_dir.mkdir(parents=True, exist_ok=True)
            
            # Create inject_time.txt file
            inject_time_path = experiment_dir / 'inject_time.txt'
            
            # Write injection timestamp (Unix timestamp as required by RE2 format)
            injection_start = injection_record.get('injection_start')
            if injection_start:
                with open(inject_time_path, 'w') as f:
                    f.write(str(injection_start))
                    
                logger.info(f"Created inject_time.txt: {inject_time_path} with timestamp {injection_start}")
                return True
            else:
                logger.error(f"No injection start timestamp found in record")
                return False
                
        except Exception as e:
            logger.error(f"Failed to create inject_time.txt file: {e}")
            return False
            
    def create_detailed_injection_log(self, service: str, fault_type: str, 
                                    experiment_number: int) -> bool:
        """
        Create detailed injection log file with full timing information.
        
        Args:
            service: Service name
            fault_type: Fault type
            experiment_number: Experiment number
            
        Returns:
            True if log created successfully, False otherwise
        """
        try:
            # Find injection record
            injection_record = None
            for record in self.injection_log:
                if (record['service'] == service and 
                    record['fault_type'] == fault_type and 
                    record['experiment_number'] == experiment_number):
                    injection_record = record
                    break
                    
            if not injection_record:
                logger.error(f"No injection record found for {service}_{fault_type}/{experiment_number}")
                return False
                
            # Create experiment directory path
            service_fault_dir = f"{service}_{fault_type}"
            experiment_dir = self.output_dir / service_fault_dir / str(experiment_number)
            experiment_dir.mkdir(parents=True, exist_ok=True)
            
            # Create detailed injection log
            injection_log_path = experiment_dir / 'injection_log.json'
            
            with open(injection_log_path, 'w') as f:
                json.dump(injection_record, f, indent=2)
                
            logger.info(f"Created detailed injection log: {injection_log_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create detailed injection log: {e}")
            return False
            
    def get_injection_timing(self, service: str, fault_type: str, 
                           experiment_number: int) -> Optional[Dict[str, Any]]:
        """
        Get injection timing information for an experiment.
        
        Args:
            service: Service name
            fault_type: Fault type
            experiment_number: Experiment number
            
        Returns:
            Dict with timing information or None if not found
        """
        try:
            for record in self.injection_log:
                if (record['service'] == service and 
                    record['fault_type'] == fault_type and 
                    record['experiment_number'] == experiment_number):
                    return record.copy()
                    
            return None
            
        except Exception as e:
            logger.error(f"Failed to get injection timing: {e}")
            return None
            
    def correlate_with_data_collection(self, service: str, fault_type: str, 
                                     experiment_number: int, 
                                     data_start_time: int, data_end_time: int) -> Dict[str, Any]:
        """
        Correlate injection timing with data collection period.
        
        Args:
            service: Service name
            fault_type: Fault type
            experiment_number: Experiment number
            data_start_time: Data collection start timestamp
            data_end_time: Data collection end timestamp
            
        Returns:
            Dict with correlation information
        """
        try:
            injection_timing = self.get_injection_timing(service, fault_type, experiment_number)
            
            if not injection_timing:
                return {
                    'injection_found': False,
                    'error': 'No injection timing record found'
                }
                
            injection_start = injection_timing.get('injection_start')
            injection_end = injection_timing.get('injection_end')
            
            correlation = {
                'injection_found': True,
                'injection_start': injection_start,
                'injection_end': injection_end,
                'data_start_time': data_start_time,
                'data_end_time': data_end_time,
                'injection_duration': injection_timing.get('injection_duration'),
                'data_collection_duration': data_end_time - data_start_time
            }
            
            # Calculate timing relationships
            if injection_start:
                correlation['injection_relative_to_data_start'] = injection_start - data_start_time
                correlation['injection_within_collection_period'] = (
                    data_start_time <= injection_start <= data_end_time
                )
                
                # Calculate pre-injection, during-injection, and post-injection periods
                correlation['pre_injection_duration'] = max(0, injection_start - data_start_time)
                
                if injection_end:
                    correlation['during_injection_duration'] = injection_end - injection_start
                    correlation['post_injection_duration'] = max(0, data_end_time - injection_end)
                else:
                    correlation['during_injection_duration'] = max(0, data_end_time - injection_start)
                    correlation['post_injection_duration'] = 0
                    
            return correlation
            
        except Exception as e:
            logger.error(f"Failed to correlate injection with data collection: {e}")
            return {'injection_found': False, 'error': str(e)}
            
    def export_all_injection_logs(self, output_file: str) -> bool:
        """
        Export all injection logs to a single file.
        
        Args:
            output_file: Path to output file
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            export_data = {
                'export_timestamp': datetime.now(tz=timezone.utc).isoformat(),
                'total_injections': len(self.injection_log),
                'injections': self.injection_log
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            logger.info(f"Exported {len(self.injection_log)} injection logs to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export injection logs: {e}")
            return False
            
    def load_injection_logs(self, input_file: str) -> bool:
        """
        Load injection logs from file.
        
        Args:
            input_file: Path to input file
            
        Returns:
            True if load successful, False otherwise
        """
        try:
            input_path = Path(input_file)
            
            if not input_path.exists():
                logger.error(f"Input file does not exist: {input_file}")
                return False
                
            with open(input_path, 'r') as f:
                data = json.load(f)
                
            if 'injections' in data:
                self.injection_log = data['injections']
                logger.info(f"Loaded {len(self.injection_log)} injection logs from {input_file}")
                return True
            else:
                logger.error("Invalid injection log file format")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load injection logs: {e}")
            return False
            
    def clear_injection_log(self) -> None:
        """Clear all injection logs."""
        self.injection_log.clear()
        logger.info("Cleared all injection logs")
        
    def get_injection_summary(self) -> Dict[str, Any]:
        """
        Get summary of recorded injections.
        
        Returns:
            Dict with injection summary
        """
        try:
            if not self.injection_log:
                return {
                    'total_injections': 0,
                    'services': [],
                    'fault_types': [],
                    'experiments': []
                }
                
            services = set()
            fault_types = set()
            experiments = set()
            completed_injections = 0
            
            for record in self.injection_log:
                services.add(record['service'])
                fault_types.add(record['fault_type'])
                experiments.add(f"{record['service']}_{record['fault_type']}/{record['experiment_number']}")
                
                if 'injection_end' in record:
                    completed_injections += 1
                    
            summary = {
                'total_injections': len(self.injection_log),
                'completed_injections': completed_injections,
                'ongoing_injections': len(self.injection_log) - completed_injections,
                'services': sorted(list(services)),
                'fault_types': sorted(list(fault_types)),
                'experiments': sorted(list(experiments))
            }
            
            # Add timing statistics for completed injections
            durations = []
            for record in self.injection_log:
                if 'injection_duration' in record:
                    durations.append(record['injection_duration'])
                    
            if durations:
                summary['duration_stats'] = {
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'avg_duration': sum(durations) / len(durations),
                    'total_injection_time': sum(durations)
                }
                
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get injection summary: {e}")
            return {}
            
    def validate_inject_time_file(self, file_path: str) -> bool:
        """
        Validate inject_time.txt file format.
        
        Args:
            file_path: Path to inject_time.txt file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"inject_time.txt file does not exist: {file_path}")
                return False
                
            with open(file_path, 'r') as f:
                content = f.read().strip()
                
            # Should contain a single Unix timestamp
            try:
                timestamp = int(content)
                
                # Validate timestamp is reasonable (between 2020 and 2030)
                min_timestamp = 1577836800  # 2020-01-01
                max_timestamp = 1893456000  # 2030-01-01
                
                if min_timestamp <= timestamp <= max_timestamp:
                    logger.info(f"inject_time.txt validation passed: {file_path}")
                    return True
                else:
                    logger.error(f"Timestamp out of reasonable range: {timestamp}")
                    return False
                    
            except ValueError:
                logger.error(f"Invalid timestamp format in inject_time.txt: {content}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to validate inject_time.txt file: {e}")
            return False