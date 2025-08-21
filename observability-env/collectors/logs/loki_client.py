"""
Loki API client for log collection with parsing and template extraction.
Implements requirements 4.2 and 6.4 for logs data collection.
"""

import requests
import pandas as pd
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
import logging
from urllib.parse import urljoin, quote
import base64
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class LokiClient:
    """Client for collecting logs from Loki with advanced processing capabilities."""
    
    def __init__(self, base_url: str = "http://localhost:3100", timeout: int = 30):
        """
        Initialize Loki client.
        
        Args:
            base_url: Loki server URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to Loki API."""
        url = urljoin(self.base_url, endpoint)
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Loki API request failed: {e}")
            raise
            
    def query_range(self, query: str, start_time: datetime, end_time: datetime,
                   limit: int = 5000, direction: str = "forward") -> pd.DataFrame:
        """
        Query Loki for log entries over a time range.
        
        Args:
            query: LogQL query string
            start_time: Start timestamp
            end_time: End timestamp
            limit: Maximum number of log entries
            direction: Query direction ("forward" or "backward")
            
        Returns:
            DataFrame with timestamp, service, level, message, labels columns
        """
        # Convert to nanosecond timestamps for Loki
        start_ns = int(start_time.timestamp() * 1e9)
        end_ns = int(end_time.timestamp() * 1e9)
        
        params = {
            'query': query,
            'start': start_ns,
            'end': end_ns,
            'limit': limit,
            'direction': direction
        }
        
        data = self._make_request('/loki/api/v1/query_range', params)
        
        if data['status'] != 'success':
            raise ValueError(f"Loki query failed: {data.get('error', 'Unknown error')}")
            
        return self._parse_range_response(data['data']['result'])
        
    def _parse_range_response(self, result: List[Dict]) -> pd.DataFrame:
        """Parse Loki range query response into DataFrame."""
        rows = []
        
        for stream in result:
            stream_labels = stream['stream']
            values = stream['values']
            
            # Extract service name from labels
            service = self._extract_service_name(stream_labels)
            
            for entry in values:
                timestamp_ns, log_line = entry
                timestamp = int(timestamp_ns) // 1000000000  # Convert to seconds
                
                # Parse log line for level and message
                level, message = self._parse_log_line(log_line)
                
                # Convert labels to JSON string
                labels_json = json.dumps(stream_labels, sort_keys=True)
                
                rows.append({
                    'timestamp': timestamp,
                    'service': service,
                    'level': level,
                    'message': message,
                    'labels': labels_json
                })
                
        return pd.DataFrame(rows)
        
    def _extract_service_name(self, labels: Dict[str, str]) -> str:
        """Extract service name from log labels."""
        # Try common service label names
        for label in ['service', 'job', 'container', 'app', 'service_name']:
            if label in labels:
                return labels[label]
        
        # Fallback to filename or unknown
        filename = labels.get('filename', '')
        if filename:
            # Extract service name from filename
            match = re.search(r'/([^/]+)\.log', filename)
            if match:
                return match.group(1)
                
        return labels.get('instance', 'unknown')
        
    def _parse_log_line(self, log_line: str) -> Tuple[str, str]:
        """Parse log line to extract level and message."""
        # Common log level patterns
        level_patterns = [
            r'\b(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b',
            r'\b(trace|debug|info|warn|warning|error|fatal|critical)\b',
            r'level[=:]\s*([a-zA-Z]+)',
            r'"level"\s*:\s*"([^"]+)"'
        ]
        
        level = 'INFO'  # Default level
        
        for pattern in level_patterns:
            match = re.search(pattern, log_line, re.IGNORECASE)
            if match:
                level = match.group(1).upper()
                break
                
        # Clean up the message (remove timestamps, levels, etc.)
        message = self._clean_log_message(log_line)
        
        return level, message
        
    def _clean_log_message(self, log_line: str) -> str:
        """Clean log message by removing common prefixes."""
        # Remove timestamp patterns
        timestamp_patterns = [
            r'^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}[.,]\d{3}[Z]?\s*',
            r'^\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}\s*',
            r'^\[\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}[.,]\d{3}[Z]?\]\s*'
        ]
        
        message = log_line
        for pattern in timestamp_patterns:
            message = re.sub(pattern, '', message)
            
        # Remove log level prefixes
        level_prefixes = [
            r'^\s*(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\s*[:\-\|]\s*',
            r'^\s*\[(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\]\s*',
            r'^\s*level[=:]\s*[a-zA-Z]+\s*'
        ]
        
        for pattern in level_prefixes:
            message = re.sub(pattern, '', message, flags=re.IGNORECASE)
            
        return message.strip()


class LogTemplateExtractor:
    """Extract log templates and patterns from log messages."""
    
    def __init__(self, min_support: int = 3, similarity_threshold: float = 0.8):
        """
        Initialize template extractor.
        
        Args:
            min_support: Minimum occurrences for a template
            similarity_threshold: Similarity threshold for template matching
        """
        self.min_support = min_support
        self.similarity_threshold = similarity_threshold
        self.templates = {}
        self.template_counter = 0
        
    def extract_templates(self, messages: List[str]) -> Dict[str, Any]:
        """
        Extract log templates from messages using pattern mining.
        
        Args:
            messages: List of log messages
            
        Returns:
            Dict with templates and their statistics
        """
        if not messages:
            return {}
            
        # Tokenize messages
        tokenized_messages = [self._tokenize_message(msg) for msg in messages]
        
        # Find common patterns
        patterns = self._find_patterns(tokenized_messages)
        
        # Create templates
        templates = {}
        template_id = 1
        
        for pattern, occurrences in patterns.items():
            if len(occurrences) >= self.min_support:
                template = self._pattern_to_template(pattern)
                templates[template_id] = {
                    'template': template,
                    'pattern': pattern,
                    'count': len(occurrences),
                    'examples': occurrences[:5]  # Keep first 5 examples
                }
                template_id += 1
                
        return templates
        
    def _tokenize_message(self, message: str) -> List[str]:
        """Tokenize log message into words and identify variables."""
        # Replace common variable patterns with placeholders
        variable_patterns = [
            (r'\b\d+\.\d+\.\d+\.\d+\b', '<IP>'),  # IP addresses
            (r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b', '<UUID>'),  # UUIDs
            (r'\b\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}', '<TIMESTAMP>'),  # Timestamps
            (r'\b\d+\b', '<NUMBER>'),  # Numbers
            (r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', '<EMAIL>'),  # Email addresses
            (r'\b/[^\s]*\b', '<PATH>'),  # File paths
            (r'\bhttps?://[^\s]+\b', '<URL>'),  # URLs
        ]
        
        processed_message = message
        for pattern, placeholder in variable_patterns:
            processed_message = re.sub(pattern, placeholder, processed_message)
            
        # Split into tokens
        tokens = re.findall(r'\S+', processed_message)
        return tokens
        
    def _find_patterns(self, tokenized_messages: List[List[str]]) -> Dict[Tuple, List[str]]:
        """Find common patterns in tokenized messages."""
        pattern_occurrences = defaultdict(list)
        
        for i, tokens in enumerate(tokenized_messages):
            # Create pattern from tokens
            pattern = tuple(tokens)
            pattern_occurrences[pattern].append(' '.join(tokens))
            
        return dict(pattern_occurrences)
        
    def _pattern_to_template(self, pattern: Tuple[str]) -> str:
        """Convert pattern tuple to template string."""
        return ' '.join(pattern)
        
    def match_message_to_template(self, message: str, templates: Dict[int, Dict]) -> Optional[int]:
        """
        Match a log message to existing templates.
        
        Args:
            message: Log message to match
            templates: Dict of templates
            
        Returns:
            Template ID if match found, None otherwise
        """
        tokenized = self._tokenize_message(message)
        message_pattern = tuple(tokenized)
        
        best_match_id = None
        best_similarity = 0.0
        
        for template_id, template_info in templates.items():
            pattern = template_info['pattern']
            similarity = self._calculate_similarity(message_pattern, pattern)
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match_id = template_id
                
        return best_match_id
        
    def _calculate_similarity(self, pattern1: Tuple, pattern2: Tuple) -> float:
        """Calculate similarity between two patterns."""
        if not pattern1 or not pattern2:
            return 0.0
            
        # Simple Jaccard similarity
        set1 = set(pattern1)
        set2 = set(pattern2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0


class LogCorrelator:
    """Correlate logs across services and time windows."""
    
    def __init__(self, correlation_window: int = 60):
        """
        Initialize log correlator.
        
        Args:
            correlation_window: Time window for correlation in seconds
        """
        self.correlation_window = correlation_window
        
    def correlate_by_trace_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Correlate logs by trace ID.
        
        Args:
            df: DataFrame with log data
            
        Returns:
            DataFrame with correlation information
        """
        if df.empty:
            return df
            
        df_corr = df.copy()
        df_corr['trace_id'] = df_corr['message'].apply(self._extract_trace_id)
        
        # Group by trace_id and add correlation info
        trace_groups = df_corr.groupby('trace_id')
        
        correlation_info = []
        for trace_id, group in trace_groups:
            if trace_id and len(group) > 1:  # Only correlate if trace_id exists and multiple logs
                services = group['service'].unique().tolist()
                span_duration = group['timestamp'].max() - group['timestamp'].min()
                
                for idx in group.index:
                    correlation_info.append({
                        'index': idx,
                        'correlated_services': services,
                        'trace_span_duration': span_duration,
                        'log_count_in_trace': len(group)
                    })
                    
        # Add correlation info to DataFrame
        if correlation_info:
            corr_df = pd.DataFrame(correlation_info).set_index('index')
            df_corr = df_corr.join(corr_df, how='left')
        else:
            df_corr['correlated_services'] = None
            df_corr['trace_span_duration'] = None
            df_corr['log_count_in_trace'] = None
            
        return df_corr
        
    def correlate_by_time_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Correlate logs within time windows.
        
        Args:
            df: DataFrame with log data
            
        Returns:
            DataFrame with time-based correlation
        """
        if df.empty:
            return df
            
        df_sorted = df.sort_values('timestamp')
        df_corr = df_sorted.copy()
        df_corr['time_window'] = df_corr['timestamp'] // self.correlation_window
        
        # Group by time window and add correlation info
        window_groups = df_corr.groupby('time_window')
        
        correlation_info = []
        for window, group in window_groups:
            services = group['service'].unique().tolist()
            error_count = len(group[group['level'].isin(['ERROR', 'FATAL', 'CRITICAL'])])
            
            for idx in group.index:
                correlation_info.append({
                    'index': idx,
                    'window_services': services,
                    'window_error_count': error_count,
                    'window_log_count': len(group)
                })
                
        # Add correlation info to DataFrame
        if correlation_info:
            corr_df = pd.DataFrame(correlation_info).set_index('index')
            df_corr = df_corr.join(corr_df, how='left')
        else:
            df_corr['window_services'] = None
            df_corr['window_error_count'] = None
            df_corr['window_log_count'] = None
            
        return df_corr.drop('time_window', axis=1)
        
    def _extract_trace_id(self, message: str) -> Optional[str]:
        """Extract trace ID from log message."""
        # Common trace ID patterns
        trace_patterns = [
            r'trace[_-]?id[=:\s]+([a-fA-F0-9]{16,32})',
            r'traceId[=:\s]+([a-fA-F0-9]{16,32})',
            r'trace[=:\s]+([a-fA-F0-9]{16,32})',
            r'\b([a-fA-F0-9]{32})\b',  # 32-char hex string
            r'\b([a-fA-F0-9]{16})\b'   # 16-char hex string
        ]
        
        for pattern in trace_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1)
                
        return None


class LogsCollector:
    """High-level logs collector with built-in processing."""
    
    def __init__(self, loki_url: str = "http://localhost:3100"):
        """Initialize logs collector."""
        self.client = LokiClient(loki_url)
        self.template_extractor = LogTemplateExtractor()
        self.correlator = LogCorrelator()
        
    def collect_service_logs(self, start_time: datetime, end_time: datetime,
                           services: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Collect logs for specified services.
        
        Args:
            start_time: Collection start time
            end_time: Collection end time
            services: List of services to collect (None for all)
            
        Returns:
            DataFrame with processed logs data
        """
        # Build LogQL query
        if services:
            service_filter = '|'.join([f'service="{svc}"' for svc in services])
            query = f'{{{service_filter}}}'
        else:
            query = '{job=~".+"}'  # Match all jobs
            
        try:
            logger.info(f"Collecting logs with query: {query}")
            df = self.client.query_range(query, start_time, end_time, limit=10000)
            
            if df.empty:
                logger.warning("No logs collected")
                return df
                
            # Add correlation information
            df = self.correlator.correlate_by_trace_id(df)
            df = self.correlator.correlate_by_time_window(df)
            
            return df.sort_values(['timestamp', 'service'])
            
        except Exception as e:
            logger.error(f"Failed to collect logs: {e}")
            return pd.DataFrame()
            
    def extract_log_templates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract log templates from collected logs.
        
        Args:
            df: DataFrame with log messages
            
        Returns:
            Dict with extracted templates and statistics
        """
        if df.empty:
            return {}
            
        # Extract templates per service
        service_templates = {}
        
        for service, group in df.groupby('service'):
            messages = group['message'].tolist()
            templates = self.template_extractor.extract_templates(messages)
            
            if templates:
                service_templates[service] = templates
                
        return service_templates
        
    def add_template_ids(self, df: pd.DataFrame, templates: Dict[str, Any]) -> pd.DataFrame:
        """
        Add template IDs to log DataFrame.
        
        Args:
            df: DataFrame with log data
            templates: Dict of extracted templates
            
        Returns:
            DataFrame with template_id column
        """
        if df.empty:
            return df
            
        df_with_templates = df.copy()
        df_with_templates['template_id'] = None
        
        for service, service_templates in templates.items():
            service_mask = df_with_templates['service'] == service
            service_logs = df_with_templates[service_mask]
            
            for idx in service_logs.index:
                message = df_with_templates.loc[idx, 'message']
                template_id = self.template_extractor.match_message_to_template(
                    message, service_templates
                )
                if template_id:
                    df_with_templates.loc[idx, 'template_id'] = template_id
                    
        return df_with_templates