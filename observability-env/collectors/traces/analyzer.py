"""
Advanced trace analysis for dependency mapping and performance insights.
Implements sophisticated trace processing and correlation analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
import logging
from collections import defaultdict, Counter, deque
import networkx as nx
from scipy import stats

logger = logging.getLogger(__name__)


class TraceAnalyzer:
    """Advanced trace analysis with dependency mapping and performance insights."""
    
    def __init__(self):
        """Initialize trace analyzer."""
        self.dependency_graph = nx.DiGraph()
        self.service_metrics = {}
        
    def build_service_dependency_graph(self, df: pd.DataFrame) -> nx.DiGraph:
        """
        Build service dependency graph from traces.
        
        Args:
            df: DataFrame with trace data
            
        Returns:
            NetworkX directed graph of service dependencies
        """
        graph = nx.DiGraph()
        
        if df.empty:
            return graph
            
        # Process each trace to build dependencies
        for trace_id, trace_group in df.groupby('trace_id'):
            spans = trace_group.sort_values('start_time')
            
            # Build parent-child relationships
            span_to_service = dict(zip(spans['span_id'], spans['service']))
            
            for _, span in spans.iterrows():
                service = span['service']
                parent_span_id = span['parent_span_id']
                
                # Add service node
                if not graph.has_node(service):
                    graph.add_node(service, call_count=0, error_count=0, total_duration=0)
                
                # Update service metrics
                graph.nodes[service]['call_count'] += 1
                graph.nodes[service]['total_duration'] += span['duration_ms']
                
                if span['error']:
                    graph.nodes[service]['error_count'] += 1
                
                # Add dependency edge
                if parent_span_id and parent_span_id in span_to_service:
                    parent_service = span_to_service[parent_span_id]
                    
                    if parent_service != service:  # Avoid self-loops
                        if graph.has_edge(parent_service, service):
                            graph.edges[parent_service, service]['weight'] += 1
                            graph.edges[parent_service, service]['total_duration'] += span['duration_ms']
                        else:
                            graph.add_edge(parent_service, service, weight=1, 
                                         total_duration=span['duration_ms'])
                            
        # Calculate derived metrics
        for node in graph.nodes():
            node_data = graph.nodes[node]
            if node_data['call_count'] > 0:
                node_data['avg_duration'] = node_data['total_duration'] / node_data['call_count']
                node_data['error_rate'] = node_data['error_count'] / node_data['call_count']
            else:
                node_data['avg_duration'] = 0
                node_data['error_rate'] = 0
                
        for edge in graph.edges():
            edge_data = graph.edges[edge]
            if edge_data['weight'] > 0:
                edge_data['avg_duration'] = edge_data['total_duration'] / edge_data['weight']
            else:
                edge_data['avg_duration'] = 0
                
        self.dependency_graph = graph
        return graph
        
    def identify_critical_path(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify critical paths in traces (longest duration paths).
        
        Args:
            df: DataFrame with trace data
            
        Returns:
            List of critical path information
        """
        critical_paths = []
        
        if df.empty:
            return critical_paths
            
        for trace_id, trace_group in df.groupby('trace_id'):
            spans = trace_group.sort_values('start_time')
            
            # Build span hierarchy
            span_hierarchy = {}
            root_spans = []
            
            for _, span in spans.iterrows():
                span_id = span['span_id']
                parent_id = span['parent_span_id']
                
                span_info = {
                    'span_id': span_id,
                    'service': span['service'],
                    'operation': span['operation'],
                    'duration_ms': span['duration_ms'],
                    'start_time': span['start_time'],
                    'children': []
                }
                
                span_hierarchy[span_id] = span_info
                
                if not parent_id:
                    root_spans.append(span_info)
                    
            # Build parent-child relationships
            for _, span in spans.iterrows():
                span_id = span['span_id']
                parent_id = span['parent_span_id']
                
                if parent_id and parent_id in span_hierarchy:
                    span_hierarchy[parent_id]['children'].append(span_hierarchy[span_id])
                    
            # Find critical path for each root span
            for root_span in root_spans:
                critical_path = self._find_critical_path_recursive(root_span)
                
                if critical_path:
                    total_duration = sum(span['duration_ms'] for span in critical_path)
                    critical_paths.append({
                        'trace_id': trace_id,
                        'path': critical_path,
                        'total_duration_ms': total_duration,
                        'path_length': len(critical_path)
                    })
                    
        # Sort by total duration (longest first)
        critical_paths.sort(key=lambda x: x['total_duration_ms'], reverse=True)
        
        return critical_paths
        
    def _find_critical_path_recursive(self, span: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recursively find the critical path from a span."""
        if not span['children']:
            return [span]
            
        # Find the child with the longest critical path
        longest_path = []
        longest_duration = 0
        
        for child in span['children']:
            child_path = self._find_critical_path_recursive(child)
            child_duration = sum(s['duration_ms'] for s in child_path)
            
            if child_duration > longest_duration:
                longest_duration = child_duration
                longest_path = child_path
                
        return [span] + longest_path
        
    def detect_performance_anomalies(self, df: pd.DataFrame, 
                                   z_threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect performance anomalies in traces using statistical analysis.
        
        Args:
            df: DataFrame with trace data
            z_threshold: Z-score threshold for anomaly detection
            
        Returns:
            DataFrame with anomaly information
        """
        if df.empty:
            return pd.DataFrame()
            
        anomalies = []
        
        # Analyze by service and operation
        for (service, operation), group in df.groupby(['service', 'operation']):
            durations = group['duration_ms'].dropna()
            
            if len(durations) < 10:  # Need sufficient data
                continue
                
            # Calculate z-scores
            mean_duration = durations.mean()
            std_duration = durations.std()
            
            if std_duration == 0:  # No variation
                continue
                
            z_scores = np.abs((durations - mean_duration) / std_duration)
            anomaly_mask = z_scores > z_threshold
            
            anomalous_spans = group[anomaly_mask]
            
            for _, span in anomalous_spans.iterrows():
                anomalies.append({
                    'trace_id': span['trace_id'],
                    'span_id': span['span_id'],
                    'service': service,
                    'operation': operation,
                    'duration_ms': span['duration_ms'],
                    'expected_duration_ms': mean_duration,
                    'z_score': z_scores[span.name],
                    'anomaly_type': 'slow' if span['duration_ms'] > mean_duration else 'fast'
                })
                
        return pd.DataFrame(anomalies)
        
    def analyze_service_health(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Analyze health metrics for each service.
        
        Args:
            df: DataFrame with trace data
            
        Returns:
            Dict with service health metrics
        """
        service_health = {}
        
        if df.empty:
            return service_health
            
        for service, group in df.groupby('service'):
            durations = group['duration_ms'].dropna()
            errors = group['error'].sum()
            total_spans = len(group)
            
            health_metrics = {
                'total_spans': total_spans,
                'error_count': errors,
                'error_rate': errors / total_spans if total_spans > 0 else 0,
                'avg_duration_ms': durations.mean() if len(durations) > 0 else 0,
                'median_duration_ms': durations.median() if len(durations) > 0 else 0,
                'p95_duration_ms': durations.quantile(0.95) if len(durations) > 0 else 0,
                'p99_duration_ms': durations.quantile(0.99) if len(durations) > 0 else 0,
                'max_duration_ms': durations.max() if len(durations) > 0 else 0,
                'min_duration_ms': durations.min() if len(durations) > 0 else 0,
                'duration_std': durations.std() if len(durations) > 0 else 0
            }
            
            # Calculate health score (0-100)
            error_score = max(0, 100 - (health_metrics['error_rate'] * 100))
            
            # Performance score based on consistency (lower std is better)
            if health_metrics['avg_duration_ms'] > 0:
                cv = health_metrics['duration_std'] / health_metrics['avg_duration_ms']
                performance_score = max(0, 100 - (cv * 50))  # Scale coefficient of variation
            else:
                performance_score = 100
                
            health_metrics['health_score'] = (error_score * 0.6 + performance_score * 0.4)
            
            service_health[service] = health_metrics
            
        return service_health
        
    def find_bottlenecks(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify bottlenecks in service dependencies.
        
        Args:
            df: DataFrame with trace data
            
        Returns:
            List of bottleneck information
        """
        bottlenecks = []
        
        if df.empty:
            return bottlenecks
            
        # Build dependency graph if not already built
        if not self.dependency_graph.nodes():
            self.build_service_dependency_graph(df)
            
        # Analyze each service for bottleneck characteristics
        for service in self.dependency_graph.nodes():
            node_data = self.dependency_graph.nodes[service]
            
            # Calculate incoming and outgoing dependencies
            incoming_edges = list(self.dependency_graph.predecessors(service))
            outgoing_edges = list(self.dependency_graph.successors(service))
            
            # Bottleneck indicators
            high_error_rate = node_data['error_rate'] > 0.05  # 5% error rate
            high_avg_duration = node_data['avg_duration'] > 1000  # 1 second
            high_call_volume = node_data['call_count'] > 100
            many_dependencies = len(outgoing_edges) > 5
            
            bottleneck_score = 0
            reasons = []
            
            if high_error_rate:
                bottleneck_score += 30
                reasons.append(f"High error rate: {node_data['error_rate']:.2%}")
                
            if high_avg_duration:
                bottleneck_score += 25
                reasons.append(f"High average duration: {node_data['avg_duration']:.1f}ms")
                
            if high_call_volume:
                bottleneck_score += 20
                reasons.append(f"High call volume: {node_data['call_count']}")
                
            if many_dependencies:
                bottleneck_score += 15
                reasons.append(f"Many dependencies: {len(outgoing_edges)}")
                
            # Fan-in bottleneck (many services depend on this one)
            if len(incoming_edges) > 3:
                bottleneck_score += 10
                reasons.append(f"High fan-in: {len(incoming_edges)} services depend on this")
                
            if bottleneck_score > 30:  # Threshold for considering as bottleneck
                bottlenecks.append({
                    'service': service,
                    'bottleneck_score': bottleneck_score,
                    'reasons': reasons,
                    'metrics': node_data,
                    'incoming_dependencies': incoming_edges,
                    'outgoing_dependencies': outgoing_edges
                })
                
        # Sort by bottleneck score (highest first)
        bottlenecks.sort(key=lambda x: x['bottleneck_score'], reverse=True)
        
        return bottlenecks
        
    def analyze_trace_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze common patterns in traces.
        
        Args:
            df: DataFrame with trace data
            
        Returns:
            Dict with pattern analysis
        """
        if df.empty:
            return {}
            
        patterns = {
            'common_service_sequences': [],
            'error_patterns': [],
            'performance_patterns': [],
            'temporal_patterns': {}
        }
        
        # Analyze service call sequences
        service_sequences = []
        for trace_id, trace_group in df.groupby('trace_id'):
            spans = trace_group.sort_values('start_time')
            sequence = spans['service'].tolist()
            service_sequences.append(tuple(sequence))
            
        # Find most common sequences
        sequence_counts = Counter(service_sequences)
        patterns['common_service_sequences'] = [
            {'sequence': list(seq), 'count': count}
            for seq, count in sequence_counts.most_common(10)
        ]
        
        # Analyze error patterns
        error_traces = df[df['error'] == True]
        if not error_traces.empty:
            error_services = error_traces['service'].value_counts()
            error_operations = error_traces['operation'].value_counts()
            
            patterns['error_patterns'] = {
                'most_error_prone_services': error_services.head(5).to_dict(),
                'most_error_prone_operations': error_operations.head(5).to_dict(),
                'total_error_spans': len(error_traces)
            }
            
        # Analyze performance patterns
        slow_threshold = df['duration_ms'].quantile(0.95)
        slow_traces = df[df['duration_ms'] >= slow_threshold]
        
        if not slow_traces.empty:
            slow_services = slow_traces['service'].value_counts()
            slow_operations = slow_traces['operation'].value_counts()
            
            patterns['performance_patterns'] = {
                'slowest_services': slow_services.head(5).to_dict(),
                'slowest_operations': slow_operations.head(5).to_dict(),
                'slow_threshold_ms': slow_threshold
            }
            
        # Analyze temporal patterns
        df_with_hour = df.copy()
        df_with_hour['hour'] = pd.to_datetime(df_with_hour['start_time'], unit='s').dt.hour
        
        hourly_stats = df_with_hour.groupby('hour').agg({
            'duration_ms': ['mean', 'count'],
            'error': 'sum'
        }).round(2)
        
        patterns['temporal_patterns'] = {
            'hourly_avg_duration': hourly_stats['duration_ms']['mean'].to_dict(),
            'hourly_request_count': hourly_stats['duration_ms']['count'].to_dict(),
            'hourly_error_count': hourly_stats['error']['sum'].to_dict()
        }
        
        return patterns


class TraceFlowAnalyzer:
    """Analyze trace flows and request paths through services."""
    
    def __init__(self):
        """Initialize trace flow analyzer."""
        self.flow_patterns = {}
        
    def analyze_request_flows(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze request flows through service architecture.
        
        Args:
            df: DataFrame with trace data
            
        Returns:
            Dict with flow analysis
        """
        if df.empty:
            return {}
            
        flows = {
            'entry_points': Counter(),
            'exit_points': Counter(),
            'flow_paths': [],
            'flow_statistics': {}
        }
        
        for trace_id, trace_group in df.groupby('trace_id'):
            spans = trace_group.sort_values('start_time')
            
            if spans.empty:
                continue
                
            # Identify entry point (first span or span without parent)
            root_spans = spans[spans['parent_span_id'].isna() | (spans['parent_span_id'] == '')]
            if not root_spans.empty:
                entry_service = root_spans.iloc[0]['service']
                flows['entry_points'][entry_service] += 1
                
            # Identify exit point (last span chronologically)
            exit_service = spans.iloc[-1]['service']
            flows['exit_points'][exit_service] += 1
            
            # Build flow path
            flow_path = self._build_flow_path(spans)
            if flow_path:
                flows['flow_paths'].append({
                    'trace_id': trace_id,
                    'path': flow_path,
                    'total_duration_ms': spans['duration_ms'].sum(),
                    'span_count': len(spans),
                    'has_errors': spans['error'].any()
                })
                
        # Calculate flow statistics
        if flows['flow_paths']:
            path_lengths = [len(flow['path']) for flow in flows['flow_paths']]
            durations = [flow['total_duration_ms'] for flow in flows['flow_paths']]
            error_flows = [flow for flow in flows['flow_paths'] if flow['has_errors']]
            
            flows['flow_statistics'] = {
                'total_flows': len(flows['flow_paths']),
                'avg_path_length': np.mean(path_lengths),
                'avg_flow_duration_ms': np.mean(durations),
                'error_flow_count': len(error_flows),
                'error_flow_rate': len(error_flows) / len(flows['flow_paths'])
            }
            
        return flows
        
    def _build_flow_path(self, spans: pd.DataFrame) -> List[Dict[str, Any]]:
        """Build flow path from spans."""
        if spans.empty:
            return []
            
        # Sort by start time to get chronological order
        sorted_spans = spans.sort_values('start_time')
        
        path = []
        for _, span in sorted_spans.iterrows():
            path.append({
                'service': span['service'],
                'operation': span['operation'],
                'duration_ms': span['duration_ms'],
                'error': span['error']
            })
            
        return path
        
    def identify_common_patterns(self, flows: Dict[str, Any], min_occurrences: int = 3) -> List[Dict[str, Any]]:
        """
        Identify common flow patterns.
        
        Args:
            flows: Flow analysis results
            min_occurrences: Minimum occurrences to consider a pattern
            
        Returns:
            List of common patterns
        """
        if not flows.get('flow_paths'):
            return []
            
        # Extract service sequences from flows
        service_sequences = []
        for flow in flows['flow_paths']:
            sequence = tuple(step['service'] for step in flow['path'])
            service_sequences.append(sequence)
            
        # Count pattern occurrences
        pattern_counts = Counter(service_sequences)
        
        # Filter by minimum occurrences and create pattern info
        common_patterns = []
        for pattern, count in pattern_counts.items():
            if count >= min_occurrences:
                # Calculate statistics for this pattern
                pattern_flows = [flow for flow in flows['flow_paths'] 
                               if tuple(step['service'] for step in flow['path']) == pattern]
                
                durations = [flow['total_duration_ms'] for flow in pattern_flows]
                error_count = sum(1 for flow in pattern_flows if flow['has_errors'])
                
                common_patterns.append({
                    'pattern': list(pattern),
                    'occurrences': count,
                    'avg_duration_ms': np.mean(durations),
                    'error_count': error_count,
                    'error_rate': error_count / count,
                    'example_trace_ids': [flow['trace_id'] for flow in pattern_flows[:3]]
                })
                
        # Sort by occurrences (most common first)
        common_patterns.sort(key=lambda x: x['occurrences'], reverse=True)
        
        return common_patterns