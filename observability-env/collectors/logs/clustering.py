"""
Log clustering and pattern analysis for log template extraction.
Advanced clustering algorithms for log message analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
import logging
import re
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
import difflib

logger = logging.getLogger(__name__)


class LogClusterer:
    """Advanced log clustering using multiple algorithms."""
    
    def __init__(self, method: str = "drain", min_cluster_size: int = 3):
        """
        Initialize log clusterer.
        
        Args:
            method: Clustering method ("drain", "tfidf", "edit_distance")
            min_cluster_size: Minimum size for a cluster
        """
        self.method = method
        self.min_cluster_size = min_cluster_size
        self.clusters = {}
        self.templates = {}
        
    def cluster_logs(self, messages: List[str]) -> Dict[int, Dict[str, Any]]:
        """
        Cluster log messages and extract templates.
        
        Args:
            messages: List of log messages
            
        Returns:
            Dict with cluster information and templates
        """
        if not messages:
            return {}
            
        if self.method == "drain":
            return self._drain_clustering(messages)
        elif self.method == "tfidf":
            return self._tfidf_clustering(messages)
        elif self.method == "edit_distance":
            return self._edit_distance_clustering(messages)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
            
    def _drain_clustering(self, messages: List[str]) -> Dict[int, Dict[str, Any]]:
        """
        Implement Drain algorithm for log clustering.
        
        Args:
            messages: List of log messages
            
        Returns:
            Dict with cluster information
        """
        # Simplified Drain implementation
        clusters = {}
        cluster_id = 1
        
        # Parse tree structure (simplified)
        parse_tree = defaultdict(lambda: defaultdict(list))
        
        for message in messages:
            tokens = self._preprocess_message(message)
            
            if not tokens:
                continue
                
            # Find matching cluster based on length and first token
            length = len(tokens)
            first_token = tokens[0] if tokens else ""
            
            # Look for existing cluster
            matched_cluster = None
            best_similarity = 0.0
            
            for existing_tokens in parse_tree[length][first_token]:
                similarity = self._calculate_token_similarity(tokens, existing_tokens)
                if similarity > best_similarity and similarity > 0.7:  # Threshold
                    best_similarity = similarity
                    matched_cluster = existing_tokens
                    
            if matched_cluster:
                # Add to existing cluster
                cluster_key = tuple(matched_cluster)
                if cluster_key not in clusters:
                    clusters[cluster_key] = {
                        'id': cluster_id,
                        'template': self._create_template(matched_cluster),
                        'messages': [],
                        'count': 0
                    }
                    cluster_id += 1
                    
                clusters[cluster_key]['messages'].append(message)
                clusters[cluster_key]['count'] += 1
            else:
                # Create new cluster
                parse_tree[length][first_token].append(tokens)
                cluster_key = tuple(tokens)
                clusters[cluster_key] = {
                    'id': cluster_id,
                    'template': self._create_template(tokens),
                    'messages': [message],
                    'count': 1
                }
                cluster_id += 1
                
        # Filter clusters by minimum size and convert to final format
        final_clusters = {}
        for cluster_key, cluster_info in clusters.items():
            if cluster_info['count'] >= self.min_cluster_size:
                final_clusters[cluster_info['id']] = {
                    'template': cluster_info['template'],
                    'count': cluster_info['count'],
                    'messages': cluster_info['messages'][:10],  # Keep sample messages
                    'pattern': list(cluster_key)
                }
                
        return final_clusters
        
    def _tfidf_clustering(self, messages: List[str]) -> Dict[int, Dict[str, Any]]:
        """
        Use TF-IDF and clustering for log grouping.
        
        Args:
            messages: List of log messages
            
        Returns:
            Dict with cluster information
        """
        if len(messages) < self.min_cluster_size:
            return {}
            
        # Preprocess messages
        processed_messages = [self._preprocess_message_string(msg) for msg in messages]
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(processed_messages)
        except ValueError:
            # Not enough features
            return {}
            
        # Determine optimal number of clusters
        n_clusters = min(max(2, len(messages) // 10), 20)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Group messages by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append((i, messages[i]))
            
        # Create final cluster format
        final_clusters = {}
        cluster_id = 1
        
        for label, message_list in clusters.items():
            if len(message_list) >= self.min_cluster_size:
                # Extract template from cluster
                cluster_messages = [msg for _, msg in message_list]
                template = self._extract_template_from_cluster(cluster_messages)
                
                final_clusters[cluster_id] = {
                    'template': template,
                    'count': len(message_list),
                    'messages': cluster_messages[:10],
                    'pattern': self._preprocess_message(template)
                }
                cluster_id += 1
                
        return final_clusters
        
    def _edit_distance_clustering(self, messages: List[str]) -> Dict[int, Dict[str, Any]]:
        """
        Use edit distance for log clustering.
        
        Args:
            messages: List of log messages
            
        Returns:
            Dict with cluster information
        """
        if len(messages) < self.min_cluster_size:
            return {}
            
        # Calculate pairwise edit distances
        n = len(messages)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._normalized_edit_distance(messages[i], messages[j])
                distance_matrix[i][j] = dist
                distance_matrix[j][i] = dist
                
        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=0.3, min_samples=self.min_cluster_size, metric='precomputed')
        cluster_labels = dbscan.fit_predict(distance_matrix)
        
        # Group messages by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            if label != -1:  # Ignore noise points
                clusters[label].append(messages[i])
                
        # Create final cluster format
        final_clusters = {}
        cluster_id = 1
        
        for label, cluster_messages in clusters.items():
            if len(cluster_messages) >= self.min_cluster_size:
                template = self._extract_template_from_cluster(cluster_messages)
                
                final_clusters[cluster_id] = {
                    'template': template,
                    'count': len(cluster_messages),
                    'messages': cluster_messages[:10],
                    'pattern': self._preprocess_message(template)
                }
                cluster_id += 1
                
        return final_clusters
        
    def _preprocess_message(self, message: str) -> List[str]:
        """Preprocess log message into tokens with variable replacement."""
        # Replace common variable patterns
        variable_patterns = [
            (r'\b\d+\.\d+\.\d+\.\d+\b', '<IP>'),
            (r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b', '<UUID>'),
            (r'\b\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}', '<TIMESTAMP>'),
            (r'\b\d+\.\d+\b', '<FLOAT>'),
            (r'\b\d+\b', '<NUMBER>'),
            (r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', '<EMAIL>'),
            (r'\b/[^\s]*\b', '<PATH>'),
            (r'\bhttps?://[^\s]+\b', '<URL>'),
            (r'\b[a-fA-F0-9]{32,}\b', '<HASH>'),
        ]
        
        processed = message
        for pattern, replacement in variable_patterns:
            processed = re.sub(pattern, replacement, processed)
            
        # Tokenize
        tokens = re.findall(r'\S+', processed.lower())
        return tokens
        
    def _preprocess_message_string(self, message: str) -> str:
        """Preprocess message for TF-IDF (return string)."""
        tokens = self._preprocess_message(message)
        return ' '.join(tokens)
        
    def _calculate_token_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Calculate similarity between token lists."""
        if not tokens1 or not tokens2:
            return 0.0
            
        # Simple token-based similarity
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
        
    def _normalized_edit_distance(self, str1: str, str2: str) -> float:
        """Calculate normalized edit distance between strings."""
        distance = difflib.SequenceMatcher(None, str1, str2).ratio()
        return 1.0 - distance  # Convert similarity to distance
        
    def _create_template(self, tokens: List[str]) -> str:
        """Create template from tokens."""
        return ' '.join(tokens)
        
    def _extract_template_from_cluster(self, messages: List[str]) -> str:
        """Extract common template from cluster of messages."""
        if not messages:
            return ""
            
        if len(messages) == 1:
            return messages[0]
            
        # Find common subsequences
        tokenized_messages = [self._preprocess_message(msg) for msg in messages]
        
        # Find the most common pattern
        if not tokenized_messages:
            return messages[0]
            
        # Use the first message as base and find common parts
        base_tokens = tokenized_messages[0]
        common_template = []
        
        for i, token in enumerate(base_tokens):
            # Check if this token appears in same position in most messages
            token_count = sum(1 for tokens in tokenized_messages 
                            if i < len(tokens) and tokens[i] == token)
            
            if token_count >= len(tokenized_messages) * 0.7:  # 70% threshold
                common_template.append(token)
            else:
                common_template.append('<*>')  # Wildcard for variable parts
                
        return ' '.join(common_template)


class LogPatternAnalyzer:
    """Analyze patterns in log clusters."""
    
    def __init__(self):
        """Initialize pattern analyzer."""
        self.pattern_stats = {}
        
    def analyze_cluster_patterns(self, clusters: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze patterns across log clusters.
        
        Args:
            clusters: Dict of log clusters
            
        Returns:
            Dict with pattern analysis results
        """
        if not clusters:
            return {}
            
        analysis = {
            'total_clusters': len(clusters),
            'total_messages': sum(c['count'] for c in clusters.values()),
            'cluster_sizes': [c['count'] for c in clusters.values()],
            'template_lengths': [],
            'common_tokens': Counter(),
            'pattern_types': defaultdict(int)
        }
        
        for cluster_id, cluster_info in clusters.items():
            template = cluster_info['template']
            tokens = template.split()
            
            analysis['template_lengths'].append(len(tokens))
            analysis['common_tokens'].update(tokens)
            
            # Classify pattern types
            pattern_type = self._classify_pattern(template)
            analysis['pattern_types'][pattern_type] += 1
            
        # Calculate statistics
        if analysis['cluster_sizes']:
            analysis['avg_cluster_size'] = np.mean(analysis['cluster_sizes'])
            analysis['median_cluster_size'] = np.median(analysis['cluster_sizes'])
            analysis['max_cluster_size'] = max(analysis['cluster_sizes'])
            
        if analysis['template_lengths']:
            analysis['avg_template_length'] = np.mean(analysis['template_lengths'])
            
        # Get most common tokens
        analysis['most_common_tokens'] = analysis['common_tokens'].most_common(20)
        
        return analysis
        
    def _classify_pattern(self, template: str) -> str:
        """Classify log pattern type."""
        template_lower = template.lower()
        
        if 'error' in template_lower or 'exception' in template_lower:
            return 'error'
        elif 'warn' in template_lower or 'warning' in template_lower:
            return 'warning'
        elif 'info' in template_lower or 'debug' in template_lower:
            return 'info'
        elif 'request' in template_lower or 'response' in template_lower:
            return 'http'
        elif 'start' in template_lower or 'stop' in template_lower or 'init' in template_lower:
            return 'lifecycle'
        elif '<number>' in template_lower or '<float>' in template_lower:
            return 'metric'
        else:
            return 'other'
            
    def find_anomalous_patterns(self, clusters: Dict[int, Dict[str, Any]], 
                              threshold: float = 0.05) -> List[Dict[str, Any]]:
        """
        Find anomalous log patterns (rare clusters).
        
        Args:
            clusters: Dict of log clusters
            threshold: Threshold for considering a pattern anomalous
            
        Returns:
            List of anomalous patterns
        """
        if not clusters:
            return []
            
        total_messages = sum(c['count'] for c in clusters.values())
        anomalous_patterns = []
        
        for cluster_id, cluster_info in clusters.items():
            frequency = cluster_info['count'] / total_messages
            
            if frequency < threshold:
                anomalous_patterns.append({
                    'cluster_id': cluster_id,
                    'template': cluster_info['template'],
                    'count': cluster_info['count'],
                    'frequency': frequency,
                    'sample_messages': cluster_info.get('messages', [])[:3]
                })
                
        # Sort by frequency (rarest first)
        anomalous_patterns.sort(key=lambda x: x['frequency'])
        
        return anomalous_patterns
        
    def detect_error_patterns(self, clusters: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect error-related log patterns.
        
        Args:
            clusters: Dict of log clusters
            
        Returns:
            List of error patterns
        """
        error_patterns = []
        
        error_keywords = [
            'error', 'exception', 'fail', 'fatal', 'critical', 'panic',
            'timeout', 'refused', 'denied', 'invalid', 'corrupt'
        ]
        
        for cluster_id, cluster_info in clusters.items():
            template = cluster_info['template'].lower()
            
            if any(keyword in template for keyword in error_keywords):
                error_patterns.append({
                    'cluster_id': cluster_id,
                    'template': cluster_info['template'],
                    'count': cluster_info['count'],
                    'error_type': self._identify_error_type(template),
                    'sample_messages': cluster_info.get('messages', [])[:3]
                })
                
        # Sort by count (most frequent errors first)
        error_patterns.sort(key=lambda x: x['count'], reverse=True)
        
        return error_patterns
        
    def _identify_error_type(self, template: str) -> str:
        """Identify specific error type from template."""
        template_lower = template.lower()
        
        if 'timeout' in template_lower:
            return 'timeout'
        elif 'connection' in template_lower and ('refused' in template_lower or 'failed' in template_lower):
            return 'connection_error'
        elif 'permission' in template_lower or 'denied' in template_lower:
            return 'permission_error'
        elif 'not found' in template_lower or '404' in template_lower:
            return 'not_found'
        elif 'invalid' in template_lower or 'malformed' in template_lower:
            return 'validation_error'
        elif 'memory' in template_lower or 'oom' in template_lower:
            return 'memory_error'
        elif 'disk' in template_lower or 'space' in template_lower:
            return 'disk_error'
        else:
            return 'general_error'