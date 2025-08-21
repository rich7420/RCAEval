"""
Logs collection module for Loki data collection and processing.
Implements requirements 4.2 and 6.4 for comprehensive logs collection.
"""

from .loki_client import LokiClient, LogTemplateExtractor, LogCorrelator, LogsCollector
from .clustering import LogClusterer, LogPatternAnalyzer

__all__ = [
    'LokiClient',
    'LogTemplateExtractor',
    'LogCorrelator',
    'LogsCollector',
    'LogClusterer',
    'LogPatternAnalyzer'
]