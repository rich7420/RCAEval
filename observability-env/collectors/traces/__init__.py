"""
Traces collection module for Jaeger data collection and processing.
Implements requirements 4.3 and 6.4 for comprehensive traces collection.
"""

from .jaeger_client import JaegerClient, TraceProcessor, TraceCorrelator, TracesCollector
from .analyzer import TraceAnalyzer, TraceFlowAnalyzer

__all__ = [
    'JaegerClient',
    'TraceProcessor',
    'TraceCorrelator',
    'TracesCollector',
    'TraceAnalyzer',
    'TraceFlowAnalyzer'
]