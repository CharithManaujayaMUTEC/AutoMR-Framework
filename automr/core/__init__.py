"""
Core components of the AutoMR framework.

This package contains the primary execution and analysis modules
responsible for metamorphic testing, prediction management,
range evaluation, visualization, and failure analysis.
"""

from .failure_analysis import FailureAnalyzer
from .model_wrapper import ModelWrapper
from .mr_executor import MRExecutor
from .range_tester import RangeTester
from .tester import MRTester

__all__ = [
    "FailureAnalyzer",
    "ModelWrapper",
    "MRExecutor",
    "RangeTester",
    "MRTester",
]