"""
Evaluation package.

This package provides utilities for generating and managing
baseline evaluation artifacts, including cached predictions,
dataset information, model summaries, and baseline metrics.
"""

from .baseline import BaselineEvaluator
from .graph_generator import GraphGenerator
from .decoder_health import DecoderHealthAnalyzer
from .final_report import FinalEvaluationReport

FinalEvaluationReport = import_module(
    ".final_report", __name__
).FinalEvaluationReport

# Public package interface.
__all__ = [
    "BaselineEvaluator",
    "GraphGenerator",
    "DecoderHealthAnalyzer",
    "FinalEvaluationReport",
]