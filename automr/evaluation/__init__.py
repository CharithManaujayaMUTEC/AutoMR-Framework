"""
Evaluation package.

This package provides utilities for generating and managing
baseline evaluation artifacts, including cached predictions,
dataset information, model summaries, and baseline metrics.
"""

from .baseline import BaselineEvaluator
from .graph_generator import GraphGenerator

# Public package interface.
__all__ = [
    "BaselineEvaluator",
    "GraphGenerator",
]