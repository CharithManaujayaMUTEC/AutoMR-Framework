"""
Evaluation package.

This package provides utilities for generating and managing
baseline evaluation artifacts, including cached predictions,
dataset information, model summaries, and baseline metrics.
"""

from .baseline import BaselineEvaluator

# Public package interface.
__all__ = [
    "BaselineEvaluator",
]