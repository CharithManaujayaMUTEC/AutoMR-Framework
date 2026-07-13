"""
Epsilon analysis package.

This package provides functionality for evaluating the effect of
different epsilon (tolerance) values on metamorphic testing results.
It includes tools for epsilon sensitivity analysis, result
summarization, and utility functions for managing relation tolerances.
"""

from .summary import EpsilonSummary
from .sensitivity import EpsilonSensitivity
from .utils import (
    generate_epsilon_values,
    apply_epsilon_to_relations,
)

# Public package interface.
__all__ = [
    "EpsilonSensitivity",
    "EpsilonSummary",
    "generate_epsilon_values",
    "apply_epsilon_to_relations",
]