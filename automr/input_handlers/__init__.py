"""
Input handlers package.

This package provides input handlers for different data modalities,
including image, tabular, and text data. Each handler implements
a common interface for validation, preprocessing, and batching.
"""

from .handler_factory import get_handler

# Public package interface.
__all__ = [
    "get_handler",
]