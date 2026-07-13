"""
Verification package.

This package provides utilities for recording transformation
examples generated during metamorphic testing, including saved
images, metadata, and summary statistics.
"""

from .image_saver import TransformationSaver

# Public package interface.
__all__ = [
    "TransformationSaver",
]