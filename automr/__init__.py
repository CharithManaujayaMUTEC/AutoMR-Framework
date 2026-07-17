"""
AutoMR framework package.

This package provides the main entry point for the AutoMR framework,
enabling users to perform metamorphic testing through the AutoMR API.
"""

from .api import AutoMR

# Public package interface.
__all__ = [
    "AutoMR",
]

__version__ = "1.1.3"