"""
Logging package.

This package provides logging utilities for the AutoMR framework,
including a lightweight logger for recording execution events.
"""

from .logger import AutoMRLogger

# Public package interface.
__all__ = [
    "AutoMRLogger",
]