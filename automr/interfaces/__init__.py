"""
Core interfaces for the AutoMR framework.

This package defines the abstract base classes that establish the
common contracts for models, transformations, metamorphic relations,
and comparators. Framework components should implement these
interfaces to ensure compatibility with the AutoMR execution engine.
"""

from .model_interface import BaseModel
from .transform_interface import BaseTransform
from .relation_interface import BaseRelation
from .comparator_interface import BaseComparator

# Public package interface.
__all__ = [
    "BaseModel",
    "BaseTransform",
    "BaseRelation",
    "BaseComparator",
]