"""
Registry package.

This package provides registries for managing metamorphic
transformations and relations. Registries enable dynamic lookup
and registration of framework components.
"""

from .transformation_registry import TransformationRegistry
from .relation_registry import RelationRegistry

# Public package interface.
__all__ = [
    "TransformationRegistry",
    "RelationRegistry",
]