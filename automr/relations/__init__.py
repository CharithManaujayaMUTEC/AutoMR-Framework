"""
Metamorphic relations package.

This package contains the built-in metamorphic relations used by
AutoMR to verify model behavior under different input
transformations. Relations define the expected relationship
between the original and transformed model outputs.
"""

from .image_relations import *
from .weather_relations import *
from .behavioral_relations import *
from .temporal_relations import *
from .inequality_relation import *

# Re-export all built-in relations.