"""
High Performance Computing (HPC) module for AutoMR.

This package provides optimized execution backends for
large-scale metamorphic testing.

Modules
-------
api.py         : HighPerformanceAutoMR
executor.py    : Multiprocessing execution engine
scheduler.py   : Work scheduling
batcher.py     : Batched model inference
cache.py       : Prediction cache
utils.py       : Utility functions
"""

from .api import HighPerformanceAutoMR
from .executor import HPCExecutor
from .scheduler import TaskScheduler
from .batcher import BatchManager
from .cache import PredictionCache

__all__ = [
    "HighPerformanceAutoMR",
    "HPCExecutor",
    "TaskScheduler",
    "BatchManager",
    "PredictionCache",
]