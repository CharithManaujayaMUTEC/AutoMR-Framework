"""
Model wrappers package.

This package provides wrapper implementations that expose a
consistent prediction interface for multiple machine learning
frameworks. Wrappers allow AutoMR to interact with different
models through a common API.
"""

from .wrapper_factory import get_wrapper
from .tensorflow_wrapper import TensorFlowWrapper
from .pytorch_wrapper import PyTorchWrapper
from .sklearn_wrapper import SklearnWrapper
from .onnx_wrapper import ONNXWrapper
from .remote_wrapper import RemoteWrapper
from .custom_wrapper import CustomWrapper
from .xgboost_wrapper import XGBoostWrapper

# Public package interface.
__all__ = [
    "get_wrapper",
    "TensorFlowWrapper",
    "PyTorchWrapper",
    "SklearnWrapper",
    "ONNXWrapper",
    "RemoteWrapper",
    "CustomWrapper",
    "XGBoostWrapper",
]