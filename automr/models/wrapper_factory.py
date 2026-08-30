"""
Model wrapper factory.

This module automatically detects the type of a machine learning
model and returns the appropriate AutoMR wrapper. It supports
TensorFlow, PyTorch, Scikit-learn, XGBoost, ONNX Runtime, remote
REST APIs, and custom models through a unified interface.
"""

import torch
import torch.nn as nn

from .tensorflow_wrapper import TensorFlowWrapper
from .pytorch_wrapper import PyTorchWrapper
from .sklearn_wrapper import SklearnWrapper
from .xgboost_wrapper import XGBoostWrapper
from .custom_wrapper import CustomWrapper
from .onnx_wrapper import ONNXWrapper
from .remote_wrapper import RemoteWrapper


def get_wrapper(model):
    """
    Return the appropriate wrapper for a given model.

    Parameters
    ----------
    model : object or str
        Machine learning model instance, ONNX model path,
        or remote inference endpoint.

    Returns
    -------
    BaseModel
        Wrapper implementing the AutoMR model interface.

    Raises
    ------
    ValueError
        If the model type is unsupported.
    """

    # -------------------------
    # String models
    # -------------------------

    # Handle model paths and remote endpoints.
    if isinstance(model, str):

        # ONNX model file.
        if model.lower().endswith(".onnx"):
            return ONNXWrapper(model)

        # Remote REST API endpoint.
        if model.startswith(("http://", "https://")):
            return RemoteWrapper(model)

        raise ValueError(f"Unknown model path: {model}")

    # -------------------------
    # Framework detection
    # -------------------------

    # Identify the originating framework.
    module = model.__class__.__module__.lower()

    # TensorFlow / Keras
    if "tensorflow" in module or "keras" in module:
        return TensorFlowWrapper(model)

    # PyTorch
    if isinstance(model, nn.Module):

        # Retrieve optional preprocessing and decoder hooks.
        preprocess = getattr(model, "_automr_preprocess", None)
        decoder = getattr(model, "_automr_decoder", None)

        return PyTorchWrapper(
            model=model,
            preprocess=preprocess,
            decoder=decoder,
        )

    # XGBoost
    if "xgboost" in module:
        return XGBoostWrapper(model)

    # Scikit-learn
    if "sklearn" in module:
        return SklearnWrapper(model)

    # Generic custom model
    if hasattr(model, "predict"):
        return CustomWrapper(model)

    # Unsupported model type.
    raise ValueError(f"Unsupported model type: {type(model)}")