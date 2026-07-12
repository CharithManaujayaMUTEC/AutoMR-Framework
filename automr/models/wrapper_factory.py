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

    # -------------------------
    # String models
    # -------------------------
    if isinstance(model, str):

        if model.lower().endswith(".onnx"):
            return ONNXWrapper(model)

        if model.startswith(("http://", "https://")):
            return RemoteWrapper(model)

        raise ValueError(f"Unknown model path: {model}")

    # -------------------------
    # Framework detection
    # -------------------------
    module = model.__class__.__module__.lower()

    # TensorFlow / Keras
    if "tensorflow" in module or "keras" in module:
        return TensorFlowWrapper(model)

    # PyTorch
    if isinstance(model, nn.Module):
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

    # scikit-learn
    if "sklearn" in module:
        return SklearnWrapper(model)

    # Generic custom model
    if hasattr(model, "predict"):
        return CustomWrapper(model)

    raise ValueError(f"Unsupported model type: {type(model)}")