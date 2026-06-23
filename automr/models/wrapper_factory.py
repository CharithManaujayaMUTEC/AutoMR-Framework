from .tensorflow_wrapper import TensorFlowWrapper
from .pytorch_wrapper import PyTorchWrapper
from .sklearn_wrapper import SklearnWrapper
from .custom_wrapper import CustomWrapper
from .onnx_wrapper import ONNXWrapper
from .remote_wrapper import RemoteWrapper

def get_wrapper(model):

    if isinstance(model, str):

        if model.lower().endswith(
            ".onnx"
        ):
            return ONNXWrapper(model)
        
    if isinstance(model, str):

        if model.startswith(
            "http://"
        ) or model.startswith(
            "https://"
        ):
            return RemoteWrapper(model)

    module = (
        model.__class__.__module__
        .lower()
    )

    if "tensorflow" in module or "keras" in module:
        return TensorFlowWrapper(model)

    if "torch" in module:
        return PyTorchWrapper(model)

    if "sklearn" in module:
        return SklearnWrapper(model)

    if hasattr(model, "predict"):
        return CustomWrapper(model)

    raise ValueError(
        f"Unsupported model type: {type(model)}"
    )