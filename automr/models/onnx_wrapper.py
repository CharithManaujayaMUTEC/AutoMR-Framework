"""
ONNX Runtime model wrapper.

This wrapper enables AutoMR to execute inference using ONNX Runtime,
providing optimized single-sample and batch prediction support.
"""

import numpy as np
import onnxruntime as ort

from automr.interfaces import BaseModel


class ONNXWrapper(BaseModel):
    """
    Generic ONNX Runtime wrapper.

    Features
    --------
    - Single prediction
    - Batch prediction
    - Optimized inference
    - Model agnostic
    """

    def __init__(self, model_path):
        """
        Initialize the ONNX Runtime session.

        Parameters
        ----------
        model_path : str
            Path to the ONNX model file.
        """

        self.session = ort.InferenceSession(
            model_path,
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

        # Cache the model input name.
        self.input_name = self.session.get_inputs()[0].name

    # ==================================================
    # Single Prediction
    # ==================================================
    def predict(self, x):
        """
        Generate a prediction for a single input.
        """

        # Convert input to contiguous float32 format.
        x = np.ascontiguousarray(
            np.asarray(x, dtype=np.float32)
        )

        # Add a batch dimension if necessary.
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)

        # Execute ONNX Runtime inference.
        pred = self.session.run(
            None,
            {self.input_name: x},
        )[0]

        pred = np.asarray(
            pred,
            dtype=np.float32,
        )

        # Return the first prediction value.
        return float(pred.reshape(-1)[0])

    # ==================================================
    # Batch Prediction
    # ==================================================
    def predict_batch(self, xs):
        """
        Predict a batch of inputs.

        Optimizations
        -------------
        - Single contiguous NumPy allocation
        - One ONNX Runtime inference call
        - Supports arbitrary output shapes
        """

        # Handle empty batches.
        if len(xs) == 0:
            return []

        # Create a contiguous batch array.
        batch = np.ascontiguousarray(
            np.asarray(xs, dtype=np.float32)
        )

        # Add a batch dimension when required.
        if batch.ndim == 3:
            batch = np.expand_dims(batch, axis=0)

        # Execute batched inference.
        preds = self.session.run(
            None,
            {self.input_name: batch},
        )[0]

        preds = np.asarray(
            preds,
            dtype=np.float32,
        )

        # Handle one-dimensional outputs.
        if preds.ndim == 1:
            return preds.tolist()

        # Flatten higher-dimensional outputs.
        preds = preds.reshape(
            preds.shape[0],
            -1,
        )

        # Return scalar predictions when applicable.
        if preds.shape[1] == 1:
            return preds[:, 0].tolist()

        # Return full prediction vectors.
        return preds.tolist()