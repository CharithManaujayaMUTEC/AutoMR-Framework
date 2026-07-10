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

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name

    # ==================================================
    # Single Prediction
    # ==================================================
    def predict(self, x):

        x = np.ascontiguousarray(
            np.asarray(x, dtype=np.float32)
        )

        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)

        pred = self.session.run(
            None,
            {self.input_name: x},
        )[0]

        pred = np.asarray(
            pred,
            dtype=np.float32,
        )

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

        if len(xs) == 0:
            return []

        batch = np.ascontiguousarray(
            np.asarray(xs, dtype=np.float32)
        )

        if batch.ndim == 3:
            batch = np.expand_dims(batch, axis=0)

        preds = self.session.run(
            None,
            {self.input_name: batch},
        )[0]

        preds = np.asarray(
            preds,
            dtype=np.float32,
        )

        if preds.ndim == 1:
            return preds.tolist()

        preds = preds.reshape(
            preds.shape[0],
            -1,
        )

        if preds.shape[1] == 1:
            return preds[:, 0].tolist()

        return preds.tolist()