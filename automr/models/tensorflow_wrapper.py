"""
TensorFlow model wrapper.

This wrapper adapts TensorFlow and Keras models to the AutoMR model
interface by providing standardized single-sample and batch prediction
methods with optimized NumPy preprocessing.
"""

import numpy as np

from automr.interfaces import BaseModel


class TensorFlowWrapper(BaseModel):
    """
    Generic TensorFlow / Keras wrapper.

    Features
    --------
    - Single prediction
    - Batch prediction
    - Optimized inference
    - Model agnostic
    """

    def __init__(self, model):
        """
        Initialize the TensorFlow wrapper.

        Parameters
        ----------
        model : tensorflow.keras.Model
            Trained TensorFlow or Keras model.
        """
        self.model = model

    # ==================================================
    # Single Prediction
    # ==================================================
    def predict(self, x):
        """
        Generate a prediction for a single input sample.
        """

        # -----------------------------------------
        # Convert to contiguous float32 array
        # -----------------------------------------
        x = np.ascontiguousarray(
            np.asarray(x, dtype=np.float32)
        )

        # Add batch dimension if needed.
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)

        # -----------------------------------------
        # Inference
        # -----------------------------------------
        pred = self.model.predict(
            x,
            verbose=0,
        )

        # Convert predictions to a NumPy array.
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
        - Single NumPy conversion
        - Contiguous memory
        - Single model.predict() call
        - Works with arbitrary output shapes
        """

        # Return immediately for empty batches.
        if len(xs) == 0:
            return []

        # -----------------------------------------
        # Create contiguous float32 batch
        # -----------------------------------------
        batch = np.ascontiguousarray(
            np.asarray(xs, dtype=np.float32)
        )

        # Ensure batch dimension exists.
        if batch.ndim == 3:
            batch = np.expand_dims(batch, axis=0)

        # -----------------------------------------
        # Batched inference
        # -----------------------------------------
        preds = self.model.predict(
            batch,
            verbose=0,
        )

        # Convert predictions to a NumPy array.
        preds = np.asarray(
            preds,
            dtype=np.float32,
        )

        # -----------------------------------------
        # Preserve one prediction per sample
        # -----------------------------------------

        # One-dimensional output.
        if preds.ndim == 1:
            return preds.tolist()

        # Flatten higher-dimensional outputs.
        preds = preds.reshape(
            preds.shape[0],
            -1,
        )

        # Single output per sample.
        if preds.shape[1] == 1:
            return preds[:, 0].tolist()

        # Multiple outputs per sample.
        return preds.tolist()