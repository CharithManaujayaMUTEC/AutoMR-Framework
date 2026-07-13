"""
XGBoost model wrapper.

This wrapper adapts XGBoost models to the AutoMR model interface by
providing standardized methods for single-sample and batch prediction.
"""

import numpy as np

from automr.interfaces import BaseModel


class XGBoostWrapper(BaseModel):
    """
    Wrapper for XGBoost models.
    """

    def __init__(self, model):
        """
        Initialize the XGBoost wrapper.

        Parameters
        ----------
        model : xgboost model
            Trained XGBoost model.
        """
        self.model = model

    def predict(self, x):
        """
        Generate a prediction for a single input sample.
        """

        # Convert the input to a float32 NumPy array.
        x = np.asarray(x, dtype=np.float32)

        # Flatten multi-dimensional inputs into a single sample.
        if x.ndim > 1:
            x = x.reshape(1, -1)

        # Perform inference and return the prediction.
        return float(self.model.predict(x)[0])

    def predict_batch(self, xs):
        """
        Generate predictions for multiple input samples.

        Returns
        -------
        list[float]
            Predictions for the input batch.
        """

        # Convert the batch to a float32 NumPy array.
        xs = np.asarray(xs, dtype=np.float32)

        # Flatten higher-dimensional inputs when required.
        if xs.ndim > 2:
            xs = xs.reshape(xs.shape[0], -1)

        # Perform native batch inference.
        return self.model.predict(xs).astype(float).tolist()