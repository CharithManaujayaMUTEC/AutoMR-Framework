"""
Scikit-learn model wrapper.

This wrapper adapts Scikit-learn models to the AutoMR model
interface by providing standardized single-sample and batch
prediction methods.
"""

import numpy as np

from automr.interfaces import BaseModel


class SklearnWrapper(BaseModel):
    """
    Wrapper for Scikit-learn models.
    """

    def __init__(self, model):
        """
        Initialize the Scikit-learn wrapper.

        Parameters
        ----------
        model : sklearn estimator
            Trained Scikit-learn model.
        """
        self.model = model

    def predict(self, x):
        """
        Generate a prediction for a single input sample.
        """

        # Convert the input to a NumPy array.
        x = np.asarray(x)

        # Flatten multi-dimensional inputs into a single sample.
        if x.ndim > 1:
            x = x.reshape(1, -1)

        # Perform inference and return the prediction.
        return float(self.model.predict(x)[0])

    def predict_batch(self, xs):
        """
        Generate predictions for multiple input samples.
        """

        # Convert the batch to a NumPy array.
        xs = np.asarray(xs)

        # Flatten higher-dimensional inputs for Scikit-learn models.
        if xs.ndim > 2:
            xs = xs.reshape(xs.shape[0], -1)

        # Perform batch inference.
        return self.model.predict(xs).astype(float).tolist()