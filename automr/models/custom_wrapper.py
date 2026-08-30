"""
Custom model wrapper.

This wrapper adapts user-defined models to the AutoMR model
interface by providing standardized single and batch prediction
methods.
"""

from automr.interfaces import BaseModel


class CustomWrapper(BaseModel):
    """
    Generic wrapper for custom models.
    """

    def __init__(self, model):
        """
        Initialize the wrapper.

        Parameters
        ----------
        model : object
            User-provided model.
        """
        self.model = model

    def predict(self, x):
        """
        Generate a prediction for a single input.
        """

        # Perform model inference.
        y = self.model.predict(x)

        # Handle list and tuple outputs.
        if isinstance(y, (list, tuple)):
            return float(y[0])

        # Handle NumPy arrays and tensor-like outputs.
        if hasattr(y, "shape"):
            return float(y.reshape(-1)[0])

        # Handle scalar outputs.
        return float(y)

    def predict_batch(self, xs):
        """
        Generate predictions for multiple inputs.
        """

        # Use native batch prediction if available.
        if hasattr(self.model, "predict_batch"):
            return self.model.predict_batch(xs)

        # Attempt framework-supported batch prediction.
        if hasattr(self.model, "predict"):
            try:
                y = self.model.predict(xs)

                if hasattr(y, "shape"):
                    return y.reshape(len(xs), -1)[:, 0].astype(float).tolist()

            except Exception:
                # Fall back to sequential prediction.
                pass

        # Sequential prediction fallback.
        return [self.predict(x) for x in xs]