"""
Model wrapper for AutoMR.

This wrapper provides a unified prediction interface for different
machine learning and deep learning models, allowing the framework
to interact with models consistently regardless of their underlying
implementation.
"""


class ModelWrapper:
    """
    Wraps a model and exposes standardized single and batch
    prediction methods.
    """

    def __init__(self, model, predict_fn=None):
        """
        Initialize the model wrapper.

        Parameters
        ----------
        model : object
            Target model instance.
        predict_fn : callable, optional
            Optional custom prediction function.
        """
        self.model = model
        self.predict_fn = predict_fn

    def predict(self, x):
        """
        Perform prediction for a single input.

        Parameters
        ----------
        x : Any
            Input sample.

        Returns
        -------
        float
            Model prediction converted to a floating-point value.
        """

        # Handle invalid input.
        if x is None:
            return 0.0

        # Use the custom prediction function if provided.
        if self.predict_fn:
            y = self.predict_fn(self.model, x)
        else:
            # Otherwise, use the model's native prediction method.
            y = self.model.predict(x)

        # Handle list or tuple outputs.
        if isinstance(y, (list, tuple)):
            return float(y[0])

        # Handle NumPy arrays and tensor-like outputs.
        if hasattr(y, "shape"):
            return float(y.flatten()[0])

        # Handle scalar outputs.
        return float(y)

    def predict_batch(self, xs):
        """
        Perform batch prediction.

        Priority:
        1. Native predict_batch()
        2. Custom predict_fn_batch()
        3. Fallback to sequential prediction

        Parameters
        ----------
        xs : iterable
            Collection of input samples.

        Returns
        -------
        list
            Predictions for all input samples.
        """

        # Use the model's native batch inference if available.
        if hasattr(self.model, "predict_batch"):
            return self.model.predict_batch(xs)

        # Use an optional custom batch prediction callback.
        if (
            self.predict_fn is not None and
            hasattr(self.predict_fn, "predict_batch")
        ):
            return self.predict_fn.predict_batch(
                self.model,
                xs,
            )

        # Fallback to sequential prediction.
        return [
            self.predict(x)
            for x in xs
        ]