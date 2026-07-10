from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Base interface for all models supported by AutoMR.
    """

    @abstractmethod
    def predict(self, x):
        """
        Generate a prediction for a single input.

        Parameters
        ----------
        x : Any
            Input sample.

        Returns
        -------
        float
            Model prediction.
        """
        pass

    @abstractmethod
    def predict_batch(self, xs):
        """
        Generate predictions for multiple inputs.

        Parameters
        ----------
        xs : Sequence[Any]
            Batch of input samples.

        Returns
        -------
        list[float]
            Predictions for each sample.
        """
        pass