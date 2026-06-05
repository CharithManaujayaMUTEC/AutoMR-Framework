from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Base interface for all models supported by AutoMR.
    """

    @abstractmethod
    def predict(self, x):
        """
        Generate prediction for input x.

        Parameters
        ----------
        x : Any
            Input data.

        Returns
        -------
        Any
            Model prediction.
        """
        pass