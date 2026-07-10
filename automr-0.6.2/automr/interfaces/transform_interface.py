from abc import ABC, abstractmethod


class BaseTransform(ABC):
    """
    Base interface for all input transformations.
    """

    @abstractmethod
    def apply(self, x, param):
        """
        Apply transformation to input.

        Parameters
        ----------
        x : Any
            Input data.

        param : Any
            Transformation parameter.

        Returns
        -------
        Any
            Transformed input.
        """
        pass

    @property
    def name(self):
        return self.__class__.__name__