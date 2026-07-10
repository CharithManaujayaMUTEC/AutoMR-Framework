from abc import ABC, abstractmethod


class BaseComparator(ABC):
    """
    Base interface for output comparison.
    """

    @abstractmethod
    def compare(self, y_true, y_pred):
        """
        Compare two outputs.

        Returns
        -------
        tuple
            (difference, passed)
        """
        pass