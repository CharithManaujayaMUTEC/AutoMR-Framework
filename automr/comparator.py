"""
Comparator definitions.

This module provides base comparator classes used to compare
model outputs during metamorphic testing.
"""


class Comparator:
    """
    Base comparator interface.
    """

    def compare(self, y1, y2):
        """
        Compare two prediction values.

        Parameters
        ----------
        y1 : float
            Original prediction.
        y2 : float
            Transformed prediction.

        Returns
        -------
        tuple
            Difference and pass/fail status.
        """
        raise NotImplementedError


class RegressionComparator(Comparator):
    """
    Comparator for regression model outputs.
    """

    def __init__(self, epsilon=0.05, relative=True):
        """
        Initialize the regression comparator.

        Parameters
        ----------
        epsilon : float, default=0.05
            Allowed prediction tolerance.
        relative : bool, default=True
            Use relative error instead of absolute error.
        """
        self.epsilon = epsilon
        self.relative = relative

    def compare(self, y1, y2):
        """
        Compare two regression predictions.

        Returns
        -------
        tuple
            (difference, passed)
        """

        # Compute the absolute prediction difference.
        diff = abs(y1 - y2)

        # Evaluate either relative or absolute tolerance.
        if self.relative:
            scale = abs(y1) + 1e-6
            passed = (diff / scale) < self.epsilon
        else:
            passed = diff < self.epsilon

        return diff, passed