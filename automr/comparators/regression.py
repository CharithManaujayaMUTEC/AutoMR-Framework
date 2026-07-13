"""
Regression comparator implementation.

This comparator evaluates metamorphic testing results for
regression models using an epsilon tolerance.
"""

from automr.interfaces import BaseComparator


class RegressionComparator(BaseComparator):
    """
    Comparator for regression tasks.

    Predictions are considered equivalent when their absolute
    difference is within the configured epsilon threshold.
    """

    def __init__(self, epsilon=0.05):
        """
        Initialize the regression comparator.

        Parameters
        ----------
        epsilon : float
            Maximum allowable prediction difference.
        """
        self.epsilon = epsilon

    def compare(self, y1, y2):
        """
        Compare two regression predictions.

        Parameters
        ----------
        y1 : float
            Original prediction.
        y2 : float
            Transformed prediction.

        Returns
        -------
        tuple
            (absolute_difference, passed)
        """

        # Compute the absolute prediction difference.
        diff = abs(float(y1) - float(y2))

        # Determine whether the difference is acceptable.
        passed = diff <= self.epsilon

        return diff, passed