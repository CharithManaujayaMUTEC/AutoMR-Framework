"""
Classification comparator implementation.

This comparator evaluates metamorphic testing results for
classification models by checking whether the predicted class
labels remain identical after applying a transformation.
"""

from automr.interfaces import BaseComparator


class ClassificationComparator(BaseComparator):
    """
    Comparator for classification tasks.

    A metamorphic relation passes only when the transformed
    prediction matches the original prediction exactly.
    """

    def compare(self, y1, y2):
        """
        Compare two classification predictions.

        Parameters
        ----------
        y1 : Any
            Original model prediction.
        y2 : Any
            Prediction after transformation.

        Returns
        -------
        tuple
            (difference, passed)
        """

        # Classification predictions must be identical.
        passed = y1 == y2

        # Difference is binary for classification.
        diff = 0 if passed else 1

        return diff, passed