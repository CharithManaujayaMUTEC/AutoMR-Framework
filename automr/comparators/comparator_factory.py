"""
Comparator factory.

Provides a unified interface for constructing the appropriate
comparator based on the selected machine learning task.
"""

from .regression import RegressionComparator
from .classification import ClassificationComparator


def get_comparator(task="regression", epsilon=0.05):
    """
    Create a comparator for the specified task.

    Parameters
    ----------
    task : str
        Supported task type ("regression" or "classification").
    epsilon : float
        Regression tolerance threshold.

    Returns
    -------
    BaseComparator
        Configured comparator instance.

    Raises
    ------
    ValueError
        If the requested task is unsupported.
    """

    if task == "regression":
        return RegressionComparator(epsilon)

    if task == "classification":
        return ClassificationComparator()

    raise ValueError(f"Unsupported task type: {task}")