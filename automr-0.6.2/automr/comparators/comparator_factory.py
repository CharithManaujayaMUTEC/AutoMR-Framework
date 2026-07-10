from .regression import RegressionComparator
from .classification import ClassificationComparator


def get_comparator(task="regression", epsilon=0.05):

    if task == "regression":
        return RegressionComparator(epsilon)

    if task == "classification":
        return ClassificationComparator()

    raise ValueError(f"Unsupported task type: {task}")