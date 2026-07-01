import numpy as np


# ==========================================================
# Dark Visibility Relation
# ==========================================================

class DarkVisibilityRelation:
    """
    Prediction should not change drastically under
    visibility degradation or localized darkness.
    """

    def __init__(self, epsilon=0.10):
        self.epsilon = epsilon

    def type(self):
        return "inequality"

    def expected(self):
        return (
            "Output should not change drastically under "
            "visibility degradation."
        )

    def check(self, y1, y2):
        change = abs(y2 - y1) / (abs(y1) + 1e-6)
        return change <= self.epsilon


# ==========================================================
# Monotonic Increase
# ==========================================================

class MonotonicIncreaseRelation:
    """
    Output should monotonically increase after
    the transformation.
    """

    def type(self):
        return "inequality"

    def expected(self):
        return "Output should increase after transformation."

    def check(self, y1, y2):
        return y2 >= y1


# ==========================================================
# Monotonic Decrease
# ==========================================================

class MonotonicDecreaseRelation:
    """
    Output should monotonically decrease after
    the transformation.
    """

    def type(self):
        return "inequality"

    def expected(self):
        return "Output should decrease after transformation."

    def check(self, y1, y2):
        return y2 <= y1


# ==========================================================
# Generic Inequality Relation
# ==========================================================

class InequalityRelation:
    """
    Generic bounded inequality relation.
    """

    def __init__(self, epsilon=0.20):
        self.epsilon = epsilon

    def type(self):
        return "inequality"

    def expected(self):
        return (
            "Output should remain within the allowed "
            "inequality bound."
        )

    def check(self, original, transformed):
        return transformed <= original + self.epsilon