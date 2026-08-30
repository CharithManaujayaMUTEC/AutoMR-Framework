import numpy as np


# ==========================================================
# Dark Visibility Relation
# ==========================================================

class DarkVisibilityRelation:
    """
    Prediction should remain approximately invariant under
    visibility degradation or localized darkness.
    """

    def __init__(self, epsilon=0.10):
        self.epsilon = epsilon

    def type(self):
        return "equality"

    def expected(self):
        return (
            "Output should remain approximately invariant under "
            "visibility degradation."
        )

    def check(self, y1, y2):
        return abs(y1 - y2) <= self.epsilon


# ==========================================================
# Monotonic Increase
# ==========================================================

class MonotonicIncreaseRelation:
    """
    Output should remain within the configured epsilon
    tolerance after the transformation.
    """

    def __init__(self, epsilon=0.05):
        self.epsilon = epsilon

    def type(self):
        return "equality"

    def expected(self):
        return (
            "Output should remain approximately invariant "
            "after transformation."
        )

    def check(self, y1, y2):
        return abs(y1 - y2) <= self.epsilon


# ==========================================================
# Monotonic Decrease
# ==========================================================

class MonotonicDecreaseRelation:
    """
    Output should remain within the configured epsilon
    tolerance after the transformation.
    """

    def __init__(self, epsilon=0.05):
        self.epsilon = epsilon

    def type(self):
        return "equality"

    def expected(self):
        return (
            "Output should remain approximately invariant "
            "after transformation."
        )

    def check(self, y1, y2):
        return abs(y1 - y2) <= self.epsilon


# ==========================================================
# Generic Equality Relation
# ==========================================================

class InequalityRelation:
    """
    Generic epsilon-based relation.
    """

    def __init__(self, epsilon=0.20):
        self.epsilon = epsilon

    def type(self):
        return "equality"

    def expected(self):
        return (
            "Output should remain within the configured "
            "epsilon tolerance."
        )

    def check(self, y1, y2):
        return abs(y1 - y2) <= self.epsilon