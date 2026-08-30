import numpy as np


# ==========================================================
# Base Temporal Relation
# ==========================================================

class TemporalRelation:
    """
    Base class for temporal metamorphic relations.
    """

    def __init__(self, epsilon=0.05, expected=None):
        self.epsilon = epsilon
        self._expected = (
            expected
            if expected is not None
            else "Temporal consistency should be preserved."
        )

    def type(self):
        return "temporal"

    def expected(self):
        return self._expected

    def check(self, y1, y2):
        return abs(y1 - y2) <= self.epsilon


# ==========================================================
# Temporal Smoothness
# ==========================================================

class TemporalSmoothnessRelation(TemporalRelation):
    """
    Consecutive frames should produce smoothly
    varying predictions.
    """

    def __init__(self, epsilon=0.03):
        super().__init__(
            epsilon=epsilon,
            expected=(
                "Predictions should change smoothly across "
                "consecutive frames."
            ),
        )


# ==========================================================
# Temporal Consistency
# ==========================================================

class TemporalConsistencyRelation(TemporalRelation):
    """
    Nearby frames should produce consistent
    predictions.
    """

    def __init__(self, epsilon=0.05):
        super().__init__(
            epsilon=epsilon,
            expected=(
                "Nearby frames should produce "
                "consistent outputs."
            ),
        )