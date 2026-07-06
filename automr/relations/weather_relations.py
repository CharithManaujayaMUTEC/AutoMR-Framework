import numpy as np


# ==========================================================
# Base Relation
# ==========================================================

class InvarianceRelation:
    """
    Generic metamorphic relation for transformations that
    should approximately preserve the model prediction.
    """

    def __init__(self, epsilon=0.05, expected=None):
        self.epsilon = epsilon
        self._expected = (
            expected
            if expected is not None
            else "Prediction should remain approximately invariant."
        )

    def type(self):
        return "equality"

    def expected(self):
        return self._expected

    def check(self, y1, y2):
        change = abs(y1 - y2) / (abs(y1) + 1e-6)
        return change < self.epsilon


# ==========================================================
# Rain
# ==========================================================

class RainRelation(InvarianceRelation):

    def __init__(self, epsilon=0.05):
        super().__init__(
            epsilon=epsilon,
            expected="Rain should not significantly affect prediction."
        )


# ==========================================================
# Snow
# ==========================================================

class SnowRelation(InvarianceRelation):

    def __init__(self, epsilon=0.05):
        super().__init__(
            epsilon=epsilon,
            expected="Snow should not significantly affect prediction."
        )


# ==========================================================
# Fog
# ==========================================================

class FogRelation(InvarianceRelation):

    def __init__(self, epsilon=0.05):
        super().__init__(
            epsilon=epsilon,
            expected="Fog should not significantly affect prediction."
        )


# ==========================================================
# Sandstorm
# ==========================================================

class SandstormRelation(InvarianceRelation):

    def __init__(self, epsilon=0.06):
        super().__init__(
            epsilon=epsilon,
            expected="Sandstorm should not significantly affect prediction."
        )


# ==========================================================
# Dust
# ==========================================================

class DustRelation(InvarianceRelation):

    def __init__(self, epsilon=0.05):
        super().__init__(
            epsilon=epsilon,
            expected="Dust should not significantly affect prediction."
        )


# ==========================================================
# Haze
# ==========================================================

class HazeRelation(InvarianceRelation):

    def __init__(self, epsilon=0.04):
        super().__init__(
            epsilon=epsilon,
            expected="Haze should not significantly affect prediction."
        )


# ==========================================================
# Smoke
# ==========================================================

class SmokeRelation(InvarianceRelation):

    def __init__(self, epsilon=0.05):
        super().__init__(
            epsilon=epsilon,
            expected="Smoke should not significantly affect prediction."
        )