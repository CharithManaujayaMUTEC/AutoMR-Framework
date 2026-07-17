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
# Brightness
# ==========================================================

class BrightnessRelation(InvarianceRelation):

    def __init__(self, epsilon=0.02):
        super().__init__(
            epsilon=epsilon,
            expected="Output should remain approximately same under localized brightness change."
        )


# ==========================================================
# Contrast
# ==========================================================

class ContrastRelation(InvarianceRelation):

    def __init__(self, epsilon=0.03):
        super().__init__(
            epsilon=epsilon,
            expected="Contrast change should not significantly affect output."
        )


# ==========================================================
# Blur
# ==========================================================

class BlurRelation(InvarianceRelation):

    def __init__(self, epsilon=0.03):
        super().__init__(
            epsilon=epsilon,
            expected="Blur should not significantly change output."
        )


# ==========================================================
# Noise
# ==========================================================

class NoiseRelation(InvarianceRelation):

    def __init__(self, epsilon=0.03):
        super().__init__(
            epsilon=epsilon,
            expected="Noise should not significantly affect prediction."
        )


# ==========================================================
# Rotation
# ==========================================================

class RotationRelation(InvarianceRelation):

    def __init__(self, epsilon=0.05):
        super().__init__(
            epsilon=epsilon,
            expected="Small rotation should not significantly change output."
        )


# ==========================================================
# Translation
# ==========================================================

class TranslationRelation(InvarianceRelation):

    def __init__(self, epsilon=0.05):
        super().__init__(
            epsilon=epsilon,
            expected="Small translation should preserve prediction consistency."
        )


# ==========================================================
# Composite
# ==========================================================

class CompositeRelation(InvarianceRelation):

    def __init__(self, epsilon=0.05):
        super().__init__(
            epsilon=epsilon,
            expected="Prediction should remain approximately consistent under composite transformations."
        )


# ==========================================================
# Flip
# ==========================================================

class FlipRelation:

    def __init__(self, epsilon=0.10):
        self.epsilon = epsilon

    def type(self):
        return "inequality"

    def expected(self):
        return "Flipped image should invert steering."

    def check(self, y1, y2):
        return abs(y2 + y1) / (abs(y1) + 1e-6) < self.epsilon
    
# ==========================================================
# Global Brightness
# ==========================================================

class GlobalBrightnessRelation(InvarianceRelation):

    def __init__(self, epsilon=0.03):
        super().__init__(
            epsilon=epsilon,
            expected="Output should remain approximately invariant under global brightness change."
        )

# ==========================================================
# Global Contrast
# ==========================================================

class GlobalContrastRelation(InvarianceRelation):

    def __init__(self, epsilon=0.03):
        super().__init__(
            epsilon=epsilon,
            expected="Output should remain approximately invariant under global contrast adjustment."
        )


# ==========================================================
# Global Blur
# ==========================================================

class GlobalBlurRelation(InvarianceRelation):

    def __init__(self, epsilon=0.04):
        super().__init__(
            epsilon=epsilon,
            expected="Output should remain approximately invariant under global blur."
        )


# ==========================================================
# Global Noise
# ==========================================================

class GlobalNoiseRelation(InvarianceRelation):

    def __init__(self, epsilon=0.04):
        super().__init__(
            epsilon=epsilon,
            expected="Output should remain approximately invariant under global sensor noise."
        )


# ==========================================================
# Global Rotation
# ==========================================================

class GlobalRotationRelation(InvarianceRelation):

    def __init__(self, epsilon=0.05):
        super().__init__(
            epsilon=epsilon,
            expected="Output should remain approximately invariant under small global rotation."
        )


# ==========================================================
# Global Translation
# ==========================================================

class GlobalTranslationRelation(InvarianceRelation):

    def __init__(self, epsilon=0.05):
        super().__init__(
            epsilon=epsilon,
            expected="Output should remain approximately invariant under small global translation."
        )