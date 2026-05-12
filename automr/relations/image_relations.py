import numpy as np

class BrightnessRelation:
    def __init__(self, tolerance=0.02):
        self.tolerance = tolerance

    def type(self):
        return "equality"

    def expected(self):
        return "Output should remain approximately same under brightness change"

    def check(self, y1, y2):
        change = abs(y1 - y2) / (abs(y1) + 1e-6)
        return change < self.tolerance


class RotationRelation:
    def __init__(self, epsilon=0.05):
        self.epsilon = epsilon

    def type(self):
        return "equality"

    def expected(self):
        return "Small rotation should not significantly change output"

    def check(self, y1, y2):
        return abs(y1 - y2) / (abs(y1) + 1e-6) < self.epsilon


class TranslationRelation:
    def __init__(self, tolerance=0.05):
        self.tolerance = tolerance

    def type(self):
        return "equality"

    def expected(self):
        return "Small translation should preserve prediction consistency"

    def check(self, y1, y2):
        change = abs(y1 - y2) / (abs(y1) + 1e-6)
        return change < self.tolerance


class NoiseRelation:
    def __init__(self, tolerance=0.03):
        self.tolerance = tolerance

    def type(self):
        return "equality"

    def expected(self):
        return "Noise should not significantly affect prediction"

    def check(self, y1, y2):
        change = abs(y1 - y2) / (abs(y1) + 1e-6)
        return change < self.tolerance


class BlurRelation:
    def __init__(self, epsilon=0.03):
        self.epsilon = epsilon

    def type(self):
        return "equality"

    def expected(self):
        return "Blur should not significantly change output"

    def check(self, y1, y2):
        return abs(y1 - y2) / (abs(y1) + 1e-6) < self.epsilon


class ContrastRelation:
    def __init__(self, epsilon=0.03):
        self.epsilon = epsilon

    def type(self):
        return "equality"

    def expected(self):
        return "Contrast change should not significantly affect output"

    def check(self, y1, y2):
        return abs(y1 - y2) / (abs(y1) + 1e-6) < self.epsilon


class FlipRelation:
    def type(self):
        return "inequality"

    def expected(self):
        return "Flipped image should invert steering"

    def check(self, y1, y2):
        return abs(y2 + y1) / (abs(y1) + 1e-6) < 0.1