import numpy as np

class BrightnessRelation:
    def __init__(self, tolerance=0.05):
        self.tolerance = tolerance

    def expected(self):
        return "Output should remain approximately same under brightness change"

    def check(self, y1, y2):
        return abs(y1 - y2) < self.tolerance


class RotationRelation:
    def __init__(self, epsilon=0.15):
        self.epsilon = epsilon

    def expected(self):
        return "Small rotation should not significantly change output"

    def check(self, y1, y2):
        return abs(y1 - y2) < self.epsilon


class TranslationRelation:
    def __init__(self, tolerance=0.15):
        self.tolerance = tolerance

    def expected(self):
        return "Small translation should preserve prediction consistency"

    def check(self, y1, y2):
        return abs(y1 - y2) < self.tolerance


class NoiseRelation:
    def __init__(self, tolerance=0.1):
        self.tolerance = tolerance

    def expected(self):
        return "Noise should not significantly affect prediction"

    def check(self, y1, y2):
        return abs(y1 - y2) < self.tolerance


class FlipRelation:
    def expected(self):
        return "Flipped image should invert steering"

    def check(self, y1, y2):
        return abs(y2 + y1) < 0.1

class BlurRelation:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def check(self, y1, y2):
        return abs(y1 - y2) < self.epsilon


class ContrastRelation:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def check(self, y1, y2):
        return abs(y1 - y2) < self.epsilon


class WeatherRelation:
    def __init__(self, epsilon=0.15):
        self.epsilon = epsilon

    def check(self, y1, y2):
        return abs(y1 - y2) < self.epsilon