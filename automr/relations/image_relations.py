import numpy as np


#  Brightness MR
class BrightnessRelation:
    def __init__(self, tolerance=0.05):
        self.tolerance = tolerance

    def type(self):
        return "equality"

    def expected(self):
        return "Output should remain approximately same under brightness change"

    def check(self, y1, y2):
        change = abs(y1 - y2) / (abs(y1) + 1e-6)
return change < self.tolerance


#  Rotation MR
class RotationRelation:
    def __init__(self, epsilon=0.15):
        self.epsilon = epsilon

    def type(self):
        return "equality"

    def expected(self):
        return "Small rotation should not significantly change output"

    def check(self, y1, y2):
        return abs(y1 - y2) / (abs(y1) + 1e-6) < self.epsilon


#  Translation MR
class TranslationRelation:
    def __init__(self, tolerance=0.15):
        self.tolerance = tolerance

    def type(self):
        return "equality"

    def expected(self):
        return "Small translation should preserve prediction consistency"

    def check(self, y1, y2):
        change = abs(y1 - y2) / (abs(y1) + 1e-6)
return change < self.tolerance


#  Noise MR
class NoiseRelation:
    def __init__(self, tolerance=0.1):
        self.tolerance = tolerance

    def type(self):
        return "equality"

    def expected(self):
        return "Noise should not significantly affect prediction"

    def check(self, y1, y2):
        change = abs(y1 - y2) / (abs(y1) + 1e-6)
return change < self.tolerance


#  Mirror / Flip MR (IMPORTANT — inequality)
class FlipRelation:
    def type(self):
        return "inequality"

    def expected(self):
        return "Flipped image should invert steering"

    def check(self, y1, y2):
        return abs(y2 + y1) < 0.1


#  Blur MR
class BlurRelation:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def type(self):
        return "equality"

    def expected(self):
        return "Blur should not significantly change output"

    def check(self, y1, y2):
        return abs(y1 - y2) / (abs(y1) + 1e-6) < self.epsilon


#  Contrast MR
class ContrastRelation:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def type(self):
        return "equality"

    def expected(self):
        return "Contrast change should not significantly affect output"

    def check(self, y1, y2):
        return abs(y1 - y2) / (abs(y1) + 1e-6) < self.epsilon


#  Weather MR
class WeatherRelation:
    def __init__(self, epsilon=0.15):
        self.epsilon = epsilon

    def type(self):
        return "equality"

    def expected(self):
        return "Weather changes should not significantly affect output"

    def check(self, y1, y2):
        return abs(y1 - y2) / (abs(y1) + 1e-6) < self.epsilon