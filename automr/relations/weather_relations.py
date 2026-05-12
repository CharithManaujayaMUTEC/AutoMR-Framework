class RainRelation:
    def __init__(self, epsilon=0.05):
        self.epsilon = epsilon

    def type(self):
        return "equality"

    def expected(self):
        return "Rain should not significantly affect prediction"

    def check(self, y1, y2):
        return abs(y1 - y2) / (abs(y1) + 1e-6) < self.epsilon


class SnowRelation:
    def __init__(self, epsilon=0.05):
        self.epsilon = epsilon

    def type(self):
        return "equality"

    def expected(self):
        return "Snow should not significantly affect prediction"

    def check(self, y1, y2):
        return abs(y1 - y2) / (abs(y1) + 1e-6) < self.epsilon


class FogRelation:
    def __init__(self, epsilon=0.05):
        self.epsilon = epsilon

    def type(self):
        return "equality"

    def expected(self):
        return "Fog should not significantly affect prediction"

    def check(self, y1, y2):
        return abs(y1 - y2) / (abs(y1) + 1e-6) < self.epsilon