class RainRelation:
    def __init__(self, epsilon=0.08):
        self.epsilon = epsilon

    def check(self, y1, y2):
        return abs(y1 - y2) / (abs(y1) + 1e-6) < self.epsilon

    def expected(self):
        return "Rain should not significantly affect prediction"


class SnowRelation:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def check(self, y1, y2):
        return abs(y1 - y2) / (abs(y1) + 1e-6) < self.epsilon

    def expected(self):
        return "Snow should not significantly affect prediction"


class FogRelation:
    def __init__(self, epsilon=0.08):
        self.epsilon = epsilon

    def check(self, y1, y2):
        return abs(y1 - y2) / (abs(y1) + 1e-6) < self.epsilon

    def expected(self):
        return "Fog should not significantly affect prediction"