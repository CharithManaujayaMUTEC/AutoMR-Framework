class TemporalSmoothnessRelation:
    def __init__(self, delta=0.03):
        self.delta = delta

    def type(self):
        return "temporal"

    def expected(self):
        return "Predictions should change smoothly across consecutive frames"

    def check(self, y1, y2):
        return abs(y1 - y2) / (abs(y1) + 1e-6) < self.delta


class TemporalConsistencyRelation:
    def __init__(self, delta=0.05):
        self.delta = delta

    def type(self):
        return "temporal"

    def expected(self):
        return "Nearby frames should produce consistent outputs"

    def check(self, y1, y2):
        return abs(y1 - y2) / (abs(y1) + 1e-6) < self.delta