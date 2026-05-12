# automr/relations/temporal_relations.py

class TemporalSmoothnessRelation:
    def __init__(self, delta=0.1):
        self.delta = delta

    def type(self):
        return "temporal"   #  REQUIRED

    def check(self, y1, y2):
        return abs(y1 - y2) < self.delta

    def expected(self):
        return "Predictions should change smoothly across consecutive frames"


class TemporalConsistencyRelation:
    def __init__(self, delta=0.2):
        self.delta = delta

    def type(self):
        return "temporal"   #  REQUIRED

    def check(self, y1, y2):
        return abs(y1 - y2) < self.delta

    def expected(self):
        return "Nearby frames should produce consistent outputs"