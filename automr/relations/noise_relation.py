
class NoiseRelation:
    def __init__(self, tolerance=0.1):
        self.tolerance = tolerance

    def check(self, y1, y2):
        return abs(y1 - y2) < self.tolerance
