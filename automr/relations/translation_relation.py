
class TranslationRelation:
    def __init__(self, tolerance=0.15):
        self.tolerance = tolerance

    def check(self, y1, y2):
        change = abs(y1 - y2) / (abs(y1) + 1e-6)
return change < self.tolerance
