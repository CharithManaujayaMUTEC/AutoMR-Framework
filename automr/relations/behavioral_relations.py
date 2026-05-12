class LessSensitiveRelation:
    """
    Output should not change drastically
    """
    def __init__(self, max_change=0.2):
        self.max_change = max_change

    def check(self, y1, y2):
        return abs(y2 - y1) <= self.max_change

    def expected(self):
        return "Output should not change drastically under degradation"


class MonotonicIncreaseRelation:
    """
    Output should increase
    """
    def check(self, y1, y2):
        return y2 >= y1

    def expected(self):
        return "Output should increase after transformation"


class MonotonicDecreaseRelation:
    """
    Output should decrease
    """
    def check(self, y1, y2):
        return y2 <= y1

    def expected(self):
        return "Output should decrease after transformation"