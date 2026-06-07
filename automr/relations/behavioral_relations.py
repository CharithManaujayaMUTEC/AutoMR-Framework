class DarkVisibiltyRelation:
    def __init__(self, max_change=0.1):
        self.max_change = max_change

    def type(self):
        return "inequality"

    def expected(self):
        return "Output should not change drastically under degradation"

    def check(self, y1, y2):
        change = abs(y2 - y1) / (abs(y1) + 1e-6)
        return change <= self.max_change


class MonotonicIncreaseRelation:
    def type(self):
        return "inequality"

    def expected(self):
        return "Output should increase after transformation"

    def check(self, y1, y2):
        return y2 >= y1


class MonotonicDecreaseRelation:
    def type(self):
        return "inequality"

    def expected(self):
        return "Output should decrease after transformation"

    def check(self, y1, y2):
        return y2 <= y1