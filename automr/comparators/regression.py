from automr.interfaces import BaseComparator


class RegressionComparator(BaseComparator):

    def __init__(self, epsilon=0.05):
        self.epsilon = epsilon

    def compare(self, y1, y2):

        diff = abs(float(y1) - float(y2))
        passed = diff <= self.epsilon

        return diff, passed