class Comparator:
    def compare(self, y1, y2):
        raise NotImplementedError


class RegressionComparator(Comparator):

    def __init__(self, epsilon=0.05):
        self.epsilon = epsilon

    def compare(self, y1, y2):
        diff = abs(y1 - y2)
        passed = diff < self.epsilon
        return diff, passed