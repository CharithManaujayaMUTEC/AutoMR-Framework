class Comparator:
    def compare(self, y1, y2):
        raise NotImplementedError


class RegressionComparator(Comparator):

    def __init__(self, epsilon=0.05, relative=True):
        self.epsilon = epsilon
        self.relative = relative

    def compare(self, y1, y2):

        diff = abs(y1 - y2)

        if self.relative:
            scale = abs(y1) + 1e-6
            passed = (diff / scale) < self.epsilon
        else:
            passed = diff < self.epsilon

        return diff, passed