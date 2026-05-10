class RotationRelation:
    def __init__(self, epsilon=0.15):
        self.epsilon = epsilon

    def check(self, original, transformed):
        return abs(original - transformed) < self.epsilon
