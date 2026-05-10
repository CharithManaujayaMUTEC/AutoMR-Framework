class InequalityRelation:
    def __init__(self, delta=0.2):
        self.delta = delta

    def check(self, original, transformed):
        return transformed <= original + self.delta
