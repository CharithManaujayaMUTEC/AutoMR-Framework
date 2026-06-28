import numpy as np

class EpsilonManager:

    def __init__(self, start=0.01, end=0.20, steps=20):
        self.start = start
        self.end = end
        self.steps = steps

    def values(self):
        return np.linspace(
            self.start,
            self.end,
            self.steps
        )