
import numpy as np

class CropRelation:
    def check(self, y1, y2):
        return np.sign(y1) == np.sign(y2)
