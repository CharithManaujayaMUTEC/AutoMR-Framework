import numpy as np
from automr.interfaces import BaseModel


class TensorFlowWrapper(BaseModel):

    def __init__(self, model):
        self.model = model

    def predict(self, x):
        x = np.asarray(x, dtype=np.float32)

        # If a single image (64,64,3) is passed, convert it to (1,64,64,3)
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)

        pred = self.model.predict(x, verbose=0)
        return pred