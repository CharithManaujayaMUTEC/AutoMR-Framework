import numpy as np

from automr.interfaces import BaseModel


class TensorFlowWrapper(BaseModel):

    def __init__(self, model):
        self.model = model

    def predict(self, x):
        x = np.asarray(x, dtype=np.float32)

        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)

        pred = self.model.predict(
            x,
            verbose=0
        )

        return float(pred.flatten()[0])

    def predict_batch(self, xs):
        batch = np.asarray(
            xs,
            dtype=np.float32
        )

        preds = self.model.predict(
            batch,
            verbose=0
        )

        return preds.flatten().tolist()