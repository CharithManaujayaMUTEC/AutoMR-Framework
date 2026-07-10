import numpy as np

from automr.interfaces import BaseModel


class XGBoostWrapper(BaseModel):

    def __init__(self, model):
        self.model = model

    def predict(self, x):
        x = np.asarray(x, dtype=np.float32)

        if x.ndim > 1:
            x = x.reshape(1, -1)

        return float(self.model.predict(x)[0])

    def predict_batch(self, xs):
        """
        Native batch prediction.
        """

        xs = np.asarray(xs, dtype=np.float32)

        if xs.ndim > 2:
            xs = xs.reshape(xs.shape[0], -1)

        return self.model.predict(xs).astype(float).tolist()