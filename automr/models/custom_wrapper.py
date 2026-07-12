from automr.interfaces import BaseModel


class CustomWrapper(BaseModel):

    def __init__(self, model):
        self.model = model

    def predict(self, x):
        y = self.model.predict(x)

        if isinstance(y, (list, tuple)):
            return float(y[0])

        if hasattr(y, "shape"):
            return float(y.reshape(-1)[0])

        return float(y)

    def predict_batch(self, xs):

        if hasattr(self.model, "predict_batch"):
            return self.model.predict_batch(xs)

        if hasattr(self.model, "predict"):
            try:
                y = self.model.predict(xs)

                if hasattr(y, "shape"):
                    return y.reshape(len(xs), -1)[:, 0].astype(float).tolist()

            except Exception:
                pass

        return [self.predict(x) for x in xs]