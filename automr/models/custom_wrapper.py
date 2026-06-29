from automr.interfaces import BaseModel


class CustomWrapper(BaseModel):

    def __init__(self, model):
        self.model = model

    def predict(self, x):
        return float(self.model.predict(x))

    def predict_batch(self, xs):
        return [
            float(self.model.predict(x))
            for x in xs
        ]