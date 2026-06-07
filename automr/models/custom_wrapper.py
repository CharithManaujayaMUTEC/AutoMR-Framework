from automr.interfaces import BaseModel
class CustomWrapper(BaseModel):

    def __init__(self, model):
        self.model = model

    def predict(self, x):
        return self.model.predict(x)

    def predict_batch(self, xs):

        if hasattr(self.model, "predict_batch"):
            return self.model.predict_batch(xs)

        return [self.model.predict(x) for x in xs]