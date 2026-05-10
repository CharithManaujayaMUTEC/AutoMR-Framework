
class ModelWrapper:
    def __init__(self, model, predict_fn=None):
        self.model = model
        self.predict_fn = predict_fn

    def predict(self, x):
        if self.predict_fn:
            return self.predict_fn(self.model, x)
        return self.model.predict(x)
