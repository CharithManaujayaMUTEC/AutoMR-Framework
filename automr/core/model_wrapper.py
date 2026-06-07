class ModelWrapper:

    def __init__(self, model, predict_fn=None):
        self.model = model
        self.predict_fn = predict_fn

    def predict(self, x):
        ...
    
    def predict_batch(self, xs):

        if hasattr(self.model, "predict_batch"):
            return self.model.predict_batch(xs)

        return [self.predict(x) for x in xs]