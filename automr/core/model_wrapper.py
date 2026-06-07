class ModelWrapper:

    def __init__(self, model, predict_fn=None):
        self.model = model
        self.predict_fn = predict_fn

    def predict(self, x):

        if x is None:
            return 0.0

        if self.predict_fn:
            y = self.predict_fn(self.model, x)
        else:
            y = self.model.predict(x)

        if isinstance(y, (list, tuple)):
            return float(y[0])

        if hasattr(y, "shape"):
            return float(y.flatten()[0])

        return float(y)

    def predict_batch(self, xs):

        if hasattr(self.model, "predict_batch"):
            return self.model.predict_batch(xs)

        return [self.predict(x) for x in xs]