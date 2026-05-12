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

        #  normalize output
        if isinstance(y, (list, tuple)):
            return float(y[0])

        if hasattr(y, "shape"):
            return float(y.flatten()[0])

        return float(y)