from automr.interfaces import BaseModel


class TensorFlowWrapper(BaseModel):

    def __init__(self, model):
        self.model = model

    def predict(self, x):
        pred = self.model.predict(x, verbose=0)
        return pred