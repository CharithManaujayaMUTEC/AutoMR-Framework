import torch
from automr.interfaces import BaseModel


class PyTorchWrapper(BaseModel):

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict(self, x):

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).float()

        with torch.no_grad():
            pred = self.model(x)

        return pred