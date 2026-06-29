import numpy as np
import torch

from automr.interfaces import BaseModel


class PyTorchWrapper(BaseModel):

    def __init__(self, model, device="cpu"):
        self.model = model.eval()
        self.device = device

    def predict(self, x):
        x = np.asarray(x, dtype=np.float32)

        if x.ndim == 3:
            x = np.transpose(x, (2, 0, 1))
            x = np.expand_dims(x, axis=0)

        x = torch.tensor(
            x,
            dtype=torch.float32,
            device=self.device
        )

        with torch.no_grad():
            pred = self.model(x)

        return float(pred.cpu().numpy().flatten()[0])

    def predict_batch(self, xs):
        batch = np.asarray(
            xs,
            dtype=np.float32
        )

        batch = np.transpose(batch, (0, 3, 1, 2))

        batch = torch.tensor(
            batch,
            dtype=torch.float32,
            device=self.device
        )

        with torch.no_grad():
            preds = self.model(batch)

        return preds.cpu().numpy().flatten().tolist()