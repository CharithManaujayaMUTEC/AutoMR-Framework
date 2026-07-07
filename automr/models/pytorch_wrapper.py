import numpy as np
import torch

from automr.interfaces import BaseModel


class PyTorchWrapper(BaseModel):

    def __init__(self, model, device="cpu", decoder=None):
        self.model = model.eval()
        self.device = device
        self.decoder = decoder

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

        # ---------- custom decoder ----------
        if self.decoder is not None:
            return float(self.decoder(pred))

        # ---------- normal pytorch ----------
        if torch.is_tensor(pred):
            return float(pred.cpu().numpy().flatten()[0])

        raise TypeError(
            f"Unsupported PyTorch output: {type(pred)}"
        )

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

        if self.decoder is not None:

            outputs = []

            if isinstance(preds, list):
                for p in preds:
                    outputs.append(float(self.decoder(p)))
            else:
                for i in range(batch.shape[0]):
                    single = {
                        k: v[i:i+1]
                        for k, v in preds.items()
                    }
                    outputs.append(float(self.decoder(single)))

            return outputs

        return preds.cpu().numpy().flatten().tolist()