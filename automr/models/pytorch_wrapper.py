import numpy as np
import torch

from automr.interfaces import BaseModel


class PyTorchWrapper(BaseModel):

    def __init__(
        self,
        model,
        device=None,
        preprocess=None,
        decoder=None,
    ):
        self.model = model.eval()

        if device is None:
            device = next(model.parameters()).device

        self.device = device
        self.preprocess = preprocess
        self.decoder = decoder

    def predict(self, x):

        # Optional preprocessing
        if self.preprocess is not None:
            x = self.preprocess(x)

        x = np.asarray(x, dtype=np.float32)

        if x.ndim == 3:
            x = np.transpose(x, (2, 0, 1))
            x = np.expand_dims(x, 0)

        x = torch.from_numpy(x).float().to(self.device)

        with torch.no_grad():
            pred = self.model(x)

        # Optional decoder
        if self.decoder is not None:
            return self.decoder(pred)

        # Tensor output
        if torch.is_tensor(pred):
            return float(pred.cpu().numpy().flatten()[0])

        # Generic outputs
        if isinstance(pred, (dict, list, tuple)):
            return pred

        raise TypeError(f"Unsupported PyTorch output: {type(pred)}")

    def predict_batch(self, xs):

        if self.preprocess is not None:
            batch = np.stack([self.preprocess(img) for img in xs])
        else:
            batch = np.asarray(xs, dtype=np.float32)

        if batch.ndim == 4:
            batch = np.transpose(batch, (0, 3, 1, 2))

        batch = torch.from_numpy(batch).float().to(self.device)

        with torch.no_grad():
            preds = self.model(batch)

        if self.decoder is not None:

            if isinstance(preds, dict):
                outputs = []

                for i in range(batch.shape[0]):
                    single = {
                        k: (v[i:i+1] if torch.is_tensor(v) else v)
                        for k, v in preds.items()
                    }

                    outputs.append(self.decoder(single))

                return outputs

            if isinstance(preds, (list, tuple)):
                return [self.decoder(p) for p in preds]

            return [self.decoder(preds)]

        if torch.is_tensor(preds):
            return preds.cpu().numpy().tolist()

        return preds