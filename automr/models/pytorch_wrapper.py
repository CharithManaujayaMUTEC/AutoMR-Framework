import numpy as np
import torch

from automr.interfaces import BaseModel


class PyTorchWrapper(BaseModel):
    """
    Generic PyTorch model wrapper.

    Features
    --------
    - Optional preprocessing hook
    - Optional decoder hook
    - Single prediction
    - Batch prediction
    - AMP support
    - torch.compile support
    - TF32 support
    - Pinned memory transfers
    """

    def __init__(
        self,
        model,
        device=None,
        preprocess=None,
        decoder=None,
    ):
        self.model = model.eval()

        if hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass

        if device is None:
            device = next(self.model.parameters()).device

        self.device = device
        self.preprocess = preprocess
        self.decoder = decoder

        if self.device.type == "cuda":

            torch.backends.cudnn.benchmark = True

            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = True

            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True

    # ==================================================
    # Single Prediction
    # ==================================================

    def predict(self, x):

        if self.preprocess is not None:
            x = self.preprocess(x)

        x = np.ascontiguousarray(
            np.asarray(x, dtype=np.float32)
        )

        if x.ndim == 3:
            x = np.transpose(x, (2, 0, 1))
            x = np.expand_dims(x, 0)

        tensor = torch.from_numpy(x)

        if self.device.type == "cuda":
            tensor = tensor.pin_memory()

        tensor = tensor.to(
            self.device,
            non_blocking=True,
        )

        with torch.inference_mode():

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.device.type == "cuda",
            ):

                pred = self.model(tensor)

        if self.decoder is not None:
            return self.decoder(pred)

        if torch.is_tensor(pred):

            return float(
                pred.detach()
                .reshape(-1)[0]
                .cpu()
                .item()
            )

        if isinstance(pred, (list, tuple, dict)):
            return pred

        return float(pred)
    
    # ==================================================
    # Batch Prediction
    # ==================================================

    def predict_batch(self, xs):
        """
        Optimized batch prediction.
        """

        if len(xs) == 0:
            return []

        # -----------------------------------------
        # Optional preprocessing
        # -----------------------------------------
        if self.preprocess is not None:
            xs = [
                self.preprocess(x)
                for x in xs
            ]

        # -----------------------------------------
        # Create contiguous NumPy batch
        # -----------------------------------------
        batch = np.ascontiguousarray(
            np.asarray(xs, dtype=np.float32)
        )

        # NHWC -> NCHW
        if batch.ndim == 4:
            batch = np.transpose(
                batch,
                (0, 3, 1, 2),
            )

        tensor = torch.from_numpy(batch)

        # Faster GPU transfer
        if self.device.type == "cuda":
            tensor = tensor.pin_memory()

        tensor = tensor.to(
            self.device,
            non_blocking=True,
        )

        # -----------------------------------------
        # Forward pass
        # -----------------------------------------
        with torch.inference_mode():

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.device.type == "cuda",
            ):

                preds = self.model(tensor)

        # -----------------------------------------
        # Decoder
        # -----------------------------------------
        if self.decoder is not None:

            outputs = []

            # Dictionary output
            if isinstance(preds, dict):

                batch_size = tensor.shape[0]

                for i in range(batch_size):

                    single_pred = {}

                    for k, v in preds.items():

                        if torch.is_tensor(v):
                            single_pred[k] = v[i:i + 1]
                        else:
                            single_pred[k] = v

                    outputs.append(
                        self.decoder(single_pred)
                    )

                return outputs

            # Tensor output
            if torch.is_tensor(preds):

                for i in range(tensor.shape[0]):

                    outputs.append(
                        self.decoder(
                            preds[i:i + 1]
                        )
                    )

                return outputs

            # Tuple/List output
            if isinstance(preds, (list, tuple)):

                for pred in preds:

                    outputs.append(
                        self.decoder(pred)
                    )

                return outputs

        # -----------------------------------------
        # Tensor output
        # -----------------------------------------
        if torch.is_tensor(preds):

            return (
                preds.detach()
                .flatten()
                .cpu()
                .numpy()
                .astype(float)
                .tolist()
            )

        # -----------------------------------------
        # List / Tuple output
        # -----------------------------------------
        if isinstance(preds, (list, tuple)):

            return [
                float(p.detach().cpu().item())
                if torch.is_tensor(p)
                else float(p)
                for p in preds
            ]

        # -----------------------------------------
        # Dictionary output
        # -----------------------------------------
        if isinstance(preds, dict):

            outputs = []

            batch_size = tensor.shape[0]

            for i in range(batch_size):

                value = None

                for _, v in preds.items():

                    if torch.is_tensor(v):
                        value = float(
                            v[i]
                            .detach()
                            .cpu()
                            .reshape(-1)[0]
                            .item()
                        )
                        break

                outputs.append(value)

            return outputs
        
        # -----------------------------------------
        # Scalar output
        # -----------------------------------------
        if np.isscalar(preds):
            return [float(preds)] * len(xs)

        return preds