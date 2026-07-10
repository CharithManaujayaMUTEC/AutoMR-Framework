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
    - Supports tensor, dict, tuple and list outputs
    - Optimized for inference
    """

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

        # Enable cuDNN autotuner for fixed-size inputs
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    # ==================================================
    # Single Prediction
    # ==================================================
    def predict(self, x):

        # -----------------------------------------
        # Optional preprocessing
        # -----------------------------------------
        if self.preprocess is not None:
            x = self.preprocess(x)

        x = np.ascontiguousarray(
            np.asarray(x, dtype=np.float32)
        )

        # HWC -> CHW
        if x.ndim == 3:
            x = np.transpose(x, (2, 0, 1))
            x = np.expand_dims(x, 0)

        x = torch.from_numpy(x).to(
            self.device,
            non_blocking=True,
        )

        # -----------------------------------------
        # Inference
        # -----------------------------------------
        with torch.inference_mode():
            pred = self.model(x)

        # -----------------------------------------
        # Optional decoder
        # -----------------------------------------
        if self.decoder is not None:
            return self.decoder(pred)

        # -----------------------------------------
        # Tensor output
        # -----------------------------------------
        if torch.is_tensor(pred):
            return float(
                pred.detach()
                .cpu()
                .reshape(-1)[0]
                .item()
            )

        # -----------------------------------------
        # Generic outputs
        # -----------------------------------------
        if isinstance(pred, (dict, list, tuple)):
            return pred

        raise TypeError(
            f"Unsupported PyTorch output: {type(pred)}"
        )

    # ==================================================
    # Batch Prediction
    # ==================================================
    def predict_batch(self, xs):
        """
        Predict a batch of inputs.

        Optimizations
        -------------
        - Batched preprocessing
        - Contiguous NumPy arrays
        - Single GPU transfer
        - torch.inference_mode()
        - Supports decoder outputs
        """

        if len(xs) == 0:
            return []

        # -----------------------------------------
        # Optional preprocessing
        # -----------------------------------------
        if self.preprocess is not None:
            xs = [
                self.preprocess(img)
                for img in xs
            ]

        # -----------------------------------------
        # Create contiguous batch
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

        batch = (
            torch.from_numpy(batch)
            .contiguous()
            .to(
                self.device,
                non_blocking=True,
            )
        )

        # -----------------------------------------
        # Forward pass
        # -----------------------------------------
        with torch.inference_mode():
            preds = self.model(batch)

        # -----------------------------------------
        # Decoder
        # -----------------------------------------
        if self.decoder is not None:

            outputs = []

            # Model returns dictionary
            if isinstance(preds, dict):

                batch_size = batch.shape[0]

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

            # Model returns tensor
            if torch.is_tensor(preds):

                for i in range(batch.shape[0]):
                    outputs.append(
                        self.decoder(
                            preds[i:i + 1]
                        )
                    )

                return outputs

            # Decoder for tuple/list outputs
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
                .cpu()
                .numpy()
                .tolist()
            )

        # -----------------------------------------
        # List / Tuple output
        # -----------------------------------------
        if isinstance(preds, (list, tuple)):
            return list(preds)

        # -----------------------------------------
        # Dictionary output
        # -----------------------------------------
        if isinstance(preds, dict):
            return preds

        return preds