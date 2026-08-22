"""
PyTorch model wrapper.

This wrapper provides a standardized interface between AutoMR and
PyTorch models. It supports single and batch inference, optional
preprocessing and decoding hooks, GPU acceleration, automatic mixed
precision (AMP), and optional model compilation.
"""

import platform
import torch
import numpy as np

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
        enable_compile=False,   # default OFF
    ):
        """
        Initialize the PyTorch wrapper.

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model.
        device : torch.device, optional
            Target execution device.
        preprocess : callable, optional
            Input preprocessing function.
        decoder : callable, optional
            Output decoding function.
        enable_compile : bool, default=False
            Enable torch.compile when supported.
        """
        self.model = model.eval()

        # -------------------------------------------------
        # Optional torch.compile
        # Disabled by default because Windows has no Triton.
        # -------------------------------------------------
        if (
            enable_compile
            and hasattr(torch, "compile")
            and platform.system() != "Windows"
        ):
            try:
                self.model = torch.compile(
                    self.model,
                    mode="reduce-overhead",
                )
                print("torch.compile enabled.")
            except Exception as e:
                print(f"torch.compile disabled: {e}")

        # Automatically detect the execution device if not provided.
        if device is None:
            device = next(self.model.parameters()).device

        self.device = device
        self.preprocess = preprocess
        self.decoder = decoder

        # Configure CUDA-specific performance optimizations.
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
        """
        Generate a prediction for a single input sample.
        """

        # Apply optional preprocessing.
        if self.preprocess is not None:
            x = self.preprocess(x)

        # Convert input to contiguous float32 NumPy format.
        x = np.ascontiguousarray(
            np.asarray(x, dtype=np.float32)
        )

        # Convert NHWC images to NCHW format.
        if x.ndim == 3:
            x = np.transpose(x, (2, 0, 1))
            x = np.expand_dims(x, 0)

        # Create a PyTorch tensor.
        tensor = torch.from_numpy(x)

        # Use pinned memory for faster GPU transfer.
        if self.device.type == "cuda":
            tensor = tensor.pin_memory()

        # Transfer the tensor to the execution device.
        tensor = tensor.to(
            self.device,
            non_blocking=True,
        ).contiguous()

        # Execute model inference.
        with torch.inference_mode():

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.device.type == "cuda",
            ):

                pred = self.model(tensor)
        # --------------------------------------------------
        # Model-specific decoder
        # --------------------------------------------------
        if self.decoder is not None:

            # Ensure tensors passed to external decoders are
            # detached from the computation graph and moved
            # from GPU/CUDA memory to CPU memory.
            #
            # This allows NumPy and OpenCV based decoder logic
            # to safely process model outputs.
            if torch.is_tensor(pred):

                pred = (
                    pred
                    .detach()
                    .cpu()
                )

            return self.decoder(pred)

        # Handle tensor outputs.
        if torch.is_tensor(pred):

            return float(
                pred.detach()
                .reshape(-1)[0]
                .cpu()
                .item()
            )

        # Return complex outputs unchanged.
        if isinstance(pred, (list, tuple, dict)):
            return pred

        # Handle scalar outputs.
        return float(pred)

    # ==================================================
    # Batch Prediction
    # ==================================================

    def predict_batch(self, xs):
        """
        Optimized batch prediction.
        """

        # Return immediately for empty batches.
        if len(xs) == 0:
            return []

        # -----------------------------------------
        # Optional preprocessing
        # -----------------------------------------
        # Apply preprocessing to every sample.
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

        # Create a PyTorch tensor.
        tensor = torch.from_numpy(batch)

        # Faster GPU transfer.
        if self.device.type == "cuda":
            tensor = tensor.pin_memory()

        # Move the batch to the target device.
        tensor = tensor.to(
            self.device,
            non_blocking=True,
        ).contiguous()

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
        # Decode predictions when a custom decoder is available.
        if self.decoder:
            decoded = []

            for i in range(len(batch)):

                if isinstance(preds, dict):

                    single_pred = {}

                    for k, v in preds.items():

                        if torch.is_tensor(v):

                            single_pred[k] = (
                                v[i:i + 1]
                                .detach()
                                .cpu()
                                .numpy()
                            )

                        else:

                            single_pred[k] = v[i:i + 1]

                    result = self.decoder(
                        single_pred
                    )

                else:

                    single_prediction = preds[i:i + 1]

                    if torch.is_tensor(single_prediction):

                        single_prediction = (
                            single_prediction
                            .detach()
                            .cpu()
                            .numpy()
                        )

                    result = self.decoder(
                        single_prediction
                    )

                decoded.append(result)

            return decoded
        
        # -----------------------------------------
        # Tensor output
        # -----------------------------------------
        # Convert tensor predictions to a Python list.
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
        # Convert each prediction to float.
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
        # Extract one prediction value from each dictionary output.
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
        # Replicate scalar predictions across the batch.
        if np.isscalar(preds):
            return [float(preds)] * len(xs)

        # Return unsupported output formats unchanged.
        return preds