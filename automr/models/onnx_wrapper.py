import numpy as np
import onnxruntime as ort

from automr.interfaces import BaseModel


class ONNXWrapper(BaseModel):

    def __init__(self, model_path):

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = (
            self.session
            .get_inputs()[0]
            .name
        )

    def predict(self, x):

        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if len(x.shape) == 3:
            x = np.expand_dims(
                x,
                axis=0
            )

        outputs = self.session.run(
            None,
            {
                self.input_name: x.astype(
                    np.float32
                )
            }
        )

        pred = outputs[0]

        return float(
            np.array(pred)
            .flatten()[0]
        )

    def predict_batch(self, xs):

        batch = np.array(
            xs,
            dtype=np.float32
        )

        outputs = self.session.run(
            None,
            {
                self.input_name: batch
            }
        )

        preds = outputs[0]

        return (
            np.array(preds)
            .flatten()
            .tolist()
        )