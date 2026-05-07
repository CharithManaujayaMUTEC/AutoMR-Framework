from automr.core.tester import MRTester

from automr.transforms.geometric import flip
from automr.transforms.brightness import increase_brightness
from automr.transforms.translation import shift_right
from automr.transforms.noise import add_noise
from automr.transforms.crop import crop_top

from automr.relations.flip_relation import FlipRelation
from automr.relations.brightness_relation import BrightnessRelation
from automr.relations.translation_relation import TranslationRelation
from automr.relations.noise_relation import NoiseRelation
from automr.relations.crop_relation import CropRelation

class AutoMR:

    def __init__(self, model):
        self.model = self._wrap_if_needed(model)
        self.tester = MRTester()

        self.transforms = [
            flip,
            increase_brightness,
            shift_right,
            add_noise,
            crop_top
        ]

        self.relations = [
            FlipRelation(),
            BrightnessRelation(),
            TranslationRelation(),
            NoiseRelation(),
            CropRelation()
        ]

    def _wrap_if_needed(self, model):

        # Case 1: already has predict()
        if hasattr(model, "predict"):
            return model

        # Case 2: assume PyTorch model
        try:
            import torch

            class TorchWrapper:
                def __init__(self, model):
                    self.model = model

                def predict(self, x):
                    x = torch.tensor(x).permute(2,0,1).float().unsqueeze(0)
                    with torch.no_grad():
                        output = self.model(x)
                        # convert 1000-d output → single scalar
                        return float(output.max().item())

            return TorchWrapper(model)

        except Exception as e:
            pass

        raise ValueError("Model must have predict() or be wrappable")

    def test(self, input_data):
        return self.tester.run_all(
            self.model,
            input_data,
            self.transforms,
            self.relations
        )
