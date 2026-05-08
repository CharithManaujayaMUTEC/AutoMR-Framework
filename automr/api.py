
from automr.core.tester import MRTester
from automr.core.range_tester import RangeTester
from automr.analysis import Analyzer

# transforms
from automr.transforms.brightness import increase_brightness
from automr.transforms.rotation import rotate_small
from automr.transforms.translation import shift_right
from automr.transforms.noise import add_noise

# relations
from automr.relations.brightness_relation import BrightnessRelation
from automr.relations.rotation_relation import RotationRelation
from automr.relations.translation_relation import TranslationRelation
from automr.relations.noise_relation import NoiseRelation


class AutoMR:

    def __init__(self, model):
        self.model = self._wrap_if_needed(model)
        self.range_tester = RangeTester()
        self.analyzer = Analyzer()

        # 🔥 define MR config centrally
        self.mr_config = {
            "brightness": {
                "transform": increase_brightness,
                "relation": BrightnessRelation(),
                "range": (0.0, 2.0)
            },
            "rotation": {
                "transform": rotate_small,
                "relation": RotationRelation(),
                "range": (-15, 15)
            },
            "translation": {
                "transform": shift_right,
                "relation": TranslationRelation(),
                "range": (0, 20)
            },
            "noise": {
                "transform": add_noise,
                "relation": NoiseRelation(),
                "range": (0, 50)
            }
        }

    def _wrap_if_needed(self, model):

        if hasattr(model, "predict"):
            return model

        import torch

        class TorchWrapper:
            def __init__(self, model):
                self.model = model

            def predict(self, x):
                x = torch.tensor(x).permute(2,0,1).float().unsqueeze(0)
                with torch.no_grad():
                    output = self.model(x)
                return float(output.max().item())

        return TorchWrapper(model)

    # 🔥 run single MR with range
    def run_mr(self, image, mr_name, samples=50):

        cfg = self.mr_config[mr_name]

        start, end = cfg["range"]

        results = self.range_tester.run_range(
            self.model,
            image,
            cfg["transform"],
            cfg["relation"],
            start,
            end,
            samples
        )

        df = self.analyzer.to_dataframe(results)
        summary = self.analyzer.summary(df)

        return df, summary

    # 🔥 run ALL MRs
    def run_all_mrs(self, image, samples=50):

        all_results = []

        for name in self.mr_config:
            df, _ = self.run_mr(image, name, samples)
            all_results.append(df)

        import pandas as pd
        return pd.concat(all_results, ignore_index=True)
