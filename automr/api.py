
from automr.core.tester import MRTester
from automr.core.range_tester import RangeTester
from automr.analysis import Analyzer

# transforms
from automr.transforms.image_transforms import (
    increase_brightness,
    rotate_small,
    shift_right,
    add_noise
)

# relations
from automr.relations.image_relations import (
    BrightnessRelation,
    RotationRelation,
    TranslationRelation,
    NoiseRelation
)


class AutoMR:

    def __init__(self, model, comparator):
        self.model = self._wrap_if_needed(model)
        self.comparator = comparator
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
    def run_mr(self, input_data, mr_name, samples=50):

    cfg = self.mr_config[mr_name]
    start, end = cfg["range"]

    results = self.range_tester.run_range(
        self.model,
        input_data,
        cfg["transform"],
        cfg["relation"],
        start,
        end,
        samples,
        self.comparator   # 🔥 NEW
    )

    df = self.analyzer.to_dataframe(results)
    summary = self.analyzer.summary(df)

    return df, summary

    # 🔥 run ALL MRs
    def run_all_mrs(self, input_data, samples=50):

    all_results = []

    for name in self.mr_config:
        df, _ = self.run_mr(input_data, name, samples)
        all_results.append(df)

    import pandas as pd
    return pd.concat(all_results, ignore_index=True)
