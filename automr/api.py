from automr.core.range_tester import RangeTester
from automr.analysis import Analyzer

# Image transforms
from automr.transforms.image_transforms import (
    increase_brightness,
    rotate_small,
    shift_right,
    add_noise,
    blur,
    adjust_contrast,
    add_fog
)

# Image relations
from automr.relations.image_relations import (
    BrightnessRelation,
    RotationRelation,
    TranslationRelation,
    NoiseRelation,
    BlurRelation,
    ContrastRelation,
    WeatherRelation
)

# Temporal
from automr.transforms.temporal_transforms import next_frame_pair
from automr.relations.temporal_relations import TemporalSmoothnessRelation

# Behavioral
from automr.transforms.behavioral_transforms import (
    reduce_visibility,
    darken
)

from automr.relations.behavioral_relations import (
    LessSensitiveRelation,
    MonotonicIncreaseRelation,
    MonotonicDecreaseRelation
)


class AutoMR:

    def __init__(self, model, comparator=None):  #  comparator optional
        self.model = self._wrap_if_needed(model)
        self.comparator = comparator
        self.range_tester = RangeTester()
        self.analyzer = Analyzer()

        #  RIGOROUS MR CONFIG (UPDATED)
        self.mr_config = {

            # -------- IMAGE MRs --------
            "brightness": {
                "transform": increase_brightness,
                "relation": BrightnessRelation(tolerance=0.02),   # stricter
                "range": (0.2, 2.5)   # wider
            },
            "rotation": {
                "transform": rotate_small,
                "relation": RotationRelation(epsilon=0.08),
                "range": (-25, 25)
            },
            "translation": {
                "transform": shift_right,
                "relation": TranslationRelation(tolerance=0.08),
                "range": (0, 40)
            },
            "noise": {
                "transform": add_noise,
                "relation": NoiseRelation(tolerance=0.05),
                "range": (0, 100)
            },
            "blur": {
                "transform": blur,
                "relation": BlurRelation(epsilon=0.05),
                "range": (1, 15)
            },
            "contrast": {
                "transform": adjust_contrast,
                "relation": ContrastRelation(epsilon=0.05),
                "range": (0.2, 3.0)
            },
            "weather": {
                "transform": add_fog,
                "relation": WeatherRelation(epsilon=0.08),
                "range": (0.0, 1.0)
            },

            # -------- TEMPORAL --------
            "temporal": {
                "transform": next_frame_pair,
                "relation": TemporalSmoothnessRelation(delta=0.05),  # stricter
                "range": (0, 100)
            },

            # -------- BEHAVIORAL --------
            "visibility": {
                "transform": reduce_visibility,
                "relation": LessSensitiveRelation(max_change=0.15),
                "range": (0.1, 1.0)
            },
            "darkness": {
                "transform": darken,
                "relation": LessSensitiveRelation(max_change=0.15),
                "range": (0.1, 1.0)
            }
        }

    # -------- MODEL WRAPPER --------
    def _wrap_if_needed(self, model):

        if hasattr(model, "predict"):
            return model

        import torch

        class TorchWrapper:
            def __init__(self, model):
                self.model = model

            def predict(self, x):
                x = torch.tensor(x).permute(2, 0, 1).float().unsqueeze(0)
                with torch.no_grad():
                    output = self.model(x)
                return float(output.max().item())

        return TorchWrapper(model)

    # -------- EXPECTED BEHAVIOR --------
    def get_expected(self, relation_name):
        for cfg in self.mr_config.values():
            if cfg["relation"].__class__.__name__ == relation_name:
                if hasattr(cfg["relation"], "expected"):
                    return cfg["relation"].expected()
        return "Standard invariance"

    # -------- RUN SINGLE MR --------
    def run_mr(self, input_data, mr_name, samples=50):

        cfg = self.mr_config[mr_name]
        start, end = cfg["range"]

        #  KEEP THIS SIMPLE & CORRECT
        data = input_data

        results = self.range_tester.run_range(
            self.model,
            data,
            cfg["transform"],
            cfg["relation"],
            start,
            end,
            samples,
            self.comparator
        )

        #  ADD SEVERITY (important for research)
        for r in results:
            r["severity"] = abs(r["difference"])

        df = self.analyzer.to_dataframe(results)
        summary = self.analyzer.summary(df)

        return df, summary

    # -------- RUN ALL NON-TEMPORAL --------
    def run_all_mrs(self, input_data, samples=50):

        all_results = []

        for name in self.mr_config:

            #  DO NOT mix temporal here
            if name == "temporal":
                continue

            df, _ = self.run_mr(input_data, name, samples)
            all_results.append(df)

        import pandas as pd
        return pd.concat(all_results, ignore_index=True)