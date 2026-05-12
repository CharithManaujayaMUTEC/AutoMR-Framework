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

    def __init__(self, model, comparator):
        self.model = self._wrap_if_needed(model)
        self.comparator = comparator
        self.range_tester = RangeTester()
        self.analyzer = Analyzer()

        # 🔥 MR CONFIG
        self.mr_config = {
            # --- Image MRs ---
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
            },
            "blur": {
                "transform": blur,
                "relation": BlurRelation(),
                "range": (1, 9)
            },
            "contrast": {
                "transform": adjust_contrast,
                "relation": ContrastRelation(),
                "range": (0.5, 2.0)
            },
            "weather": {
                "transform": add_fog,
                "relation": WeatherRelation(),
                "range": (0.0, 0.7)
            },

            # --- Temporal MR ---
            "temporal": {
                "transform": next_frame_pair,
                "relation": TemporalSmoothnessRelation(),
                "range": (0, 50)
            },

            # --- Behavioral MRs ---
            "visibility": {
                "transform": reduce_visibility,
                "relation": LessSensitiveRelation(max_change=0.3),
                "range": (0.1, 0.8)
            },
            "darkness": {
                "transform": darken,
                "relation": LessSensitiveRelation(max_change=0.3),
                "range": (0.2, 0.8)
            }
        }

    # 🔥 Model wrapper (for PyTorch support)
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

    # 🔥 Run single MR
    def run_mr(self, input_data, mr_name, samples=50):

        cfg = self.mr_config[mr_name]
        start, end = cfg["range"]

        # ✅ Handle temporal vs image safely
        if mr_name == "temporal":
            data = input_data
        else:
            data = input_data[0] if isinstance(input_data, list) else input_data

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

        df = self.analyzer.to_dataframe(results)
        summary = self.analyzer.summary(df)

        return df, summary

    # 🔥 Run all MRs
    def run_all_mrs(self, input_data, samples=50):

        all_results = []

        for name in self.mr_config:

            # 🔥 skip temporal here (handled separately)
            if name == "temporal":
                continue

            df, _ = self.run_mr(input_data, name, samples)
            all_results.append(df)

        import pandas as pd
        return pd.concat(all_results, ignore_index=True)