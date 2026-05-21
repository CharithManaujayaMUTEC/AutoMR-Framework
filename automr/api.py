from automr.core.range_tester import RangeTester
from automr.analysis import Analyzer

# Image transforms
from automr.transforms.image_transforms import (
    increase_brightness,
    rotate_small,
    shift_right,
    add_noise,
    blur,
    adjust_contrast
)

# Weather transforms
from automr.transforms.weather_transforms import (
    add_rain,
    add_snow,
    add_fog
)

# Image relations
from automr.relations.image_relations import (
    BrightnessRelation,
    RotationRelation,
    TranslationRelation,
    NoiseRelation,
    BlurRelation,
    ContrastRelation
)

# Weather relations
from automr.relations.weather_relations import (
    RainRelation,
    SnowRelation,
    FogRelation
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

    def __init__(self, model, strict=True):
        self.model = self._wrap_if_needed(model)
        self.range_tester = RangeTester()
        self.analyzer = Analyzer()

        #  STRICT MODE
        if strict:
            eps_small = 0.015
            eps_medium = 0.025
        else:
            eps_small = 0.03
            eps_medium = 0.05

        #  MR CONFIG (FIXED PARAM NAMES)
        self.mr_config = {

            # ---------- IMAGE ----------
            "brightness": {
                "transform": increase_brightness,
                "relation": BrightnessRelation(tolerance=eps_small),  
                "range": (0.1, 3.0)
            },
            "rotation": {
                "transform": rotate_small,
                "relation": RotationRelation(epsilon=eps_small),
                "range": (-60, 60)
            },
            "translation": {
                "transform": shift_right,
                "relation": TranslationRelation(tolerance=eps_small),  
                "range": (0, 80)
            },
            "noise": {
                "transform": add_noise,
                "relation": NoiseRelation(tolerance=eps_small),  
                "range": (0, 150)
            },
            "blur": {
                "transform": blur,
                "relation": BlurRelation(epsilon=eps_small),
                "range": (1, 31)
            },
            "contrast": {
                "transform": adjust_contrast,
                "relation": ContrastRelation(epsilon=eps_small),
                "range": (0.1, 4.0)
            },

            # ---------- WEATHER ----------
            "rain": {
                "transform": add_rain,
                "relation": RainRelation(epsilon=eps_medium),
                "range": (0.0, 1.5)
            },
            "snow": {
                "transform": add_snow,
                "relation": SnowRelation(epsilon=eps_medium),
                "range": (0.0, 1.5)
            },
            "fog": {
                "transform": add_fog,
                "relation": FogRelation(epsilon=eps_medium),
                "range": (0.0, 1.5)
            },

            # ---------- TEMPORAL ----------
            "temporal": {
                "transform": next_frame_pair,
                "relation": TemporalSmoothnessRelation(delta=eps_small),
                "range": (0, 150)
            },

            # ---------- BEHAVIORAL ----------
            "visibility": {
                "transform": reduce_visibility,
                "relation": LessSensitiveRelation(max_change=0.08),
                "range": (0.05, 1.5)
            },
            "darkness": {
                "transform": darken,
                "relation": LessSensitiveRelation(max_change=0.08),
                "range": (0.05, 1.5)
            }
        }

    # ---------- MODEL WRAPPER ----------
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

    # ---------- EXPECTED ----------
    def get_expected(self, relation_name):
        for cfg in self.mr_config.values():
            if cfg["relation"].__class__.__name__ == relation_name:
                if hasattr(cfg["relation"], "expected"):
                    return cfg["relation"].expected()
        return "Invariant or monotonic behavior expected"

    # ---------- RUN SINGLE ----------
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
            comparator=None
        )

        for r in results:
            r["severity"] = abs(r["difference"])

        df = self.analyzer.to_dataframe(results)
        summary = self.analyzer.summary(df)

        return df, summary

    # ---------- RUN ALL ----------
    def run_all_mrs(self, input_data, samples=50):

        all_results = []

        for name in self.mr_config:

            if name == "temporal":
                continue

            df, _ = self.run_mr(input_data, name, samples)
            all_results.append(df)

        import pandas as pd
        return pd.concat(all_results, ignore_index=True)


    # ---------- RUN DATASET ----------
    def run_dataset(
        self,
        dataset,
        max_samples=None,
        samples_per_mr=5,
        include_temporal=True,
        show_progress=False
    ):
        import pandas as pd

        all_results = []

        if max_samples:
            dataset = dataset[:max_samples]

        # Temporal MR once
        df_temp = None
        if include_temporal:
            df_temp, _ = self.run_mr(dataset, "temporal", samples=samples_per_mr)

        iterator = dataset
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(dataset, desc="Running AutoMR")

        for i, sample in enumerate(iterator):

            if sample is None:
                continue

            df_img = self.run_all_mrs(sample, samples=samples_per_mr)

            if df_temp is not None:
                df = pd.concat([df_img, df_temp], ignore_index=True)
            else:
                df = df_img

            df["sample_id"] = i

            df["expected_behavior"] = df["mr"].apply(self.get_expected)
            df["actual_behavior"] = df["status"].apply(
                lambda x: "Consistent" if x == "PASS" else "Violation"
            )

            all_results.append(df)

        return pd.concat(all_results, ignore_index=True)


    # ---------- ANALYSIS ----------
    def analyze(self, df):
        from automr.core.failure_analysis import FailureAnalyzer

        analyzer = FailureAnalyzer()

        return {
            "failure_summary": analyzer.failure_rate_per_mr(df),
            "severity_summary": analyzer.severity_per_mr(df),
            "worst_cases": analyzer.worst_cases(df, top_k=10),
            "regions": analyzer.failure_regions(df),
        }


    # ---------- SAVE ----------
    def save_results(self, df, results, output_dir="results"):
        import os
        os.makedirs(output_dir, exist_ok=True)

        df.to_csv(f"{output_dir}/automr_results.csv", index=False)
        results["failure_summary"].to_csv(f"{output_dir}/failure_summary.csv", index=False)
        results["severity_summary"].to_csv(f"{output_dir}/severity_summary.csv")
        results["worst_cases"].to_csv(f"{output_dir}/worst_cases.csv", index=False)

        with open(f"{output_dir}/failure_regions.txt", "w") as f:
            for k, v in results["regions"].items():
                f.write(f"{k}: {v}\n")


    # ---------- FULL PIPELINE ----------
    def run_full_test(
        self,
        dataset,
        max_samples=None,
        samples_per_mr=5,
        show_progress=False,
        save=True,
        output_dir="results",
        verbose=True
    ):
        df = self.run_dataset(
            dataset,
            max_samples=max_samples,
            samples_per_mr=samples_per_mr,
            show_progress=show_progress
        )

        results = self.analyze(df)

        if save:
            self.save_results(df, results, output_dir)

        if verbose:
            print("\n=== AutoMR Results ===")
            print(results["failure_summary"])
            print("\n--- Severity ---")
            print(results["severity_summary"])

        return df, results