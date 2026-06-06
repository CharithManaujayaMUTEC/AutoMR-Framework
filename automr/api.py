from automr.core.range_tester import RangeTester
from automr.analysis import Analyzer
from automr.models import get_wrapper
from automr.comparators import get_comparator
from automr.input_handlers import get_handler
from automr.registry import (
    TransformationRegistry,
    RelationRegistry
)

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

        def __init__(
            self,
            model,
            task="regression",
            input_type="image",
            epsilon=0.05,
            strict=True
        ):
        self.input_handler = get_handler(input_type)
        self.model = get_wrapper(model)
        self.range_tester = RangeTester()
        self.analyzer = Analyzer()
        self.comparator = get_comparator(
            task=task,
            epsilon=epsilon
        )

        #  STRICT MODE
        if strict:
            eps_small = 0.015
            eps_medium = 0.025
        else:
            eps_small = 0.03
            eps_medium = 0.05

        self._register_default_mrs(
            eps_small,
            eps_medium
        )

        #  MR CONFIG (FIXED PARAM NAMES)
        self.transform_registry = TransformationRegistry()
        self.relation_registry = RelationRegistry()
        self.mr_ranges = {}

    def _register_default_mrs(self, eps_small, eps_medium):

    # IMAGE
    self.transform_registry.register(
        "brightness",
        increase_brightness
    )
    self.relation_registry.register(
        "brightness",
        BrightnessRelation(tolerance=eps_small)
    )
    self.mr_ranges["brightness"] = (0.1, 3.0)

    self.transform_registry.register(
        "rotation",
        rotate_small
    )
    self.relation_registry.register(
        "rotation",
        RotationRelation(epsilon=eps_small)
    )
    self.mr_ranges["rotation"] = (-60, 60)

    self.transform_registry.register(
        "translation",
        shift_right
    )
    self.relation_registry.register(
        "translation",
        TranslationRelation(tolerance=eps_small)
    )
    self.mr_ranges["translation"] = (0, 80)

    self.transform_registry.register(
        "noise",
        add_noise
    )
    self.relation_registry.register(
        "noise",
        NoiseRelation(tolerance=eps_small)
    )
    self.mr_ranges["noise"] = (0, 150)

    self.transform_registry.register(
        "blur",
        blur
    )
    self.relation_registry.register(
        "blur",
        BlurRelation(epsilon=eps_small)
    )
    self.mr_ranges["blur"] = (1, 31)

    self.transform_registry.register(
        "contrast",
        adjust_contrast
    )
    self.relation_registry.register(
        "contrast",
        ContrastRelation(epsilon=eps_small)
    )
    self.mr_ranges["contrast"] = (0.1, 4.0)

    # WEATHER
    self.transform_registry.register("rain", add_rain)
    self.relation_registry.register(
        "rain",
        RainRelation(epsilon=eps_medium)
    )
    self.mr_ranges["rain"] = (0.0, 1.5)

    self.transform_registry.register("snow", add_snow)
    self.relation_registry.register(
        "snow",
        SnowRelation(epsilon=eps_medium)
    )
    self.mr_ranges["snow"] = (0.0, 1.5)

    self.transform_registry.register("fog", add_fog)
    self.relation_registry.register(
        "fog",
        FogRelation(epsilon=eps_medium)
    )
    self.mr_ranges["fog"] = (0.0, 1.5)

    # TEMPORAL
    self.transform_registry.register(
        "temporal",
        next_frame_pair
    )
    self.relation_registry.register(
        "temporal",
        TemporalSmoothnessRelation(delta=eps_small)
    )
    self.mr_ranges["temporal"] = (0, 150)

    # BEHAVIORAL
    self.transform_registry.register(
        "visibility",
        reduce_visibility
    )
    self.relation_registry.register(
        "visibility",
        LessSensitiveRelation(max_change=0.08)
    )
    self.mr_ranges["visibility"] = (0.05, 1.5)

    self.transform_registry.register(
        "darkness",
        darken
    )
    self.relation_registry.register(
        "darkness",
        LessSensitiveRelation(max_change=0.08)
    )
    self.mr_ranges["darkness"] = (0.05, 1.5)

    # ---------- MODEL WRAPPER ----------
    #def _wrap_if_needed(self, model):

    #    if hasattr(model, "predict"):
    #        return model

    #    import torch

    #    class TorchWrapper:
    #        def __init__(self, model):
    #            self.model = model

    #        def predict(self, x):
    #            x = torch.tensor(x).permute(2, 0, 1).float().unsqueeze(0)
    #            with torch.no_grad():
    #                output = self.model(x)
    #            return float(output.max().item())

        return TorchWrapper(model)

    # ---------- EXPECTED ----------
    def get_expected(self, relation_name):
        for name in self.relation_registry.list():

        relation = self.relation_registry.get(name)

        if relation.__class__.__name__ == relation_name:

        if hasattr(relation, "expected"):
            return relation.expected()
            if cfg["relation"].__class__.__name__ == relation_name:
                if hasattr(cfg["relation"], "expected"):
                    return cfg["relation"].expected()
        return "Invariant or monotonic behavior expected"

    # ---------- RUN SINGLE ----------
    def run_mr(self, input_data, mr_name, samples=50):

        input_data = self.input_handler.preprocess(input_data)

        transform = self.transform_registry.get(mr_name)
        relation = self.relation_registry.get(mr_name)

        start, end = self.mr_ranges[mr_name]

        results = self.range_tester.run_range(
            self.model,
            input_data,
            transform,
            relation,
            start,
            end,
            samples,
            comparator=self.comparator
        )

    # ---------- RUN ALL ----------
    def run_all_mrs(self, input_data, samples=50):

        all_results = []

        for name in self.transform_registry.list():

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

            sample = self.input_handler.preprocess(sample)

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