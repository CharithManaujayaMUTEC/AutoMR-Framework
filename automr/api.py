import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from automr.core.range_tester import RangeTester
from automr.analysis import Analyzer
from automr.models import get_wrapper
from automr.comparators import get_comparator
from automr.input_handlers import get_handler
from automr.registry import (
    TransformationRegistry,
    RelationRegistry
)
from automr.evaluation import BaselineEvaluator
from automr.logging import AutoMRLogger
from automr.verification import TransformationSaver
#from automr.epsilon import EpsilonManager
from automr.epsilon.utils import apply_epsilon_to_relations
from automr.epsilon.utils import generate_epsilon_values
from automr.epsilon.sensitivity import EpsilonSensitivity
from automr.epsilon.summary import EpsilonSummary

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
    DarkVisibiltyRelation,
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
        range_threshold=5.0
    ):
        self.image_saver = TransformationSaver()
        self.logger = AutoMRLogger()
        self.input_handler = get_handler(input_type)
        self.model = get_wrapper(model)
        self.range_tester = RangeTester()
        self.analyzer = Analyzer()
        self.comparator = get_comparator(
            task=task,
            epsilon=epsilon
        )
        self.range_threshold = range_threshold
        self.transform_registry = TransformationRegistry()
        self.relation_registry = RelationRegistry()
        self.mr_ranges = {}

        #self.epsilon_manager = EpsilonManager()

        self._register_default_mrs(epsilon)

    def _register_default_mrs(self, epsilon):

        # IMAGE
        self.transform_registry.register(
            "brightness",
            increase_brightness
        )
        self.relation_registry.register(
            "brightness",
            BrightnessRelation(tolerance=epsilon)
        )
        self.mr_ranges["brightness"] = (0.1, 3.0)

        self.transform_registry.register(
            "rotation",
            rotate_small
        )
        self.relation_registry.register(
            "rotation",
            RotationRelation(epsilon=epsilon)
        )
        self.mr_ranges["rotation"] = (-60, 60)

        self.transform_registry.register(
            "translation",
            shift_right
        )
        self.relation_registry.register(
            "translation",
            TranslationRelation(tolerance=epsilon)
        )
        self.mr_ranges["translation"] = (0, 80)

        self.transform_registry.register(
            "noise",
            add_noise
        )
        self.relation_registry.register(
            "noise",
            NoiseRelation(tolerance=epsilon)
        )
        self.mr_ranges["noise"] = (0, 150)

        self.transform_registry.register(
            "blur",
            blur
        )
        self.relation_registry.register(
            "blur",
            BlurRelation(epsilon=epsilon)
        )
        self.mr_ranges["blur"] = (1, 31)

        self.transform_registry.register(
            "contrast",
            adjust_contrast
        )
        self.relation_registry.register(
            "contrast",
            ContrastRelation(epsilon=epsilon)
        )
        self.mr_ranges["contrast"] = (0.1, 4.0)

        # WEATHER
        self.transform_registry.register("rain", add_rain)
        self.relation_registry.register(
            "rain",
            RainRelation(epsilon=epsilon)
        )
        self.mr_ranges["rain"] = (0.0, 1.5)

        self.transform_registry.register("snow", add_snow)
        self.relation_registry.register(
            "snow",
            SnowRelation(epsilon=epsilon)
        )
        self.mr_ranges["snow"] = (0.0, 1.5)

        self.transform_registry.register("fog", add_fog)
        self.relation_registry.register(
            "fog",
            FogRelation(epsilon=epsilon)
        )
        self.mr_ranges["fog"] = (0.0, 1.5)

        # TEMPORAL
        self.transform_registry.register(
            "temporal",
            next_frame_pair
        )
        self.relation_registry.register(
            "temporal",
            TemporalSmoothnessRelation(delta=epsilon)
        )
        self.mr_ranges["temporal"] = (0, 150)

        # BEHAVIORAL
        self.transform_registry.register(
            "visibility",
            reduce_visibility
        )
        self.relation_registry.register(
            "visibility",
            DarkVisibiltyRelation(max_change=epsilon)
        )
        self.mr_ranges["visibility"] = (0.05, 1.5)

        self.transform_registry.register(
            "darkness",
            darken
        )
        self.relation_registry.register(
            "darkness",
            DarkVisibiltyRelation(max_change=epsilon)
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

    # ---------- PLUGIN API ----------
    def register_transform(
        self,
        name,
        transform,
        relation,
        param_range
    ):
        self.transform_registry.register(
            name,
            transform
        )

        self.relation_registry.register(
            name,
            relation
        )

        self.mr_ranges[name] = param_range

    def list_transforms(self):
        return self.transform_registry.list()

    def list_relations(self):
        return self.relation_registry.list()
        
    # ---------- EXPECTED ----------
    def get_expected(self, relation_name):

        for name in self.relation_registry.list():

            relation = self.relation_registry.get(name)

            if relation.__class__.__name__ == relation_name:

                if hasattr(relation, "expected"):
                    return relation.expected()

        return "Invariant or monotonic behavior expected"
    # ---------- RUN SINGLE ----------
    def run_mr(self, input_data, mr_name, samples=50):

        #input_data = self.input_handler.preprocess(input_data)

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
            comparator=self.comparator,
            image_saver=self.image_saver,
            range_threshold=self.range_threshold
        )

        for r in results:
            r["severity"] = abs(r["difference"])
            self.logger.log(
                    f"MR={mr_name} "
                    f"param={r['param']} "
                    f"orig={r['original']} "
                    f"trans={r['transformed']} "
                    f"diff={r['difference']} "
                    f"pass={r['passed']}"
                )

        df = self.analyzer.to_dataframe(results)
        summary = self.analyzer.summary(df)

        return df, summary

    def run_all_mrs(self, input_data, samples=50):

       # if epsilon is not None:
       #     apply_epsilon_to_relations(
       #         self.relation_registry,
       #         epsilon
       #     )

        mr_names = [
            name
            for name in self.transform_registry.list()
            if name != "temporal"
        ]

        def run_single_mr(name):

            df, _ = self.run_mr(
                input_data,
                name,
                samples
            )

            return df

        max_workers = min(
            len(mr_names),
            8
        )

        with ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:

            results = list(
                executor.map(
                    run_single_mr,
                    mr_names
                )
            )

        return pd.concat(
            results,
            ignore_index=True
        )
    
    def _process_single_sample(self, args):

        i, sample, samples_per_mr, df_temp = args

        #sample = self.input_handler.preprocess(sample)

        #if sample is None:
        #    return None

        df_img = self.run_all_mrs(
            sample,
            samples=samples_per_mr,
        )

        if df_temp is not None:
            df = pd.concat(
                [df_img, df_temp],
                ignore_index=True
            )
        else:
            df = df_img

        df["sample_id"] = i

        df["expected_behavior"] = df["mr"].apply(
            self.get_expected
        )

        df["actual_behavior"] = df["status"].apply(
            lambda x: "Consistent"
            if x == "PASS"
            else "Violation"
        )

        return df

    # ---------- RUN DATASET ----------
    def run_dataset(
        self,
        dataset,
        max_samples=None,
        samples_per_mr=5,
        include_temporal=True,
        show_progress=False,
        epsilon=None,
    ):
        if epsilon is not None:
            apply_epsilon_to_relations(
                self.relation_registry,
                epsilon
            )

        if max_samples:
            dataset = dataset[:max_samples]

        all_results = []

        df_temp = None

        if include_temporal:
            try:
                df_temp, _ = self.run_mr(
                    dataset,
                    "temporal",
                    samples=samples_per_mr
                )
            except Exception:
                df_temp = None

        iterator = enumerate(dataset)

        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(dataset),
                desc="Running AutoMR"
            )

        for i, sample in iterator:

            #sample = self.input_handler.preprocess(sample)

            #if sample is None:
            #    continue

            df_img = self.run_all_mrs(
                sample,
                samples=samples_per_mr
            )

            if df_temp is not None:
                df = pd.concat(
                    [df_img, df_temp],
                    ignore_index=True
                )
            else:
                df = df_img

            df["sample_id"] = i

            df["expected_behavior"] = df["mr"].apply(
                self.get_expected
            )

            df["actual_behavior"] = df["status"].apply(
                lambda x:
                "Consistent"
                if x == "PASS"
                else "Violation"
            )

            all_results.append(df)

        return pd.concat(
            all_results,
            ignore_index=True
        )

    # ---------- ANALYSIS ----------
    def analyze(self, df):
        from automr.core.failure_analysis import FailureAnalyzer

        analyzer = FailureAnalyzer()

        return {
            "failure_summary": analyzer.failure_rate_per_mr(df),
            "severity_summary": analyzer.severity_per_mr(df),
            "worst_cases": analyzer.worst_cases(df, top_k=10),
            "regions": analyzer.failure_regions(df),
            "range_summary": analyzer.range_summary(df),
            "range_analysis": analyzer.range_analysis(df),
            "prediction_trace": self.analyzer.prediction_trace(df)
        }


    # ---------- SAVE ----------
    def save_results(self, df, results, output_dir="results"):
        
        os.makedirs(output_dir, exist_ok=True)

        df.to_csv(f"{output_dir}/automr_results.csv", index=False)
        results["failure_summary"].to_csv(f"{output_dir}/failure_summary.csv", index=False)
        results["severity_summary"].to_csv(f"{output_dir}/severity_summary.csv")
        results["worst_cases"].to_csv(f"{output_dir}/worst_cases.csv", index=False)
        results["range_summary"].to_csv(f"{output_dir}/range_summary.csv",index=False)
        results["range_analysis"].to_csv(f"{output_dir}/range_analysis.csv",index=False)
        results["prediction_trace"].to_csv(f"{output_dir}/prediction_trace.csv",index=False)
        if "epsilon_summary" in results:
            results["epsilon_summary"].to_csv(
                f"{output_dir}/epsilon_summary.csv",
                index=False
            )
        if "epsilon_report" in results:
            with open(f"{output_dir}/epsilon_report.txt", "w") as f:
                for k, v in results["epsilon_report"].items():
                    f.write(f"{k}: {v}\n")

        with open(f"{output_dir}/failure_regions.txt", "w") as f:
            for k, v in results["regions"].items():
                f.write(f"{k}: {v}\n")

    def save_baseline(
        self,
        dataset,
        output_dir="results",
        labels=None
    ):

        baseline = BaselineEvaluator(
            output_dir
        )

        baseline.save_dataset_info(
            dataset
        )

        predictions = []

        y_true = []
        y_pred = []

        for idx, sample in enumerate(dataset):

            try:

                #sample = self.input_handler.preprocess(
                #    sample
                #)

                pred = float(
                    self.model.predict(sample)
                )

                predictions.append({
                    "sample_id": idx,
                    "prediction": pred
                })

                if labels is not None:

                    y_true.append(
                        float(labels[idx])
                    )

                    y_pred.append(
                        pred
                    )

            except Exception as e:

                print(
                    f"Baseline failed: {idx}"
                )

        baseline.save_predictions(
            predictions
        )

        prediction_values = [
            pred["prediction"]
            for pred in predictions
        ]

        if prediction_values:
            baseline.save_basic_metrics(
                prediction_values
            )

        if labels is not None:

            mse = np.mean(
                (
                    np.array(y_true) -
                    np.array(y_pred)
                ) ** 2
            )

            mae = np.mean(
                np.abs(
                    np.array(y_true) -
                    np.array(y_pred)
                )
            )

            rmse = np.sqrt(mse)

            metrics = {
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse)
            }

            baseline.save_metrics(
                metrics
            )

    def save_model_summary(
        self,
        output_dir="results"
    ):

        try:

            model = self.model.model

            lines = []

            model.summary(
                print_fn=lambda x:
                lines.append(x)
            )

            text = "\n".join(lines)

        except Exception:

            text = str(
                type(self.model)
            )

        baseline = BaselineEvaluator(
            output_dir
        )

        baseline.save_model_summary(
            text
        )

    # ---------- FULL PIPELINE ----------
    def run_full_test(
        self,
        dataset,
        max_samples=None,
        samples_per_mr=5,
        show_progress=False,
        save=True,
        output_dir="results",
        verbose=True,
        epsilon_min=None,
        epsilon_max=None,
        epsilon_count=None,
    ):

        self.save_model_summary(
            output_dir
        )

        self.save_baseline(
            dataset,
            output_dir
        )

        # --------------------------------------------
        # Standard AutoMR
        # --------------------------------------------

        if epsilon_min is None:

            df = self.run_dataset(
                dataset,
                max_samples=max_samples,
                samples_per_mr=samples_per_mr,
                show_progress=show_progress
            )

            results = self.analyze(df)

        # --------------------------------------------
        # Epsilon Sensitivity Analysis
        # --------------------------------------------
        else:

            if epsilon_max is None:
                raise ValueError("epsilon_max must be provided.")

            if epsilon_count is None:
                raise ValueError("epsilon_count must be provided.")

            if epsilon_min >= epsilon_max:
                raise ValueError("epsilon_min must be smaller than epsilon_max.")

            if epsilon_count < 2:
                raise ValueError("epsilon_count must be at least 2.")

            epsilon_values = generate_epsilon_values(
                epsilon_min,
                epsilon_max,
                epsilon_count,
            )

            sensitivity = EpsilonSensitivity(self)

            dfs = sensitivity.run(
                dataset,
                epsilon_values,
                max_samples=max_samples,
                samples_per_mr=samples_per_mr,
                show_progress=show_progress,
            )

            summary = EpsilonSummary()

            summary_df, report = summary.summarize(dfs)

            summary.print_report(report)

            if len(dfs):
                df = pd.concat(dfs, ignore_index=True)
                results = self.analyze(df)
            else:
                df = pd.DataFrame()
                results = {}

            results["epsilon_summary"] = summary_df
            results["epsilon_report"] = report

        if save and results:
            self.save_results(
                df,
                results,
                output_dir
            )

        if verbose:

            print("\n=== AutoMR Results ===")
            if "failure_summary" in results:
                print(results["failure_summary"])

            print("\n--- Severity ---")
            if "severity_summary" in results:
                print(results["severity_summary"])

        return df, results