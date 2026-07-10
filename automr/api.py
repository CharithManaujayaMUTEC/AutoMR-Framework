import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor
from automr.core.range_tester import RangeTester
from automr.analysis import Analyzer
from automr.models import get_wrapper
from automr.comparators import get_comparator
from automr.input_handlers import get_handler
from automr.registry import (
    TransformationRegistry,
    RelationRegistry,
)

from automr.registry.default_transforms import (
    register_default_transforms,
)

from automr.registry.default_relations import (
    register_default_relations,
)

from automr.evaluation import BaselineEvaluator
from automr.logging import AutoMRLogger
from automr.verification import TransformationSaver

from automr.epsilon.utils import (
    apply_epsilon_to_relations,
    generate_epsilon_values,
)

from automr.epsilon.sensitivity import (
    EpsilonSensitivity,
)

from automr.epsilon.summary import (
    EpsilonSummary,
)

class AutoMR:
    def __init__(
        self,
        model,
        task="regression",
        input_type="image",
        epsilon=0.05,
        range_threshold=5.0,
    ):

        # --------------------------------------------------
        # Core Components
        # --------------------------------------------------

        self.image_saver = TransformationSaver()
        self.logger = AutoMRLogger()
        self.input_handler = get_handler(input_type)
        self.model = get_wrapper(model)
        self.range_tester = RangeTester()
        self.analyzer = Analyzer()

        self.task = task
        self.range_threshold = range_threshold

        # --------------------------------------------------
        # Registries
        # --------------------------------------------------

        self.transform_registry = TransformationRegistry()
        self.relation_registry = RelationRegistry()

        # MR parameter ranges
        self.mr_ranges = {}

        # Comparator
        self.comparator = get_comparator(
            task=task,
            epsilon=epsilon,
        )

        # Register built-in transformations and relations
        self._register_default_mrs(epsilon)

    def _register_default_mrs(self, epsilon):
            """
            Register all built-in transformations,
            relations and their parameter ranges.
            """

            # Register built-in transformations
            register_default_transforms(
                self.transform_registry
            )

            # Register built-in relations
            register_default_relations(
                self.relation_registry,
                epsilon
            )

            # Default parameter ranges for each MR
            self.mr_ranges = {

                # -------------------------------------------------
                # Image
                # -------------------------------------------------
                "brightness": (0.1, 3.0),
                "rotation": (-60, 60),
                "translation": (0, 80),
                "noise": (0, 150),
                "blur": (1, 31),
                "contrast": (0.1, 4.0),
                "composite": (0.1, 1.5),

                # -------------------------------------------------
                # Weather
                # -------------------------------------------------
                "rain": (0.0, 1.5),
                "snow": (0.0, 1.5),
                "fog": (0.0, 1.5),
                "sandstorm": (0.0, 1.5),
                "dust": (0.0, 1.5),
                "haze": (0.0, 1.5),
                "smoke": (0.0, 1.5),

                # -------------------------------------------------
                # Behavioral
                # -------------------------------------------------
                "visibility": (0.05, 1.5),
                "darkness": (0.05, 1.5),

                # -------------------------------------------------
                # Temporal
                # -------------------------------------------------
                "temporal": (0, 150),
            }

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

    # ==================================================
    # Plugin API
    # ==================================================

    def register_transform(
        self,
        name,
        transform,
        relation,
        param_range,
    ):
        """
        Register a custom transformation and its
        corresponding metamorphic relation.
        """

        self.transform_registry.register(
            name,
            transform,
        )

        self.relation_registry.register(
            name,
            relation,
        )

        self.mr_ranges[name] = param_range

    def unregister_transform(self, name):
        """
        Remove a registered transformation.
        """

        if name in self.transform_registry.transforms:
            del self.transform_registry.transforms[name]

        if name in self.relation_registry.relations:
            del self.relation_registry.relations[name]

        if name in self.mr_ranges:
            del self.mr_ranges[name]

    def has_transform(self, name):
        """
        Check whether a transformation exists.
        """

        return name in self.transform_registry.transforms

    def get_transform(self, name):
        """
        Return a registered transformation.
        """

        return self.transform_registry.get(name)

    def get_relation(self, name):
        """
        Return the relation associated with a transformation.
        """

        return self.relation_registry.get(name)

    def list_transforms(self):
        """
        List all registered transformations.
        """

        return sorted(self.transform_registry.list())

    def list_relations(self):
        """
        List all registered relations.
        """

        return sorted(self.relation_registry.list())
          
    # ---------- EXPECTED ----------
    def get_expected(self, relation_name):

        for name in self.relation_registry.list():

            relation = self.relation_registry.get(name)

            if relation.__class__.__name__ == relation_name:

                if hasattr(relation, "expected"):
                    return relation.expected()

        return "Invariant or monotonic behavior expected"

    # ==================================================
    # Run a Single Metamorphic Relation
    # ==================================================

    def run_mr(
        self,
        input_data,
        mr_name,
        samples=50,
        original_prediction=None,
        prediction_cache=None,      # NEW
    ):
        """
        Execute a single metamorphic relation.

        Optimization:
        - Original prediction is computed only once.
        - Reused across every MR.
        """

        if not self.has_transform(mr_name):
            raise ValueError(
                f"Unknown metamorphic relation: '{mr_name}'"
            )

        transform = self.get_transform(mr_name)
        relation = self.get_relation(mr_name)

        start, end = self.mr_ranges[mr_name]

        results = self.range_tester.run_range(
            model=self.model,
            input_data=input_data,
            transform_fn=transform,
            relation=relation,
            start=start,
            end=end,
            num_samples=samples,
            comparator=self.comparator,
            image_saver=self.image_saver,
            range_threshold=self.range_threshold,
            original_prediction=original_prediction,
            prediction_cache=prediction_cache,
        )

        for result in results:

            result["severity"] = abs(result["difference"])

            self.logger.log(
                f"MR={mr_name} "
                f"param={result['param']} "
                f"orig={result['original']} "
                f"trans={result['transformed']} "
                f"diff={result['difference']} "
                f"pass={result['passed']}"
            )

        df = self.analyzer.to_dataframe(results)

        summary = self.analyzer.summary(df)

        return df, summary

    # ==================================================
    # Run All Metamorphic Relations
    # ==================================================

    def run_all_mrs(
        self,
        input_data,
        samples=50,
        exclude=None,
        original_prediction=None,
    ):
        """
        Execute all registered metamorphic relations.

        Optimized:
        - Compute original prediction once.
        - Keep MR execution sequential.
        - Each MR already performs batched inference internally.
        """

        if exclude is None:
            exclude = ["temporal"]

        if original_prediction is None:
            original_prediction = float(
                self.model.predict(input_data)
            )

        dfs = []

        for mr_name in self.list_transforms():

            if mr_name in exclude:
                continue

            df, _ = self.run_mr(
                input_data=input_data,
                mr_name=mr_name,
                samples=samples,
                original_prediction=original_prediction,
            )

            dfs.append(df)

        return pd.concat(
            dfs,
            ignore_index=True,
        )

    def _process_single_sample(self, args):
        """
        Process one sample.

        Optimized:
        - Original prediction computed only once.
        - Reused across every MR.
        """

        sample_id, sample, samples_per_mr, df_temp = args

        # Compute original prediction once
        original_prediction = float(
            self.model.predict(sample)
        )

        # Run all image MRs
        df_img = self.run_all_mrs(
            input_data=sample,
            samples=samples_per_mr,
            original_prediction=original_prediction,
        )

        # Merge temporal results if available
        if df_temp is not None and not df_temp.empty:
            df = pd.concat(
                [df_img, df_temp],
                ignore_index=True,
                copy=False,
            )
        else:
            df = df_img

        df["sample_id"] = sample_id

        df["expected_behavior"] = [
            self.get_expected(mr)
            for mr in df["mr"]
        ]

        df["actual_behavior"] = np.where(
            df["status"] == "PASS",
            "Consistent",
            "Violation",
        )

        return df

    #for dashboard
    def set_epsilon(self, epsilon):
        """
        Update the comparator and every registered MR with a new epsilon.
        """

        apply_epsilon_to_relations(
            self.relation_registry,
            epsilon
        )

        self.comparator = get_comparator(
            task=self.task,
            epsilon=epsilon
        )

    # ==================================================
    # Run Dataset
    # ==================================================

    def run_dataset(
        self,
        dataset,
        max_samples=None,
        samples_per_mr=5,
        include_temporal=True,
        show_progress=False,
        epsilon=None,
        prediction_cache=None,   # NEW
    ):
        """
        Run AutoMR over an entire dataset.

        Optimizations:
        - Faster dataset limiting
        - Progress bar with IPS
        - Background image pre-loading
        - Timing statistics
        - No architecture changes
        """

        import time
        from concurrent.futures import ThreadPoolExecutor

        # -------------------------------------------------------
        # Update epsilon if requested
        # -------------------------------------------------------
        if epsilon is not None:
            self.set_epsilon(epsilon)

        # -------------------------------------------------------
        # Limit dataset
        # -------------------------------------------------------
        if max_samples is not None:
            total_images = min(max_samples, len(dataset))
        else:
            total_images = len(dataset)

        # -------------------------------------------------------
        # Temporal MR (run only once)
        # -------------------------------------------------------
        df_temp = None

        if include_temporal:
            try:
                temporal_data = [
                    dataset[i]
                    for i in range(total_images)
                ]

                df_temp, _ = self.run_mr(
                    input_data=temporal_data,
                    mr_name="temporal",
                    samples=samples_per_mr,
                )

            except Exception:
                df_temp = None

        # -------------------------------------------------------
        # Timing
        # -------------------------------------------------------
        start_time = time.time()

        load_time = 0.0
        process_time = 0.0

        all_results = []

        # ---------------------------------------
        # Shared cache (reuse predictions)
        # ---------------------------------------
        if prediction_cache is None:
            prediction_cache = {}

        # -------------------------------------------------------
        # Progress bar
        # -------------------------------------------------------
        if show_progress:

            iterator = tqdm(
                range(total_images),
                total=total_images,
                desc="Running AutoMR",
                unit="image",
                colour="cyan",
                dynamic_ncols=True,
            )

        else:

            iterator = range(total_images)

        # ---------------------------------------
        # Parallel image loading
        # ---------------------------------------
        loader = ThreadPoolExecutor(max_workers=2)

        future = loader.submit(dataset.__getitem__, 0)

        # ---------------------------------------
        # Parallel MR execution
        # ---------------------------------------
        workers = max(
            1,
            min(
                multiprocessing.cpu_count(),
                8,
            ),
        )

        executor = ThreadPoolExecutor(max_workers=workers)

        pending = []

        for idx in iterator:

            try:

                # ---------------------------------------------
                # Load image
                # ---------------------------------------------
                t0 = time.perf_counter()

                sample = future.result()

                load_time += time.perf_counter() - t0

                # ---------------------------------------------
                # Preload next image
                # ---------------------------------------------
                if idx + 1 < total_images:
                    future = executor.submit(
                        dataset.__getitem__,
                        idx + 1,
                    )

                # ---------------------------------------------
                # Execute all MRs
                # ---------------------------------------------
                t0 = time.perf_counter()

                future_df = executor.submit(
                    self._process_single_sample,
                    (
                        idx,
                        sample,
                        samples_per_mr,
                        df_temp,
                        prediction_cache,
                    ),
                )

                pending.append(future_df)

                process_time += time.perf_counter() - t0

                # ---------------------------------------------
                # Update progress
                # ---------------------------------------------
                if show_progress:

                    elapsed = time.time() - start_time

                    ips = (
                        (idx + 1) / elapsed
                        if elapsed > 0
                        else 0
                    )

                    total_profile = (
                        load_time + process_time
                    )

                    load_percent = (
                        100 * load_time / total_profile
                        if total_profile > 0
                        else 0
                    )

                    mr_percent = (
                        100 * process_time / total_profile
                        if total_profile > 0
                        else 0
                    )

                    iterator.set_postfix({
                        "IPS": f"{ips:.2f}",
                        "Done": f"{idx+1}/{total_images}",
                        "Load%": f"{load_percent:.1f}",
                        "MR%": f"{mr_percent:.1f}",
                    })

            except Exception as e:

                executor.shutdown(wait=False)

                print(f"\nError processing sample {idx}")
                print(type(e).__name__, e)

                raise

        # ---------------------------------------
        # Collect completed jobs
        # ---------------------------------------
        for future in pending:

            df = future.result()

            if df is not None:
                all_results.append(df)

        loader.shutdown(wait=True)
        executor.shutdown(wait=True)

        # -------------------------------------------------------
        # Nothing generated
        # -------------------------------------------------------
        if not all_results:
            return pd.DataFrame()

        # -------------------------------------------------------
        # Merge results once (much faster than repeated concat)
        # -------------------------------------------------------
        result_df = pd.concat(
            all_results,
            ignore_index=True,
        )

        total_time = time.time() - start_time

        print("\n========================================")
        print("Dataset processing completed")
        print("========================================")
        print(f"Images processed : {total_images}")
        print(f"Total time       : {total_time:.2f} sec")
        print(f"Average IPS      : {total_images/total_time:.2f}")
        print(f"Image loading    : {load_time:.2f} sec")
        print(f"MR processing    : {process_time:.2f} sec")
        print("========================================")

        return result_df

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
        labels=None,
        reuse_existing=True,
    ):

        baseline = BaselineEvaluator(output_dir)

        # --------------------------------------------------
        # Reuse existing predictions if available
        # --------------------------------------------------
        if reuse_existing and baseline.baseline_exists():

            cached = baseline.load_predictions()

            if len(cached) == len(dataset):
                print("\nUsing cached baseline predictions...")
                return cached

            print("\nCached predictions do not match dataset size.")
            print("Regenerating baseline predictions...")

        # --------------------------------------------------
        # Generate baseline predictions
        # --------------------------------------------------
        baseline.save_dataset_info(dataset)

        predictions = []

        y_true = []
        y_pred = []

        import time
        from tqdm import tqdm

        start = time.time()

        iterator = tqdm(
            enumerate(dataset),
            total=len(dataset),
            desc="Generating baseline predictions",
            unit="image",
        )

        for idx, sample in iterator:

            try:

                pred = float(
                    self.model.predict(sample)
                )

                predictions.append({
                    "sample_id": idx,
                    "prediction": pred
                })

                if labels is not None:

                    y_true.append(float(labels[idx]))
                    y_pred.append(pred)

                elapsed = time.time() - start

                ips = (idx + 1) / elapsed if elapsed > 0 else 0

                iterator.set_postfix({
                    "IPS": f"{ips:.2f}",
                    "Done": f"{idx + 1}/{len(dataset)}"
                })

            except Exception as e:
                print(f"\nBaseline failed: {idx}")
                print(type(e).__name__, e)
                raise

        # --------------------------------------------------
        # Save predictions
        # --------------------------------------------------
        baseline.save_predictions(predictions)

        prediction_values = [
            p["prediction"]
            for p in predictions
        ]

        baseline.save_basic_metrics(
            prediction_values
        )

        if labels is not None:

            mse = np.mean(
                (
                    np.array(y_true)
                    - np.array(y_pred)
                ) ** 2
            )

            mae = np.mean(
                np.abs(
                    np.array(y_true)
                    - np.array(y_pred)
                )
            )

            rmse = np.sqrt(mse)

            baseline.save_metrics({
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse)
            })

        return prediction_values

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

        baseline_predictions = self.save_baseline(
            dataset=dataset,
            output_dir=output_dir,
            reuse_existing=True,
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
                output_dir=output_dir,
            )

            summary = EpsilonSummary()

            summary_df, report = summary.summarize(dfs)

            summary.print_report(report)

            df = pd.concat(dfs, ignore_index=True)

            results = self.analyze(df)

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