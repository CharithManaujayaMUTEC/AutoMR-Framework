import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import os
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor

from automr.core.range_tester import RangeTester
from automr.analysis import Analyzer
from automr.models import get_wrapper
from automr.comparators import get_comparator
from automr.input_handlers import get_handler
from automr.banner import print_banner

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
from automr.evaluation import GraphGenerator
from automr.evaluation import DecoderHealthAnalyzer
from automr.evaluation import FinalEvaluationReport
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
    """
    Main API of the AutoMR framework.

    AutoMR provides an end-to-end interface for automated
    metamorphic testing of machine learning models. It
    manages model wrapping, transformation registration,
    metamorphic relation execution, dataset testing,
    epsilon sensitivity analysis, result analysis,
    logging, and persistence.
    """

    def __init__(
        self,
        model,
        task="regression",
        input_type="image",
        epsilon=0.05,
        range_threshold=5.0,
        transform_ranges=None,
    ):

        # --------------------------------------------------
        # Core Components
        # --------------------------------------------------
        # Initialize all major framework components used
        # throughout the testing workflow.

        # Display startup banner once
        print_banner()

        self.image_saver = TransformationSaver()
        self.logger = AutoMRLogger()
        self.input_handler = get_handler(input_type)
        self.model = get_wrapper(model)
        self.range_tester = RangeTester()
        self.analyzer = Analyzer()
        self.graph_generator = GraphGenerator()

        # Store framework configuration.
        self.task = task
        self.range_threshold = range_threshold

        # Optional user-defined transformation configuration
        self.transform_ranges = transform_ranges

        # --------------------------------------------------
        # Registries
        # --------------------------------------------------
        # Maintain collections of available transformations
        # and metamorphic relations.

        self.transform_registry = TransformationRegistry()
        self.relation_registry = RelationRegistry()

        # Parameter search ranges for every MR.
        self.mr_ranges = {}

        # Initialize the comparator used to determine
        # whether transformed outputs satisfy the MR.
        self.comparator = get_comparator(
            task=task,
            epsilon=epsilon,
        )

        # Register all built-in metamorphic relations.
        self._register_default_mrs(epsilon)

    def _register_default_mrs(self, epsilon):
        """
        Register the built-in transformations,
        corresponding metamorphic relations, and
        their default parameter ranges.
        """

        # Register built-in transformations.
        register_default_transforms(
            self.transform_registry
        )

        # Register built-in metamorphic relations.
        register_default_relations(
            self.relation_registry,
            epsilon
        )

        # Default parameter ranges explored for
        # each supported metamorphic relation.
        self.mr_ranges = {

            "brightness": {
                "start": 0.1,
                "end": 3.0,
                "samples": 5,
            },

            "rotation": {
                "start": -60,
                "end": 60,
                "samples": 5,
            },

            "translation": {
                "start": 0,
                "end": 80,
                "samples": 5,
            },

            "noise": {
                "start": 0,
                "end": 150,
                "samples": 5,
            },

            "blur": {
                "start": 1,
                "end": 31,
                "samples": 5,
            },

            "contrast": {
                "start": 0.1,
                "end": 4.0,
                "samples": 5,
            },

            "composite": {
                "start": 0.1,
                "end": 1.5,
                "samples": 5,
            },

            "global_brightness": {
                "start": 0.1,
                "end": 3.0,
                "samples": 5,
            },

            "global_contrast": {
                "start": 0.1,
                "end": 4.0,
                "samples": 5,
            },

            "global_blur": {
                "start": 1,
                "end": 31,
                "samples": 5,
            },

            "global_noise": {
                "start": 0,
                "end": 150,
                "samples": 5,
            },

            "global_rotation": {
                "start": -60,
                "end": 60,
                "samples": 5,
            },

            "global_translation": {
                "start": 0,
                "end": 80,
                "samples": 5,
            },

            "rain": {
                "start": 0.0,
                "end": 1.5,
                "samples": 5,
            },

            "snow": {
                "start": 0.0,
                "end": 1.5,
                "samples": 5,
            },

            "fog": {
                "start": 0.0,
                "end": 1.5,
                "samples": 5,
            },

            "sandstorm": {
                "start": 0.0,
                "end": 1.5,
                "samples": 5,
            },

            "dust": {
                "start": 0.0,
                "end": 1.5,
                "samples": 5,
            },

            "haze": {
                "start": 0.0,
                "end": 1.5,
                "samples": 5,
            },

            "smoke": {
                "start": 0.0,
                "end": 1.5,
                "samples": 5,
            },

            "visibility": {
                "start": 0.05,
                "end": 1.5,
                "samples": 5,
            },

            "darkness": {
                "start": 0.05,
                "end": 1.5,
                "samples": 5,
            },

            "temporal": {
                "start": 0,
                "end": 150,
                "samples": 5,
            },
        }

        # --------------------------------------------------
        # Override defaults with user configuration
        # --------------------------------------------------

        if self.transform_ranges:

            for mr_name, config in self.transform_ranges.items():

                if mr_name not in self.mr_ranges:
                    continue

                self.mr_ranges[mr_name].update(config)

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
        Register a custom transformation together with its
        corresponding metamorphic relation and parameter range.

        Once registered, the new MR becomes available to the
        complete AutoMR testing pipeline.
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

    # ==================================================
    # Custom Extension API
    # ==================================================

    def register_custom_transformation(
        self,
        name,
        transform,
    ):
        """
        Register a custom transformation.

        This is an extension API and does not modify or
        replace the existing register_transform() behavior.

        Parameters
        ----------
        name : str
            Unique transformation name.

        transform : callable
            Transformation function or transformation object.

        Returns
        -------
        self
            Enables method chaining.
        """

        self.transformation_registry.register(
            name,
            transform,
        )

        return self


    def register_custom_relation(
        self,
        name,
        relation,
    ):
        """
        Register a custom metamorphic relation.

        This is an extension API and does not modify or
        replace existing relation registration behavior.

        Parameters
        ----------
        name : str
            Unique relation name.

        relation : callable
            Relation function or relation object.

        Returns
        -------
        self
            Enables method chaining.
        """

        self.relation_registry.register(
            name,
            relation,
        )

        return self


    def register_custom_mr(
        self,
        name,
        transform,
        relation,
        param_range=None,
    ):
        """
        Register a complete custom metamorphic relation.

        Registers the transformation and relation using the
        existing AutoMR registration mechanism.

        Existing APIs remain unchanged.

        Parameters
        ----------
        name : str
            Base name of the custom MR.

        transform : callable
            Input transformation.

        relation : callable
            Metamorphic relation.

        param_range : tuple, optional
            Optional parameter range.

        Returns
        -------
        self
            Enables method chaining.
        """

        transform_name = f"{name}_transform"

        relation_name = f"{name}_relation"

        self.register_custom_transformation(
            name=transform_name,
            transform=transform,
        )

        self.register_custom_relation(
            name=relation_name,
            relation=relation,
        )

        # Preserve the existing registration path when
        # a parameter range is supplied.
        if param_range is not None:

            self.register_transform(
                name=name,
                transform=transform,
                relation=relation,
                param_range=param_range,
            )

        return self
    
    def unregister_transform(self, name):
        """
        Remove a previously registered transformation,
        relation and its associated parameter range.
        """

        if name in self.transform_registry.transforms:
            del self.transform_registry.transforms[name]

        if name in self.relation_registry.relations:
            del self.relation_registry.relations[name]

        if name in self.mr_ranges:
            del self.mr_ranges[name]


    def has_transform(self, name):
        """
        Check whether a transformation is currently
        registered within AutoMR.
        """

        return name in self.transform_registry.transforms


    def get_transform(self, name):
        """
        Retrieve a registered transformation by name.
        """

        return self.transform_registry.get(name)


    def get_relation(self, name):
        """
        Retrieve the metamorphic relation associated
        with a registered transformation.
        """

        return self.relation_registry.get(name)


    def list_transforms(self):
        """
        Return all registered transformation names
        in alphabetical order.
        """

        return sorted(self.transform_registry.list())


    def list_relations(self):
        """
        Return all registered relation names in
        alphabetical order.
        """

        return sorted(self.relation_registry.list())


    # ---------- EXPECTED ----------

    def get_expected(self, relation_name):
        """
        Retrieve the human-readable expected behavior
        defined by a registered metamorphic relation.
        """

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
    ):
        """
        Execute a single metamorphic relation over one input.

        The original prediction is computed only once and
        reused across all sampled transformation parameters
        for improved execution efficiency.
        """

        if not self.has_transform(mr_name):
            raise ValueError(
                f"Unknown metamorphic relation: '{mr_name}'"
            )

        # Retrieve the transformation and its relation.
        transform = self.get_transform(mr_name)
        relation = self.get_relation(mr_name)

        # Retrieve the parameter search range.
        cfg = self.mr_ranges[mr_name]

        start = cfg["start"]
        end = cfg["end"]

        # Use MR-specific sample count unless caller overrides it
        samples = cfg.get("samples", samples)

        # Execute parameter range testing.
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
        )

        # Record severity and write execution logs.
        for result in results:
            
            epsilon = max(
                getattr(
                    relation,
                    "tolerance",
                    getattr(relation, "epsilon", 0.01),
                ),
                1e-12,
            )

            result["severity"] = (
                abs(result["difference"]) / epsilon
            )

            self.logger.log(
                f"MR={mr_name} "
                f"param={result['param']} "
                f"orig={result['original']} "
                f"trans={result['transformed']} "
                f"diff={result['difference']} "
                f"pass={result['passed']}"
            )

        # Convert raw results into a DataFrame.
        df = self.analyzer.to_dataframe(results)

        # Generate summary statistics.
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
        Execute every registered metamorphic relation for a
        single input sample and combine the results into one
        DataFrame.
        """

        if exclude is None:
            exclude = ["temporal"]

        # Compute the original prediction only once.
        if original_prediction is None:
            original_prediction = float(
                self.model.predict(input_data)
            )

        dfs = []

        # Execute each registered MR.
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

        # Merge every MR result into a single DataFrame.
        return pd.concat(
            dfs,
            ignore_index=True,
        )
    
    def _process_single_sample(self, args):
        """
        Execute the complete AutoMR workflow for a single
        dataset sample.

        The original prediction is generated once and reused
        across all metamorphic relations to minimize repeated
        model inference.
        """

        sample_id, sample, samples_per_mr, df_temp = args

        # Compute the original model prediction once.
        original_prediction = float(
            self.model.predict(sample)
        )

        # Execute all image-based metamorphic relations.
        df_img = self.run_all_mrs(
            input_data=sample,
            samples=samples_per_mr,
            original_prediction=original_prediction,
        )

        # Merge temporal MR results if available.
        if df_temp is not None and not df_temp.empty:

            df = pd.concat(
                [df_img, df_temp],
                ignore_index=True,
                copy=False,
            )

        else:
            df = df_img

        # Record the dataset sample identifier.
        df["sample_id"] = sample_id

        # Store the expected behavior for each relation.
        df["expected_behavior"] = [
            self.get_expected(mr)
            for mr in df["mr"]
        ]

        # Convert the pass/fail outcome into a readable label.
        df["actual_behavior"] = np.where(
            df["status"] == "PASS",
            "Consistent",
            "Violation",
        )

        return df


    # --------------------------------------------------
    # Epsilon Configuration
    # --------------------------------------------------

    def set_epsilon(self, epsilon):
        """
        Update the framework tolerance.

        The supplied epsilon value is propagated to every
        registered metamorphic relation and to the output
        comparator so subsequent executions use the same
        tolerance.
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
    ):
        """
        Execute AutoMR over an entire dataset.

        Features
        --------
        - Optional dataset size limiting.
        - Automatic epsilon update.
        - Shared prediction cache.
        - Parallel image loading.
        - Parallel sample execution.
        - Optional temporal MR execution.
        - Progress monitoring with timing statistics.
        """

        import time
        from concurrent.futures import ThreadPoolExecutor

        # -------------------------------------------------------
        # Update epsilon if requested.
        # -------------------------------------------------------
        if epsilon is not None:
            self.set_epsilon(epsilon)

        # -------------------------------------------------------
        # Determine the number of samples to process.
        # -------------------------------------------------------
        if max_samples is not None:
            total_images = min(max_samples, len(dataset))
        else:
            total_images = len(dataset)

        # -------------------------------------------------------
        # Execute the temporal MR once for the dataset.
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

            except Exception as e:

                print(f"Temporal MR skipped: {e}")
                df_temp = None

        # -------------------------------------------------------
        # Initialize timing statistics.
        # -------------------------------------------------------
        start_time = time.time()

        load_time = 0.0
        process_time = 0.0

        all_results = []

        # -------------------------------------------------------
        # Create dataset iterator.
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

        # -------------------------------------------------------
        # Background image loader.
        # -------------------------------------------------------
        loader = ThreadPoolExecutor(max_workers=2)

        future = loader.submit(
            dataset.__getitem__,
            0
        )

        # -------------------------------------------------------
        # Parallel execution pool.
        # -------------------------------------------------------
        workers = max(
            1,
            min(
                multiprocessing.cpu_count(),
                8,
            ),
        )

        executor = ThreadPoolExecutor(
            max_workers=workers
        )

        pending = []

        # -------------------------------------------------------
        # Process every dataset sample.
        # -------------------------------------------------------
        for idx in iterator:

            try:

                # Measure image loading time.
                t0 = time.perf_counter()

                sample = future.result()

                load_time += (
                    time.perf_counter() - t0
                )

                # Begin loading the next sample while the
                # current one is being processed.
                if idx + 1 < total_images:

                    future = executor.submit(
                        dataset.__getitem__,
                        idx + 1,
                    )

                # Measure scheduling overhead.
                t0 = time.perf_counter()

                future_df = executor.submit(
                    self._process_single_sample,
                    (
                        idx,
                        sample,
                        samples_per_mr,
                        df_temp,
                    ),
                )

                pending.append(future_df)

                process_time += (
                    time.perf_counter() - t0
                )

                # Update the progress display.
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

        # -------------------------------------------------------
        # Collect all completed sample executions.
        # -------------------------------------------------------
        for future in pending:

            df = future.result()

            if df is not None:
                all_results.append(df)

        # Cleanly shut down background workers.
        loader.shutdown(wait=True)
        executor.shutdown(wait=True)

        # -------------------------------------------------------
        # Return an empty DataFrame if no results were produced.
        # -------------------------------------------------------
        if not all_results:
            return pd.DataFrame()

        # -------------------------------------------------------
        # Merge all sample results into a single DataFrame.
        # -------------------------------------------------------
        result_df = pd.concat(
            all_results,
            ignore_index=True,
        )

        total_time = time.time() - start_time

        # -------------------------------------------------------
        # Display execution statistics.
        # -------------------------------------------------------
        print("\n========================================")
        print("Dataset processing completed")
        print("========================================")
        print(f"Images processed : {total_images}")
        print(f"Total time       : {total_time:.2f} sec")
        print(f"Average IPS      : {total_images / total_time:.2f}")
        print(f"Image loading    : {load_time:.2f} sec")
        print(f"MR processing    : {process_time:.2f} sec")
        print("========================================")

        return result_df


    # ==================================================
    # Analysis
    # ==================================================

    def analyze(self, df):
        """
        Perform post-processing analysis on AutoMR results.

        Generates failure statistics, severity summaries,
        parameter range analysis, prediction traces, and
        identifies the most significant failures.
        """

        from automr.core.failure_analysis import FailureAnalyzer

        analyzer = FailureAnalyzer()

        return {
            "failure_summary": analyzer.failure_rate_per_mr(df),
            "severity_summary": analyzer.severity_per_mr(df),
            "worst_cases": analyzer.worst_cases(df, top_k=10),
            "regions": analyzer.failure_regions(df),
            "range_summary": analyzer.range_summary(df),
            "range_analysis": analyzer.range_analysis(df),
            "prediction_trace": self.analyzer.prediction_trace(df),
        }


    # ==================================================
    # Save Results
    # ==================================================

    def save_results(self, df, results, output_dir="results"):
        """
        Persist all generated AutoMR outputs to disk.

        Depending on the executed workflow, this may include
        epsilon sensitivity reports in addition to the standard
        failure analysis outputs.
        """

        os.makedirs(output_dir, exist_ok=True)

        self.graph_generator.output_dir = Path(output_dir)
        self.graph_generator.graph_dir = Path(output_dir) / "graphs"
        self.graph_generator.graph_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.graph_generator.worst_cases_graph(
            results["worst_cases"]
        )

        # Main AutoMR output.
        df.to_csv(
            f"{output_dir}/automr_results.csv",
            index=False,
        )

        # Analysis summaries.
        results["failure_summary"].to_csv(
            f"{output_dir}/failure_summary.csv",
            index=False,
        )

        results["severity_summary"].to_csv(
            f"{output_dir}/severity_summary.csv"
        )

        results["worst_cases"].to_csv(
            f"{output_dir}/worst_cases.csv",
            index=False,
        )

        results["range_summary"].to_csv(
            f"{output_dir}/range_summary.csv",
            index=False,
        )

        results["range_analysis"].to_csv(
            f"{output_dir}/range_analysis.csv",
            index=False,
        )

        results["prediction_trace"].to_csv(
            f"{output_dir}/prediction_trace.csv",
            index=False,
        )

        # -----------------------------------------
        # Generate Per-MR Graphs
        # -----------------------------------------

        self.graph_generator.generate_all(
            df,
            epsilon=self.comparator.epsilon
        )

        # Failure Rate

        self.graph_generator.failure_rate_graph(
            results["failure_summary"]
        )

        # Severity

        self.graph_generator.severity_graph(
            results["severity_summary"]
        )

        # Range Analysis

        self.graph_generator.range_analysis_graph(
            results["range_analysis"]
        )

        # Save epsilon sensitivity outputs when available.
        if "epsilon_summary" in results:

            self.graph_generator.epsilon_curve(
                results["epsilon_summary"]
            )

            results["epsilon_summary"].to_csv(
                f"{output_dir}/epsilon_summary.csv",
                index=False,
            )

        if "epsilon_report" in results:

            with open(
                f"{output_dir}/epsilon_report.txt",
                "w",
            ) as f:

                for k, v in results["epsilon_report"].items():
                    f.write(f"{k}: {v}\n")

        # Save detected failure regions.
        with open(
            f"{output_dir}/failure_regions.txt",
            "w",
        ) as f:

            for k, v in results["regions"].items():
                f.write(f"{k}: {v}\n")

    def save_baseline(
        self,
        dataset,
        output_dir="results",
        labels=None,
        reuse_existing=True,
    ):
        """
        Generate and store baseline model predictions.

        Existing cached predictions can be reused when they
        match the supplied dataset size. Basic prediction
        statistics and optional regression metrics are also
        saved for later comparison.
        """

        baseline = BaselineEvaluator(output_dir)

        # --------------------------------------------------
        # Reuse cached predictions whenever possible.
        # --------------------------------------------------
        if reuse_existing and baseline.baseline_exists():

            cached = baseline.load_predictions()

            if len(cached) == len(dataset):

                print("\nUsing cached baseline predictions...")
                return cached

            print("\nCached predictions do not match dataset size.")
            print("Regenerating baseline predictions...")

        # --------------------------------------------------
        # Store dataset information.
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

        # --------------------------------------------------
        # Generate predictions for every dataset sample.
        # --------------------------------------------------
        for idx, sample in iterator:

            try:

                pred = float(
                    self.model.predict(sample)
                )

                predictions.append({
                    "sample_id": idx,
                    "prediction": pred
                })

                # Collect labels if regression metrics
                # should be computed.
                if labels is not None:

                    y_true.append(float(labels[idx]))
                    y_pred.append(pred)

                elapsed = time.time() - start

                ips = (
                    (idx + 1) / elapsed
                    if elapsed > 0
                    else 0
                )

                iterator.set_postfix({
                    "IPS": f"{ips:.2f}",
                    "Done": f"{idx + 1}/{len(dataset)}"
                })

            except Exception as e:

                print(f"\nBaseline failed: {idx}")
                print(type(e).__name__, e)
                raise

        # --------------------------------------------------
        # Save generated predictions.
        # --------------------------------------------------
        baseline.save_predictions(predictions)

        prediction_values = [
            p["prediction"]
            for p in predictions
        ]

        # Save basic prediction statistics.
        baseline.save_basic_metrics(
            prediction_values
        )

        # --------------------------------------------------
        # Compute regression metrics if labels exist.
        # --------------------------------------------------
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
        """
        Save a textual summary of the wrapped model.

        TensorFlow/Keras models provide a detailed summary,
        while unsupported models fall back to storing their
        wrapper type.
        """

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

    # ==================================================
    # Full AutoMR Pipeline
    # ==================================================

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
        """
        Execute the complete AutoMR evaluation pipeline.

        Pipeline
        --------
        1. Save model information.
        2. Generate or reuse baseline predictions.
        3. Run either standard AutoMR or epsilon sensitivity.
        4. Analyze generated results.
        5. Save reports.
        6. Optionally print summaries.

        Returns
        -------
        tuple
            (results_dataframe, analysis_dictionary)
        """

        # --------------------------------------------------
        # Save model information.
        # --------------------------------------------------
        self.save_model_summary(
            output_dir
        )

        # --------------------------------------------------
        # Generate or reuse baseline predictions.
        # --------------------------------------------------
        baseline_predictions = self.save_baseline(
            dataset=dataset,
            output_dir=output_dir,
            reuse_existing=True,
        )

        # --------------------------------------------------
        # Decoder health validation.
        #
        # This is warning-only and never blocks
        # metamorphic testing.
        # --------------------------------------------------
        decoder_health = DecoderHealthAnalyzer()

        decoder_health_report = decoder_health.analyze(
            predictions=baseline_predictions,
            output_path=str(
                Path(output_dir)
                / "decoder_health.json"
            ),
        )

        if verbose:

            print(
                "\n=== Decoder Health ==="
            )

            print(
                f"Status: "
                f"{decoder_health_report['status']}"
            )

            print(
                f"Distribution: "
                f"{decoder_health_report['distribution_diagnostic']}"
            )

            if decoder_health_report["warnings"]:

                print("\nWarnings:")

                for warning in (
                    decoder_health_report["warnings"]
                ):

                    print(
                        f"- {warning}"
                    )

        # --------------------------------------------------
        # Standard AutoMR execution.
        # --------------------------------------------------
        if epsilon_min is None:

            df = self.run_dataset(
                dataset,
                max_samples=max_samples,
                samples_per_mr=samples_per_mr,
                show_progress=show_progress,
            )

            results = self.analyze(df)

        # --------------------------------------------------
        # Epsilon sensitivity analysis.
        # --------------------------------------------------
        else:

            if epsilon_max is None:
                raise ValueError(
                    "epsilon_max must be provided."
                )

            if epsilon_count is None:
                raise ValueError(
                    "epsilon_count must be provided."
                )

            if epsilon_min >= epsilon_max:
                raise ValueError(
                    "epsilon_min must be smaller than epsilon_max."
                )

            if epsilon_count < 2:
                raise ValueError(
                    "epsilon_count must be at least 2."
                )

            # Generate epsilon values.
            epsilon_values = generate_epsilon_values(
                epsilon_min,
                epsilon_max,
                epsilon_count,
            )

            # Execute sensitivity analysis.
            sensitivity = EpsilonSensitivity(self)

            dfs = sensitivity.run(
                dataset,
                epsilon_values,
                max_samples=max_samples,
                samples_per_mr=samples_per_mr,
                show_progress=show_progress,
                output_dir=output_dir,
            )

            # Summarize epsilon results.
            summary = EpsilonSummary()

            summary_df, report = summary.summarize(dfs)

            summary.print_report(report)

            # Merge all epsilon runs.
            df = pd.concat(
                dfs,
                ignore_index=True,
            )

            results = self.analyze(df)

            results["epsilon_summary"] = summary_df
            results["epsilon_report"] = report

        # --------------------------------------------------
        # Attach decoder health diagnostics.
        # --------------------------------------------------
        if results is None:

            results = {}

        results["decoder_health"] = (
            decoder_health_report
        )

        # --------------------------------------------------
        # Save analysis outputs.
        # --------------------------------------------------
        if save and results:

            self.save_results(
                df,
                results,
                output_dir,
            )

        # --------------------------------------------------
        # Print concise summaries.
        # --------------------------------------------------
        if verbose:

            print("\n=== AutoMR Results ===")

            if "failure_summary" in results:
                print(results["failure_summary"])

            print("\n--- Severity ---")

            if "severity_summary" in results:
                print(results["severity_summary"])

        # --------------------------------------------------
        # Generate consolidated final evaluation report.
        # --------------------------------------------------
        final_report_generator = FinalEvaluationReport(
            output_dir=output_dir
        )

        final_report = (
            final_report_generator.generate_and_save(
                test_summary=results.get(
                    "summary"
                )
                if isinstance(results, dict)
                else None,

                epsilon_summary=results.get(
                    "epsilon_summary"
                )
                if isinstance(results, dict)
                else None,
            )
        )

        if isinstance(results, dict):

            results["final_evaluation_report"] = (
                final_report
            )

        return df, results