"""
Range testing module.

This module evaluates metamorphic relations across a range of
transformation parameter values. It supports both CPU and GPU
execution, prediction caching, batched inference, and temporal
metamorphic relations.
"""

import numpy as np
import time
import torch


class RangeTester:
    """
    Executes range-based metamorphic testing for a single relation.
    """

    def generate_range(self, start, end, num_samples):
        """
        Generate evenly spaced parameter values for range testing.
        """
        return np.linspace(start, end, num_samples)

    def run_range(
        self,
        model,
        input_data,
        transform_fn,
        relation,
        start,
        end,
        num_samples,
        comparator=None,
        image_saver=None,
        range_threshold=5.0,
        original_prediction=None,
        prediction_cache=None,   # Cache for epsilon runs
    ):
        """
        Optimized range execution.

        Optimizations
        -------------
        - Reuse original prediction
        - Batch transformed inference
        - Cache transformed predictions
        """

        # Generate parameter values for the current transformation.
        values = self.generate_range(
            start,
            end,
            num_samples,
        )

        # Store all test results.
        results = []

        # -----------------------------------------
        # Temporal MR
        # -----------------------------------------
        # Handle temporal relations separately since they
        # operate on pairs of sequential inputs.
        if (
            hasattr(relation, "type")
            and relation.type() == "temporal"
        ):

            for v in values:

                # Generate a pair of temporal inputs.
                pair = transform_fn(
                    input_data,
                    int(v),
                )

                if pair is None:
                    continue

                x1, x2 = pair

                if x1 is None or x2 is None:
                    continue

                # Predict both temporal samples.
                y1 = float(model.predict(x1))
                y2 = float(model.predict(x2))

                # Compute prediction difference.
                diff = abs(y1 - y2)

                # Verify the temporal relation.
                passed = relation.check(
                    y1,
                    y2,
                )

                # Store the result.
                results.append({

                    "mr": relation.__class__.__name__,
                    "param": float(v),
                    "original": y1,
                    "transformed": y2,
                    "difference": diff,
                    "percent_change": 0.0,
                    "passed": bool(passed),

                })

            return results

        # -----------------------------------------
        # Original prediction (reuse if available)
        # -----------------------------------------
        # Avoid repeated inference by reusing the
        # original prediction whenever possible.
        if original_prediction is None:
            original = float(
                model.predict(input_data)
            )
        else:
            original = float(
                original_prediction
            )

        # -----------------------------------------
        # Generate transformations
        # Supports CPU (NumPy) and GPU (Torch)
        # -----------------------------------------

        transformed_images = []
        cache_keys = []

        for v in values:

            # Generate transformed input.
            transformed = transform_fn(
                input_data,
                v,
            )

            transformed_images.append(transformed)

            # Create a cache key for this transformation.
            cache_keys.append(
                (
                    relation.__class__.__name__,
                    float(v),
                )
            )

            # Save preview images only when data is on CPU.
            if (
                image_saver is not None
                and isinstance(transformed, np.ndarray)
            ):
                try:
                    image_saver.save(
                        mr_name=relation.__class__.__name__,
                        param=float(v),
                        original=input_data,
                        transformed=transformed,
                    )
                except Exception:
                    # Ignore preview saving failures.
                    pass
        
        # -----------------------------------------
        # Prediction cache
        # -----------------------------------------
        # Reuse predictions across epsilon runs.
        outputs = [None] * len(values)

        missing = []
        missing_idx = []

        if prediction_cache is not None:

            for i, key in enumerate(cache_keys):

                # Use cached prediction when available.
                if key in prediction_cache:
                    outputs[i] = prediction_cache[key]
                else:
                    missing.append(
                        transformed_images[i]
                    )
                    missing_idx.append(i)

        else:

            # Cache disabled; predict every sample.
            missing = transformed_images
            missing_idx = list(
                range(len(values))
            )

        # -----------------------------------------
        # Batch predict only missing images
        # -----------------------------------------
        if missing:

            # GPU backend
            if isinstance(missing[0], torch.Tensor):

                # Stack tensors into a single batch.
                batch = torch.stack(
                    missing,
                    dim=0,
                )

                preds = model.predict_batch(batch)

            # CPU backend
            else:

                preds = model.predict_batch(
                    missing
                )

            # Store predictions and update cache.
            for idx, pred in zip(
                missing_idx,
                preds,
            ):

                pred = float(pred)

                outputs[idx] = pred

                if prediction_cache is not None:
                    prediction_cache[
                        cache_keys[idx]
                    ] = pred

        # Convert predictions to a NumPy array.
        outputs = np.asarray(
            outputs,
            dtype=np.float32,
        )

        # Nothing to analyze.
        if outputs.size == 0:
            return results

        # Compute range statistics.
        min_output = float(outputs.min())
        max_output = float(outputs.max())

        range_change = (
            max_output - min_output
        )

        range_percent_change = (
            abs(range_change)
            /
            (abs(original) + 1e-6)
        ) * 100.0

        range_passed = (
            range_percent_change
            <= range_threshold
        )

        # -----------------------------------------
        # Analyze predictions
        # -----------------------------------------
        for v, output in zip(values, outputs):

            output = float(output)

            # Retrieve the configured tolerance.
            tolerance = getattr(
                relation,
                "tolerance",
                getattr(relation, "epsilon", 0.01),
            )
            # Final pass/fail decision.
            difference = abs(output - original)

            passed = difference <= tolerance

            pct = (
                difference /
                (abs(original) + 1e-6)
            ) * 100.0

            results.append({

                "mr": relation.__class__.__name__,
                "param": float(v),
                "original": original,
                "transformed": output,
                "difference": float(difference),
                "percent_change": float(pct),
                "passed": passed,
                "range_change": float(range_change),
                "range_percent_change": float(range_percent_change),
                "range_passed": bool(range_passed),

            })

        return results

# Legacy implementation retained below for reference.
# def generate_range(self, start, end, num_samples):
#     return np.linspace(start, end, num_samples)

# def run_range(self, model, input_data, transform_fn, relation,
#               start, end, num_samples, comparator):

#     FIX: everything below must be indented
"""         values = self.generate_range(start, end, num_samples)
        results = []

        is_temporal = hasattr(relation, "type") and relation.type() == "temporal"

        if not is_temporal:
            original = model.predict(input_data)

        for v in values:

            # ===== TEMPORAL =====
            if is_temporal:

                pair = transform_fn(input_data, int(v))
                if pair is None:
                    continue

                x1, x2 = pair
                if x1 is None or x2 is None:
                    continue

                y1 = model.predict(x1)
                y2 = model.predict(x2)

                diff = abs(y1 - y2)
                pct = 0.0
                passed = relation.check(y1, y2)

                results.append({
                    "mr": relation.__class__.__name__,
                    "param": float(v),
                    "original": float(y1),
                    "transformed": float(y2),
                    "difference": float(diff),
                    "percent_change": float(pct),
                    "passed": bool(passed)
                })

                continue

            # ===== NORMAL =====
            transformed = transform_fn(input_data, v)
            output = model.predict(transformed)

            # dynamic strictness
            severity_weight = 1 + (abs(v) / (end + 1e-6))

            base_pass = relation.check(original, output)

            if severity_weight > 1.5:
                passed = base_pass and abs(output - original) < (0.5 * getattr(relation, "tolerance", getattr(relation, "epsilon", 0.01)))
            else:
                passed = base_pass

            # HARD FAIL
            if abs(output - original) > 0.01:
                passed = False

            # comparator only logs diff
            if comparator:
                diff, _ = comparator.compare(original, output)
            else:
                diff = output - original

            pct = (diff / (abs(original) + 1e-6)) * 100

            results.append({
                "mr": relation.__class__.__name__,
                "param": float(v),
                "original": float(original),
                "transformed": float(output),
                "difference": float(diff),
                "percent_change": float(pct),
                "passed": bool(passed)
            })

        return results
"""