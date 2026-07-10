import numpy as np
class RangeTester:

    def generate_range(self, start, end, num_samples):
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

        values = self.generate_range(
            start,
            end,
            num_samples,
        )

        results = []

        # -----------------------------------------
        # Temporal MR
        # -----------------------------------------
        if (
            hasattr(relation, "type")
            and relation.type() == "temporal"
        ):

            for v in values:

                pair = transform_fn(
                    input_data,
                    int(v),
                )

                if pair is None:
                    continue

                x1, x2 = pair

                if x1 is None or x2 is None:
                    continue

                y1 = float(model.predict(x1))
                y2 = float(model.predict(x2))

                diff = abs(y1 - y2)

                passed = relation.check(
                    y1,
                    y2,
                )

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
        if original_prediction is None:
            original = float(
                model.predict(input_data)
            )
        else:
            original = float(
                original_prediction
            )

        transformed_images = []
        cache_keys = []

        # -----------------------------------------
        # Generate transformations
        # -----------------------------------------
        for v in values:

            transformed = transform_fn(
                input_data,
                v,
            )

            transformed_images.append(
                transformed
            )

            cache_keys.append(
                (
                    relation.__class__.__name__,
                    float(v),
                )
            )

            if image_saver is not None:

                try:

                    image_saver.save(
                        mr_name=relation.__class__.__name__,
                        param=float(v),
                        original=input_data,
                        transformed=transformed,
                    )

                except Exception:
                    pass

        # -----------------------------------------
        # Prediction cache
        # -----------------------------------------
        outputs = [None] * len(values)

        missing = []
        missing_idx = []

        if prediction_cache is not None:

            for i, key in enumerate(cache_keys):

                if key in prediction_cache:
                    outputs[i] = prediction_cache[key]
                else:
                    missing.append(
                        transformed_images[i]
                    )
                    missing_idx.append(i)

        else:

            missing = transformed_images
            missing_idx = list(
                range(len(values))
            )

        # -----------------------------------------
        # Batch predict only missing images
        # -----------------------------------------
        if missing:

            preds = model.predict_batch(
                missing
            )

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

        outputs = np.asarray(
            outputs,
            dtype=np.float32,
        )

        if outputs.size == 0:
            return results

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
        for v, output in zip(
            values,
            outputs,
        ):

            output = float(output)

            base_pass = relation.check(
                original,
                output,
            )

            tolerance = getattr(
                relation,
                "tolerance",
                getattr(
                    relation,
                    "epsilon",
                    0.01,
                ),
            )

            passed = (
                base_pass
                and
                abs(output - original)
                <= tolerance
            )

            if comparator:
                diff, _ = comparator.compare(
                    original,
                    output,
                )
            else:
                diff = output - original

            pct = (
                diff
                /
                (abs(original) + 1e-6)
            ) * 100.0

            results.append({

                "mr": relation.__class__.__name__,
                "param": float(v),
                "original": original,
                "transformed": output,
                "difference": float(diff),
                "percent_change": float(pct),
                "passed": bool(passed),
                "range_change": float(range_change),
                "range_percent_change": float(range_percent_change),
                "range_passed": bool(range_passed),

            })

        return results
  
#    def generate_range(self, start, end, num_samples):
#        return np.linspace(start, end, num_samples)

#    def run_range(self, model, input_data, transform_fn, relation,
#                  start, end, num_samples, comparator):

        #  FIX: everything below must be indented
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

            #  dynamic strictness
            severity_weight = 1 + (abs(v) / (end + 1e-6))

            base_pass = relation.check(original, output)

            if severity_weight > 1.5:
                passed = base_pass and abs(output - original) < (0.5 * getattr(relation, "tolerance", getattr(relation, "epsilon", 0.01)))
            else:
                passed = base_pass

            #  FIX 3: HARD FAIL (ADD THIS BLOCK HERE)
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

        return results """