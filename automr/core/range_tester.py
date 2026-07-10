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
        original_prediction=None,   # Cached original prediction
    ):

        values = self.generate_range(
            start,
            end,
            num_samples
        )

        results = []

        is_temporal = (
            hasattr(relation, "type")
            and relation.type() == "temporal"
        )

        # ==================================================
        # TEMPORAL MR
        # ==================================================
        if is_temporal:

            for v in values:

                pair = transform_fn(
                    input_data,
                    int(v)
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
                    y2
                )

                results.append({
                    "mr": relation.__class__.__name__,
                    "param": float(v),
                    "original": float(y1),
                    "transformed": float(y2),
                    "difference": float(diff),
                    "percent_change": 0.0,
                    "passed": bool(passed)
                })

            return results

        # ==================================================
        # NORMAL MRs
        # ==================================================

        # -----------------------------------------
        # Reuse cached original prediction if given
        # Otherwise compute it once.
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

        for v in values:

            transformed = transform_fn(
                input_data,
                v
            )

            transformed_images.append(
                transformed
            )

            # Save verification image
            if image_saver is not None:

                try:

                    image_saver.save(
                        mr_name=relation.__class__.__name__,
                        param=float(v),
                        original=input_data,
                        transformed=transformed
                    )

                except Exception:
                    pass

        # ==================================================
        # BATCH PREDICTION
        # ==================================================
        outputs = model.predict_batch(
            transformed_images
        )

        outputs = [
            float(o)
            for o in outputs
        ]

        if len(outputs) == 0:
            return results

        min_output = min(outputs)
        max_output = max(outputs)

        range_change = (
            max_output - min_output
        )

        range_percent_change = (
            abs(range_change)
            /
            (abs(original) + 1e-6)
        ) * 100

        range_passed = (
            range_percent_change
            <= range_threshold
        )

        # ==================================================
        # ANALYZE RESULTS
        # ==================================================
        for v, output in zip(values, outputs):

            output = float(output)

            severity_weight = (
                1
                +
                (
                    abs(v)
                    /
                    (abs(end) + 1e-6)
                )
            )

            base_pass = relation.check(
                original,
                output
            )

            tolerance = getattr(
                relation,
                "tolerance",
                getattr(
                    relation,
                    "epsilon",
                    0.01
                )
            )

            if severity_weight > 1.5:

                passed = (
                    base_pass
                    and
                    abs(output - original)
                    < (0.5 * tolerance)
                )

            else:

                passed = base_pass

            # Hard fail
            if abs(
                output - original
            ) > 0.01:

                passed = False

            if comparator:

                diff, _ = comparator.compare(
                    original,
                    output
                )

            else:

                diff = (
                    output - original
                )

            pct = (
                diff
                /
                (abs(original) + 1e-6)
            ) * 100

            # Range threshold fail
            if abs(pct) > range_threshold:
                passed = False

            results.append({

                "mr": relation.__class__.__name__,
                "param": float(v),
                "original": float(original),
                "transformed": float(output),
                "difference": float(diff),
                "percent_change": float(pct),
                "passed": bool(passed),

                "range_change": float(
                    range_change
                ),

                "range_percent_change": float(
                    range_percent_change
                ),

                "range_passed": bool(
                    range_passed
                )
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