import numpy as np

class RangeTester:

    def generate_range(self, start, end, num_samples):
        return np.linspace(start, end, num_samples)

    def run_range(self, model, input_data, transform_fn, relation, start, end, num_samples, comparator):

        values = self.generate_range(start, end, num_samples)
        results = []

        #  only compute original for NON-temporal
        if not isinstance(input_data, list):
            original = model.predict(input_data)

        for v in values:

            #  TEMPORAL CASE
            if isinstance(input_data, list):

                pair = transform_fn(input_data, int(v))

                if pair is None:
                    continue

                x1, x2 = pair

                if x1 is None or x2 is None:
                    continue

                y1 = model.predict(x1)
                y2 = model.predict(x2)

                diff = abs(y1 - y2)

                results.append({
                    "mr": relation.__class__.__name__,
                    "param": float(v),
                    "original": float(y1),
                    "transformed": float(y2),
                    "difference": float(diff),
                    "percent_change": 0.0,
                    "passed": bool(relation.check(y1, y2))
                })

                continue

            # IMAGE FLOW (UNCHANGED)
            transformed = transform_fn(input_data, v)
            output = model.predict(transformed)

            #  Behavioral override
            if "Monotonic" in relation.__class__.__name__ or "LessSensitive" in relation.__class__.__name__:
                diff = output - original
                passed = relation.check(original, output)

            elif comparator:
                diff, passed = comparator.compare(original, output)

            else:
                diff = output - original
                passed = relation.check(original, output)

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