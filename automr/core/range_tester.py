import numpy as np

class RangeTester:

    def generate_range(self, start, end, num_samples):
        return np.linspace(start, end, num_samples)

    def run_range(self, model, input_data, transform_fn, relation,
                  start, end, num_samples, comparator):

        # ✅ FIX: everything below must be indented
        values = self.generate_range(start, end, num_samples)
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

            # 🔥 MR decides
            passed = relation.check(original, output)

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