import numpy as np

class RangeTester:

    def generate_range(self, start, end, num_samples):
        return np.linspace(start, end, num_samples)

    def run_range(self, model, input_data, transform_fn, relation,
                  start, end, num_samples, comparator):

        values = self.generate_range(start, end, num_samples)
        results = []

        relation_type = relation.type() if hasattr(relation, "type") else "equality"

        #  ONLY compute original for non-temporal
        if relation_type != "temporal":
            original = model.predict(input_data)

        for v in values:

            # =========================
            #  TEMPORAL CASE
            # =========================
            if relation_type == "temporal":

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

                continue  # skip normal flow

            # =========================
            #  IMAGE / BEHAVIORAL CASE
            # =========================
            transformed = transform_fn(input_data, v)
            output = model.predict(transformed)

            #  RELATION-TYPE BASED LOGIC
            if relation_type == "equality":

                if comparator:
                    diff, passed = comparator.compare(original, output)
                else:
                    diff = output - original
                    passed = relation.check(original, output)

            elif relation_type == "inequality":

                diff = output - original
                passed = relation.check(original, output)

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