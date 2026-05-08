
import numpy as np

class RangeTester:

    def run_range(self, model, image, transform_fn, relation, values):

        results = []

        original = model.predict(image)

        for v in values:
            transformed = transform_fn(image, v)
            output = model.predict(transformed)

            diff = output - original
            pct = (diff / (abs(original) + 1e-6)) * 100

            results.append({
                "param": v,
                "original": float(original),
                "transformed": float(output),
                "difference": float(diff),
                "percent_change": float(pct),
                "passed": bool(relation.check(original, output))
            })

        return results
