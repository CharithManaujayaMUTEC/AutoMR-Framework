
import numpy as np

class RangeTester:

    def generate_range(self, start, end, num_samples):
        return np.linspace(start, end, num_samples)

    def run_range(self, model, image, transform_fn, relation, start, end, num_samples):

        values = self.generate_range(start, end, num_samples)
        results = []

        original = model.predict(image)

        for v in values:
            transformed = transform_fn(image, v)
            output = model.predict(transformed)

            diff = output - original
            pct = (diff / (abs(original) + 1e-6)) * 100

            results.append({
                "mr": relation.__class__.__name__,
                "param": float(v),
                "original": float(original),
                "transformed": float(output),
                "difference": float(diff),
                "percent_change": float(pct),
                "passed": bool(relation.check(original, output))
            })

        return results
