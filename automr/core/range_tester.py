import numpy as np

class RangeTester:

    def generate_range(self, start, end, num_samples):
        return np.linspace(start, end, num_samples)

    def run_range(self, model, input_data, transform_fn, relation, start, end, num_samples, comparator):

        values = self.generate_range(start, end, num_samples)
        results = []

        original = model.predict(input_data)

        for v in values:
            transformed_input = transform_fn(input_data, v)
            output = model.predict(transformed_input)

            diff, passed = comparator.compare(original, output)

            results.append({
                "mr": relation.__class__.__name__,
                "param": float(v),
                "original": original,
                "transformed": output,
                "difference": diff,
                "passed": passed,
                "expected_behavior": relation.expected(),
                "actual_behavior": "Consistent" if passed else "Violation"
            })

        return results