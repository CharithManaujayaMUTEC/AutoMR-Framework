class MRTester:

    def run(self, model, input_data, transform, relation, comparator):

        original = model.predict(input_data)
        transformed_input = transform(input_data)
        transformed = model.predict(transformed_input)

        diff, passed = comparator.compare(original, transformed)

        return {
            "mr": relation.__class__.__name__,
            "original": original,
            "transformed": transformed,
            "difference": diff,
            "passed": passed,
            "expected_behavior": relation.expected(),
            "actual_behavior": "Consistent" if passed else "Violation"
        }

    def run_all(self, model, input_data, transforms, relations, comparator):

        results = []

        for transform, relation in zip(transforms, relations):
            results.append(self.run(model, input_data, transform, relation, comparator))

        return results