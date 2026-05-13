class MRTester:

    def run(self, model, input_data, transform, relation, comparator):

        original = model.predict(input_data)
        transformed_input = transform(input_data)
        transformed = model.predict(transformed_input)

        #  MR decides
        passed = relation.check(original, transformed)

        # comparator only logs diff
        if comparator:
            diff, _ = comparator.compare(original, transformed)
        else:
            diff = transformed - original

        return {
            "mr": relation.__class__.__name__,
            "original": original,
            "transformed": transformed,
            "difference": diff,
            "passed": passed,
            "expected_behavior": relation.expected() if hasattr(relation, "expected") else "N/A",
            "actual_behavior": "Consistent" if passed else "Violation"
        }