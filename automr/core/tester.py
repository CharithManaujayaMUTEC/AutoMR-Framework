class MRTester:

    def run(self, model, input_data, transform, relation, comparator):

        original = model.predict(input_data)
        transformed_input = transform(input_data)
        transformed = model.predict(transformed_input)

        #  RELATION-FIRST LOGIC
        if hasattr(relation, "type") and relation.type() in ["behavioral", "temporal"]:
            diff = transformed - original
            passed = relation.check(original, transformed)
        elif comparator:
            diff, passed = comparator.compare(original, transformed)
        else:
            diff = transformed - original
            passed = relation.check(original, transformed)

        return {
            "mr": relation.__class__.__name__,
            "original": original,
            "transformed": transformed,
            "difference": diff,
            "passed": passed,
            "expected_behavior": relation.expected() if hasattr(relation, "expected") else "N/A",
            "actual_behavior": "Consistent" if passed else "Violation"
        }