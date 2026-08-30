"""
Metamorphic Relation (MR) tester.

This module provides a simple executor for evaluating a single
metamorphic relation on a single input sample.
"""


class MRTester:
    """
    Executes a single metamorphic test case.
    """

    def run(self, model, input_data, transform, relation, comparator):
        """
        Execute a single metamorphic relation.

        Parameters
        ----------
        model : object
            Wrapped model used for prediction.
        input_data : Any
            Original input sample.
        transform : callable
            Transformation applied to the input.
        relation : object
            Metamorphic relation used for verification.
        comparator : object
            Comparator used to calculate prediction differences.

        Returns
        -------
        dict
            Dictionary containing the test result and metadata.
        """

        # Generate the prediction for the original input.
        original = model.predict(input_data)

        # Apply the transformation to create the follow-up input.
        transformed_input = transform(input_data)

        # Generate the prediction for the transformed input.
        transformed = model.predict(transformed_input)

        # Let the metamorphic relation determine whether the test passes.
        passed = relation.check(original, transformed)

        # Use the comparator to compute the prediction difference.
        if comparator:
            diff, _ = comparator.compare(original, transformed)
        else:
            # Fallback difference calculation.
            diff = transformed - original

        # Return the complete test result.
        return {
            "mr": relation.__class__.__name__,
            "original": original,
            "transformed": transformed,
            "difference": diff,
            "passed": passed,
            "expected_behavior": relation.expected() if hasattr(relation, "expected") else "N/A",
            "actual_behavior": "Consistent" if passed else "Violation"
        }