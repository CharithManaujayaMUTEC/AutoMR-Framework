
class MRTester:

    def run(self, model, input_data, transform, relation):

        # Step 1: original prediction
        original_output = model.predict(input_data)

        # Step 2: transform input
        transformed_input = transform(input_data)

        # Step 3: new prediction
        transformed_output = model.predict(transformed_input)

        # Step 4: check MR
        passed = relation.check(original_output, transformed_output)

        return {
            "original_output": original_output,
            "transformed_output": transformed_output,
            "passed": passed
        }
