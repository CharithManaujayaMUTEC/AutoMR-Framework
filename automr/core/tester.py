
class MRTester:

    def run(self, model, input_data, transform, relation):

        original_output = model.predict(input_data)

        transformed_input = transform(input_data)

        transformed_output = model.predict(transformed_input)

        return {
            "mr": relation.__class__.__name__,
            "original": original_output,
            "transformed": transformed_output,
            "passed": relation.check(original_output, transformed_output)
        }

    def run_all(self, model, input_data, transforms, relations):

        results = []

        for transform, relation in zip(transforms, relations):
            result = self.run(model, input_data, transform, relation)
            results.append(result)

        return results
