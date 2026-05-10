
class MRTester:

    def run(self, model, input_data, transform, relation):

        original = model.predict(input_data)
        transformed_input = transform(input_data)
        transformed = model.predict(transformed_input)

        diff = transformed - original
        pct = (diff / (abs(original) + 1e-6)) * 100

        return {
            "mr": relation.__class__.__name__,
            "original": float(original),
            "transformed": float(transformed),
            "difference": float(diff),
            "percent_change": float(pct),
            "passed": bool(relation.check(original, transformed))
        }

    def run_all(self, model, input_data, transforms, relations):

        results = []

        for transform, relation in zip(transforms, relations):
            results.append(self.run(model, input_data, transform, relation))

        return results
