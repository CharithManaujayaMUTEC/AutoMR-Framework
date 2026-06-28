class EpsilonSensitivity:

    def __init__(self, api):
        self.api = api

    def run(
        self,
        dataset,
        epsilon_values,
        **kwargs
    ):

        all_results = []

        for eps in epsilon_values:

            df = self.api.run_dataset(
                dataset,
                epsilon=eps,
                **kwargs
            )

            df["epsilon"] = eps

            # Only keep epsilons that actually produced failures
            if (~df["passed"]).any():
                all_results.append(df)

        return all_results