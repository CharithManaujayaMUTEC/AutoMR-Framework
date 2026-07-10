import os
import pandas as pd

class EpsilonSensitivity:

    def __init__(self, api):
        self.api = api

    def run(
        self,
        dataset,
        epsilon_values,
        max_samples=None,
        samples_per_mr=5,
        show_progress=False,
        output_dir="results",
    ):
        """
        Optimized epsilon sensitivity analysis.

        Optimizations:
        -------------------------
        - Reuse transformed predictions
        - Reuse original predictions
        - Only comparator changes
        - No repeated inference across epsilon values
        """

        all_results = []

        # --------------------------------------------------
        # Shared cache across ALL epsilon runs
        # --------------------------------------------------
        prediction_cache = {}

        for eps in epsilon_values:

            print(f"\n===== Testing epsilon={eps:.4f} =====")

            df = self.api.run_dataset(
                dataset=dataset,
                max_samples=max_samples,
                samples_per_mr=samples_per_mr,
                show_progress=show_progress,
                epsilon=eps,

                # Shared cache
                prediction_cache=prediction_cache,
            )

            df["epsilon"] = eps

            all_results.append(df)

            # Save only if failures exist
            if (~df["passed"]).any():

                print(f"Failures detected for epsilon={eps:.4f}")

                results = self.api.analyze(df)

                self.api.save_results(
                    df,
                    results,
                    output_dir=os.path.join(
                        output_dir,
                        f"epsilon_{eps:.4f}",
                    ),
                )

            else:

                print(f"No failures for epsilon={eps:.4f}")

        return all_results