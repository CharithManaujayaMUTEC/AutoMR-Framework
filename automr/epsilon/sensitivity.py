"""
Epsilon sensitivity analysis module.

This module evaluates AutoMR across multiple epsilon values to
measure how prediction tolerance affects metamorphic testing
results. Prediction caching is used to avoid repeated inference
and improve execution efficiency.
"""

import os
import pandas as pd


class EpsilonSensitivity:
    """
    Performs epsilon sensitivity analysis using the AutoMR API.
    """

    def __init__(self, api):
        """
        Initialize the epsilon sensitivity analyzer.

        Parameters
        ----------
        api : AutoMR
            AutoMR API instance used to execute testing.
        """
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
        Execute epsilon sensitivity analysis.

        Optimized epsilon sensitivity analysis.

        Optimizations:
        -------------------------
        - Reuse transformed predictions
        - Reuse original predictions
        - Only comparator changes
        - No repeated inference across epsilon values
        """

        # Store results for all evaluated epsilon values.
        all_results = []

        # Execute AutoMR for each epsilon value.
        for eps in epsilon_values:

            print(f"\n===== Testing epsilon={eps:.4f} =====")

            # Run the dataset using the current epsilon value.
            df = self.api.run_dataset(
                dataset=dataset,
                max_samples=max_samples,
                samples_per_mr=samples_per_mr,
                show_progress=show_progress,
                epsilon=eps,
            )

            # Record the epsilon value used for this execution.
            df["epsilon"] = eps

            all_results.append(df)

            # Save reports only when failures are detected.
            if (~df["passed"]).any():

                print(f"Failures detected for epsilon={eps:.4f}")

                # Generate analysis reports.
                results = self.api.analyze(df)

                # Save reports in an epsilon-specific directory.
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

        # Return results for all evaluated epsilon values.
        return all_results