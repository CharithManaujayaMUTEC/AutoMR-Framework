"""
Epsilon summary module.

This module summarizes the results of epsilon sensitivity analysis
by computing failure statistics, identifying the first failure
epsilon, estimating stabilization, and recommending an epsilon
value for future evaluations.
"""

import pandas as pd


class EpsilonSummary:
    """
    Generates summary statistics and reports for epsilon
    sensitivity analysis.
    """

    def __init__(self, stabilization_delta=0.01):
        """
        Initialize the epsilon summary analyzer.

        Parameters
        ----------
        stabilization_delta : float
            Maximum change in failure rate (%) between consecutive
            epsilon values to consider the curve stabilized.
        """
        self.stabilization_delta = stabilization_delta

    def summarize(self, dfs):
        """
        Summarize the results from multiple epsilon runs.

        Parameters
        ----------
        dfs : list[pd.DataFrame]
            List of DataFrames returned from each epsilon run.

        Returns
        -------
        summary_df : pd.DataFrame
            Failure statistics per epsilon.

        report : dict
            Overall epsilon analysis.
        """

        # Handle the case where no results are available.
        if len(dfs) == 0:
            return pd.DataFrame(), {
                "first_failure_epsilon": None,
                "recommended_epsilon": None,
                "stabilization_epsilon": None,
                "max_failure_rate": 0
            }

        # Store summary statistics for each epsilon.
        summary = []

        # Process each epsilon execution.
        for df in dfs:

            # Retrieve the evaluated epsilon value.
            epsilon = float(df["epsilon"].iloc[0])

            # Compute pass/fail statistics.
            total = len(df)
            failed = (~df["passed"]).sum()
            passed = total - failed

            # Calculate the failure rate.
            failure_rate = failed / total

            # Store the statistics.
            summary.append({
                "epsilon": epsilon,
                "total": total,
                "passed": passed,
                "failed": failed,
                "failure_rate": failure_rate
            })

        # Build a sorted summary table.
        summary_df = (
            pd.DataFrame(summary)
            .sort_values("epsilon")
            .reset_index(drop=True)
        )

        # ----------------------------------------
        # First epsilon that produced failures
        # ----------------------------------------

        # Find all epsilon values with at least one failure.
        failed_rows = summary_df[
            summary_df["failed"] > 0
        ]

        if len(failed_rows):

            # Record the first epsilon where failures appear.
            first_failure = float(
                failed_rows.iloc[0]["epsilon"]
            )

        else:

            # No failures were detected.
            first_failure = None

        # ----------------------------------------
        # Stabilization detection
        # ----------------------------------------

        # Detect where the failure-rate curve stabilizes.
        stabilization = None

        rates = summary_df["failure_rate"].values
        eps = summary_df["epsilon"].values

        for i in range(1, len(rates)):

            # Measure the change between consecutive failure rates.
            change = abs(rates[i] - rates[i - 1])

            if change <= self.stabilization_delta:

                stabilization = float(eps[i])
                break

        # ----------------------------------------
        # Recommended epsilon
        # ----------------------------------------

        # Prefer the stabilization point when available.
        recommended = stabilization

        # Otherwise, recommend the first failing epsilon.
        if recommended is None:
            recommended = first_failure

        # Construct the summary report.
        report = {

            "first_failure_epsilon": first_failure,

            "recommended_epsilon": recommended,

            "stabilization_epsilon": stabilization,

            "max_failure_rate": float(
                summary_df["failure_rate"].max()
            )
        }

        return summary_df, report

    def print_report(self, report):
        """
        Print a formatted epsilon analysis report.

        Parameters
        ----------
        report : dict
            Summary report returned by summarize().
        """

        print("\n========== EPSILON ANALYSIS ==========")

        print(
            f"First Failure Epsilon : {report['first_failure_epsilon']}"
        )

        print(
            f"Recommended Epsilon   : {report['recommended_epsilon']}"
        )

        print(
            f"Stabilization Epsilon : {report['stabilization_epsilon']}"
        )

        print(
            f"Maximum Failure Rate  : {report['max_failure_rate']:.2%}"
        )

        print("======================================")