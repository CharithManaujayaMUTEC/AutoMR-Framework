import pandas as pd


class EpsilonSummary:

    def __init__(self, stabilization_delta=0.01):
        """
        stabilization_delta:
            Maximum change in failure rate (%) between consecutive
            epsilon values to consider the curve stabilized.
        """
        self.stabilization_delta = stabilization_delta

    def summarize(self, dfs):
        """
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

        if len(dfs) == 0:
            return pd.DataFrame(), {
                "first_failure_epsilon": None,
                "recommended_epsilon": None,
                "stabilization_epsilon": None,
                "max_failure_rate": 0
            }

        summary = []

        for df in dfs:

            epsilon = float(df["epsilon"].iloc[0])

            total = len(df)
            failed = (~df["passed"]).sum()
            passed = total - failed

            failure_rate = failed / total

            summary.append({
                "epsilon": epsilon,
                "total": total,
                "passed": passed,
                "failed": failed,
                "failure_rate": failure_rate
            })

        summary_df = (
            pd.DataFrame(summary)
            .sort_values("epsilon")
            .reset_index(drop=True)
        )

        # ----------------------------------------
        # First epsilon that produced failures
        # ----------------------------------------

        failed_rows = summary_df[
            summary_df["failed"] > 0
        ]

        if len(failed_rows):

            first_failure = float(
                failed_rows.iloc[0]["epsilon"]
            )

        else:

            first_failure = None

        # ----------------------------------------
        # Stabilization detection
        # ----------------------------------------

        stabilization = None

        rates = summary_df["failure_rate"].values
        eps = summary_df["epsilon"].values

        for i in range(1, len(rates)):

            change = abs(rates[i] - rates[i - 1])

            if change <= self.stabilization_delta:

                stabilization = float(eps[i])
                break

        # ----------------------------------------
        # Recommended epsilon
        # ----------------------------------------

        recommended = stabilization

        if recommended is None:
            recommended = first_failure

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