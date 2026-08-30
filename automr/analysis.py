"""
Analysis utilities.

This module provides helper functions for converting raw test
results into pandas DataFrames and generating summary statistics
for metamorphic testing experiments.
"""

import pandas as pd


class Analyzer:
    """
    Utility class for analyzing AutoMR test results.
    """

    def to_dataframe(self, results):
        """
        Convert a list of test results into a pandas DataFrame.

        Parameters
        ----------
        results : list
            Collection of test result dictionaries.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing all test results.
        """
        if not results:
            return pd.DataFrame()

        # Create the DataFrame and add a human-readable status column.
        df = pd.DataFrame(results)
        df["status"] = df["passed"].apply(
            lambda x: "PASS" if x else "FAIL"
        )
        return df

    def summary(self, df):
        """
        Compute summary statistics for a test run.

        Parameters
        ----------
        df : pandas.DataFrame
            Test result DataFrame.

        Returns
        -------
        dict
            Summary statistics including totals and pass rate.
        """

        if df.empty:
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0
            }

        # Compute overall statistics.
        total = len(df)
        passed = df["passed"].sum()
        failed = total - passed

        return {
            "total": total,
            "passed": int(passed),
            "failed": int(failed),
            "pass_rate": float(passed / total * 100)
        }

    def prediction_trace(self, df):
        """
        Extract prediction trace information.

        Parameters
        ----------
        df : pandas.DataFrame
            Test result DataFrame.

        Returns
        -------
        pandas.DataFrame
            Prediction trace for each executed test.
        """

        cols = [
            "sample_id",
            "mr",
            "param",
            "original",
            "transformed",
            "difference",
            "percent_change",
            "passed"
        ]

        return df[cols].copy()