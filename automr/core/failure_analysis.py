"""
Failure analysis utilities for AutoMR.

This module provides methods for analyzing metamorphic testing
results, including failure rates, severity metrics, worst-case
violations, failure regions, and range-based summaries.
"""

import pandas as pd
import numpy as np


class FailureAnalyzer:
    """
    Performs post-execution analysis on AutoMR test results.
    """

    def failure_rate_per_mr(self, df):
        """
        Calculate the failure rate for each metamorphic relation (MR).

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing AutoMR execution results.

        Returns
        -------
        pandas.DataFrame
            Failure statistics for each MR, sorted by failure rate.
        """

        # Compute the total and passed test counts for each MR.
        summary = df.groupby("mr")["passed"].agg(
            total="count",
            passed="sum"
        ).reset_index()

        # Compute failure statistics.
        summary["failed"] = summary["total"] - summary["passed"]
        summary["failure_rate"] = summary["failed"] / summary["total"]

        # Return MRs ordered by highest failure rate.
        return summary.sort_values(by="failure_rate", ascending=False)

    def severity_per_mr(self, df):
        """
        Calculate the average severity of failed test cases.

        Returns
        -------
        pandas.DataFrame
            Columns:
                mr
                severity
        """

        if "severity" not in df.columns:
            raise ValueError(
                "Results do not contain a severity column. "
                "Severity should be computed during MR execution."
            )

        failed = df[df["passed"] == False]

        # No failures
        if failed.empty:
            return pd.DataFrame({
                "mr": sorted(df["mr"].unique()),
                "severity": [0.0] * len(df["mr"].unique())
            })

        severity = (
            failed.groupby("mr")["severity"]
            .mean()
            .reindex(
                sorted(df["mr"].unique()),
                fill_value=0.0,
            )
            .reset_index()
        )

        severity.columns = ["mr", "severity"]

        return severity.sort_values(
            by="severity",
            ascending=False,
        ).reset_index(drop=True)

    def worst_cases(self, df, top_k=10):
        """
        Retrieve the most severe metamorphic violations.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing AutoMR execution results.
        top_k : int, default=10
            Number of worst cases to return.

        Returns
        -------
        pandas.DataFrame
            Top-ranked violations by severity.
        """

        if "severity" not in df.columns:
            raise ValueError(
                "Results do not contain a severity column. "
                "Severity should be computed during MR execution."
            )

        # Return the most severe violations.
        return df.sort_values(by="severity", ascending=False).head(top_k)

    def failure_regions(self, df):
        """
        Detect continuous parameter regions where failures occur.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing AutoMR execution results.

        Returns
        -------
        dict
            Dictionary mapping each MR to its detected failure regions.
        """

        regions = {}

        # Analyze failures independently for each MR.
        for mr in df["mr"].unique():
            sub = df[(df["mr"] == mr) & (df["passed"] == False)]

            # Skip relations with no failures.
            if sub.empty:
                continue

            # Sort failed parameter values.
            params = sorted(sub["param"].values)

            grouped = []
            current = [params[0]]

            # Group nearby parameter values into continuous regions.
            # Estimate the parameter spacing used for this MR.
            if len(params) > 1:
                step = np.median(np.diff(params))
            else:
                step = 0.0

            threshold = step * 1.5

            for i in range(1, len(params)):
                if abs(params[i] - params[i - 1]) <= threshold:
                    current.append(params[i])
                else:
                    grouped.append((min(current), max(current)))
                    current = [params[i]]

            # Store the final region.
            grouped.append((min(current), max(current)))
            regions[mr] = grouped

        return regions
    
    def range_summary(self, df):
        """
        Extract unique range-testing results for each MR.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing AutoMR execution results.

        Returns
        -------
        pandas.DataFrame
            Unique range summary records.
        """

        cols = [
            "mr",
            "range_change",
            "range_percent_change",
            "range_passed"
        ]

        # Remove duplicate range records.
        return (
            df[cols]
            .drop_duplicates()
            .reset_index(drop=True)
        )
    
    def range_analysis(self, df):
        """
        Generate aggregated statistics from range-testing results.

        Returns
        -------
        pandas.DataFrame
        """

        return (
            df.groupby("mr")
            .agg(
                max_range_change=("range_change", "max"),
                max_range_percent_change=("range_percent_change", "max"),
                range_passed=("range_passed", "first"),
                mean_difference=("difference", "mean"),
                max_difference=("difference", "max"),
            )
            .reset_index()
        )