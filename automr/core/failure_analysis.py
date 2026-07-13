"""
Failure analysis utilities for AutoMR.

This module provides methods for analyzing metamorphic testing
results, including failure rates, severity metrics, worst-case
violations, failure regions, and range-based summaries.
"""

import pandas as pd


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
        Calculate the average violation severity for each MR.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing AutoMR execution results.

        Returns
        -------
        pandas.Series
            Mean severity for each MR.
        """

        # Create the severity column if it does not already exist.
        if "severity" not in df.columns:
            df["severity"] = df["difference"].abs()

        # Compute the average severity for each MR.
        return df.groupby("mr")["severity"].mean().sort_values(ascending=False)

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

        # Create the severity column if necessary.
        if "severity" not in df.columns:
            df["severity"] = df["difference"].abs()

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
            for i in range(1, len(params)):
                if abs(params[i] - params[i-1]) < 0.05:
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

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing AutoMR execution results.

        Returns
        -------
        pandas.DataFrame
            Aggregated range statistics for each MR.
        """

        # Aggregate range and prediction statistics.
        return (
            df.groupby("mr")
            .agg({
                "range_change": "max",
                "range_percent_change": "max",
                "range_passed": "first",
                "difference": ["mean", "max"]
            })
            .reset_index()
        )