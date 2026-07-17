"""
AutoMR Graph Generator

Generates evaluation graphs from AutoMR result DataFrames.

Author: AutoMR Framework
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GraphGenerator:
    """
    Generates all evaluation graphs for AutoMR.

    Expected dataframe columns
    --------------------------
    sample_id
    mr
    param
    original
    transformed
    difference
    percent_change
    severity
    passed
    status
    """

    def __init__(
        self,
        output_dir: str = "results",
        dpi: int = 300,
        figsize=(10, 6),
        style: str = "default",
    ):

        self.output_dir = Path(output_dir)
        self.graph_dir = self.output_dir / "graphs"

        self.dpi = dpi
        self.figsize = figsize

        self.graph_dir.mkdir(parents=True, exist_ok=True)

        plt.style.use(style)

    # -------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------

    def _mr_directory(self, mr_name: str) -> Path:

        directory = self.graph_dir / mr_name.replace(" ", "_").lower()
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _summary_directory(self) -> Path:

        directory = self.graph_dir / "summary"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _save_plot(
        self,
        fig,
        path: Path,
    ):

        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def _save_dataframe(
        self,
        df: pd.DataFrame,
        path: Path,
    ):

        df.to_csv(path, index=False)

    def _horizontal_reference_lines(
        self,
        ax,
        epsilon: Optional[float],
    ):

        ax.axhline(
            1.0,
            color="green",
            linestyle="--",
            linewidth=2,
            label="Expected",
        )

        if epsilon is not None:

            ax.axhline(
                1 + epsilon,
                color="red",
                linestyle=":",
                linewidth=1.5,
            )

            ax.axhline(
                1 - epsilon,
                color="red",
                linestyle=":",
                linewidth=1.5,
            )

    def _overall_directory(self) -> Path:
        """
        Directory for overall framework graphs.
        """
        output = self.graph_dir / "overall"
        output.mkdir(parents=True, exist_ok=True)
        return output

    # -------------------------------------------------------------
    # Parameter vs Prediction
    # -------------------------------------------------------------

    def parameter_vs_prediction(
        self,
        df: pd.DataFrame,
        mr_name: Optional[str] = None,
        epsilon: Optional[float] = None,
    ):

        if mr_name is not None:
            df = df[df["mr"] == mr_name]

        if df.empty:
            return

        output = self._mr_directory(df.iloc[0]["mr"])

        self._save_dataframe(
            df[["param", "transformed", "passed"]],
            output / "parameter_vs_prediction.csv",
        )

        fig, ax = plt.subplots(figsize=self.figsize)

        passed = df[df["passed"]]
        failed = df[~df["passed"]]

        if not passed.empty:
            ax.scatter(
                passed["param"],
                passed["transformed"],
                s=40,
                label="Passed",
            )

        if not failed.empty:
            ax.scatter(
                failed["param"],
                failed["transformed"],
                marker="x",
                s=50,
                label="Failed",
            )

        ax.set_title(
            f"{df.iloc[0]['mr']} - Parameter vs Prediction"
        )

        ax.set_xlabel("Transformation Parameter")
        ax.set_ylabel("Prediction")

        ax.grid(True, alpha=0.3)
        ax.legend()

        self._save_plot(
            fig,
            output / "parameter_vs_prediction.png",
        )

    # -------------------------------------------------------------
    # Test Case vs Prediction
    # -------------------------------------------------------------

    def testcase_vs_prediction(
        self,
        df: pd.DataFrame,
        mr_name: Optional[str] = None,
        epsilon: Optional[float] = None,
    ):

        if mr_name is not None:
            df = df[df["mr"] == mr_name]

        if df.empty:
            return

        output = self._mr_directory(df.iloc[0]["mr"])

        self._save_dataframe(
            df[["sample_id", "transformed", "passed"]],
            output / "testcase_vs_prediction.csv",
        )

        fig, ax = plt.subplots(figsize=self.figsize)

        passed = df[df["passed"]]
        failed = df[~df["passed"]]

        if not passed.empty:
            ax.scatter(
                passed["sample_id"],
                passed["transformed"],
                s=40,
                label="Passed",
            )

        if not failed.empty:
            ax.scatter(
                failed["sample_id"],
                failed["transformed"],
                marker="x",
                s=50,
                label="Failed",
            )

        ax.set_title(
            f"{df.iloc[0]['mr']} - Test Case vs Prediction"
        )

        ax.set_xlabel("Sample ID")
        ax.set_ylabel("Prediction")

        ax.grid(True, alpha=0.3)
        ax.legend()

        self._save_plot(
            fig,
            output / "testcase_vs_prediction.png",
        )

    # -------------------------------------------------------------
    # Failure Rate per MR
    # -------------------------------------------------------------

    def failure_rate_graph(
        self,
        failure_df: pd.DataFrame,
    ):
        """
        Expected columns
        ----------------
        mr
        failure_rate
        """

        if failure_df.empty:
            return

        output = self._summary_directory()

        self._save_dataframe(
            failure_df,
            output / "failure_rate.csv",
        )

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.bar(
            failure_df["mr"],
            failure_df["failure_rate"],
        )

        ax.set_title("Failure Rate per Metamorphic Relation")
        ax.set_xlabel("Metamorphic Relation")
        ax.set_ylabel("Failure Rate (%)")

        ax.set_ylim(0, 100)

        plt.xticks(rotation=45, ha="right")

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        self._save_plot(
            fig,
            output / "failure_rate.png",
        )

    # -------------------------------------------------------------
    # Severity per MR
    # -------------------------------------------------------------

    def severity_graph(
        self,
        severity_df: pd.DataFrame,
    ):
        """
        Expected columns
        ----------------
        mr
        severity
        """

        if severity_df.empty:
            return

        output = self._summary_directory()

        self._save_dataframe(
            severity_df,
            output / "severity.csv",
        )

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.bar(
            severity_df["mr"],
            severity_df["severity"],
        )

        ax.set_title("Failure Severity per Metamorphic Relation")

        ax.set_xlabel("Metamorphic Relation")
        ax.set_ylabel("Average Severity")

        plt.xticks(rotation=45, ha="right")

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        self._save_plot(
            fig,
            output / "severity.png",
        )

    # -------------------------------------------------------------
    # Range Analysis
    # -------------------------------------------------------------

    def range_analysis_graph(
        self,
        range_df: pd.DataFrame,
    ):

        if range_df.empty:
            return

        output = self._summary_directory()

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.bar(
            range_df["mr"],
            range_df["max_difference"],
        )

        ax.set_title(
            "Maximum Prediction Difference by MR"
        )

        ax.set_xlabel(
            "Metamorphic Relation"
        )

        ax.set_ylabel(
            "Maximum Difference"
        )

        plt.xticks(
            rotation=45,
            ha="right",
        )

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        self._save_plot(
            fig,
            output / "range_analysis.png",
        )
    # -------------------------------------------------------------
    # Epsilon Curve
    # -------------------------------------------------------------

    def epsilon_curve(
        self,
        epsilon_df: pd.DataFrame,
    ):
        """
        Expected columns
        ----------------
        epsilon
        failure_rate
        """

        if epsilon_df.empty:
            return

        output = self._summary_directory()

        self._save_dataframe(
            epsilon_df,
            output / "epsilon_curve.csv",
        )

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(
            epsilon_df["epsilon"],
            epsilon_df["failure_rate"],
            marker="o",
            linewidth=2,
        )

        ax.set_title("Failure Rate vs Epsilon")

        ax.set_xlabel("Epsilon")

        ax.set_ylabel("Failure Rate (%)")

        ax.grid(True)

        self._save_plot(
            fig,
            output / "epsilon_curve.png",
        )

    # -------------------------------------------------------------
    # Generate All
    # -------------------------------------------------------------

    def generate_all(
        self,
        results_df: pd.DataFrame,
        epsilon: Optional[float] = None,
    ):
        """
        Generates all per-MR graphs.

        Summary graphs should be generated separately using the
        outputs from FailureAnalyzer.
        """

        if results_df.empty:
            return

        for mr in sorted(results_df["mr"].unique()):

            mr_df = (
                results_df[
                    results_df["mr"] == mr
                ]
                .sort_values("param")
                .reset_index(drop=True)
            )

            self.parameter_vs_prediction(
                mr_df,
                epsilon=epsilon,
            )

            self.testcase_vs_prediction(
                mr_df,
                epsilon=epsilon,
            )

    # -------------------------------------------------------------
    # Generate Complete Evaluation
    # -------------------------------------------------------------

    def generate_complete(
        self,
        results_df: pd.DataFrame,
        failure_rate_df: Optional[pd.DataFrame] = None,
        severity_df: Optional[pd.DataFrame] = None,
        range_df: Optional[pd.DataFrame] = None,
        epsilon_df: Optional[pd.DataFrame] = None,
        epsilon: Optional[float] = None,
    ):

        self.generate_all(
            results_df,
            epsilon=epsilon,
        )

        if failure_rate_df is not None:
            self.failure_rate_graph(
                failure_rate_df
            )

        if severity_df is not None:
            self.severity_graph(
                severity_df
            )

        if range_df is not None:
            self.range_analysis_graph(
                range_df
            )

        if epsilon_df is not None:
            self.epsilon_curve(
                epsilon_df
            )

        self.pass_fail_pie(results_df)

        self.prediction_distribution(results_df)

        self.difference_distribution(results_df)

        self.failure_heatmap(results_df)

        if (
            failure_rate_df is not None
            and severity_df is not None
        ):
            self.summary_dashboard(
                results_df,
                failure_rate_df,
                severity_df,
            )

    def pass_fail_pie(self, df: pd.DataFrame):

        output = self._overall_directory()

        passed = int(df["passed"].sum())
        failed = len(df) - passed

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.pie(
            [passed, failed],
            labels=["Passed", "Failed"],
            autopct="%1.1f%%",
            startangle=90,
        )

        ax.set_title("Overall Test Results")

        self._save_plot(
            fig,
            output / "pass_fail_pie.png",
        )

    def prediction_distribution(self, df: pd.DataFrame):

        output = self._overall_directory()

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.hist(
            df["transformed"],
            bins=30,
        )

        ax.set_title("Prediction Distribution")
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Frequency")

        ax.grid(alpha=0.3)

        self._save_plot(
            fig,
            output / "prediction_distribution.png",
        )

    def difference_distribution(self, df: pd.DataFrame):

        output = self._overall_directory()

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.hist(
            df["difference"],
            bins=30,
        )

        ax.set_title("Prediction Difference Distribution")
        ax.set_xlabel("Difference")
        ax.set_ylabel("Frequency")

        ax.grid(alpha=0.3)

        self._save_plot(
            fig,
            output / "difference_distribution.png",
        )

    def worst_cases_graph(self, worst_df: pd.DataFrame):

        if worst_df.empty:
            return

        output = self._overall_directory()

        fig, ax = plt.subplots(figsize=(10, 6))

        labels = [
            f"{mr}\n{sid}"
            for mr, sid in zip(
                worst_df["mr"],
                worst_df["sample_id"],
            )
        ]

        ax.barh(
            labels,
            worst_df["severity"],
        )

        ax.set_title("Top Worst Metamorphic Violations")
        ax.set_xlabel("Severity")

        self._save_plot(
            fig,
            output / "worst_cases.png",
        )

    def failure_heatmap(self, df: pd.DataFrame):

        output = self._overall_directory()

        heatmap = pd.crosstab(
            df["mr"],
            df["passed"],
        )

        fig, ax = plt.subplots(figsize=(8, 8))

        im = ax.imshow(
            heatmap.values,
            aspect="auto",
        )

        ax.set_xticks(range(len(heatmap.columns)))
        ax.set_xticklabels(
            ["Failed", "Passed"]
        )

        ax.set_yticks(range(len(heatmap.index)))
        ax.set_yticklabels(heatmap.index)

        plt.colorbar(im)

        ax.set_title("Failure Heatmap")

        self._save_plot(
            fig,
            output / "failure_heatmap.png",
        )

    def summary_dashboard(
        self,
        df,
        failure_df,
        severity_df,
    ):

        output = self._overall_directory()

        fig, axs = plt.subplots(
            2,
            2,
            figsize=(14, 10),
        )

        axs[0,0].bar(
            failure_df["mr"],
            failure_df["failure_rate"],
        )
        axs[0,0].set_title("Failure Rate")

        axs[0,1].bar(
            severity_df["mr"],
            severity_df["severity"],
        )
        axs[0,1].set_title("Severity")

        axs[1,0].hist(
            df["difference"],
            bins=25,
        )
        axs[1,0].set_title("Difference Distribution")

        passed = int(df["passed"].sum())
        failed = len(df) - passed

        axs[1,1].pie(
            [passed, failed],
            labels=["Passed", "Failed"],
            autopct="%1.1f%%",
        )
        axs[1,1].set_title("Pass vs Fail")

        plt.tight_layout()

        self._save_plot(
            fig,
            output / "summary_dashboard.png",
        )