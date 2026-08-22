"""
Decoder Health Diagnostics for AutoMR.

Analyzes already-generated baseline predictions to detect
potential decoder problems before metamorphic testing.

This module is warning-only and never blocks test execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


class DecoderHealthAnalyzer:
    """
    Analyze scalar baseline predictions for potential
    decoder anomalies.

    Checks:
    - Variance / standard deviation
    - Prediction range
    - Unique-value ratio
    - Saturation
    - Clipping
    - Basic distribution diagnostics

    The analyzer is warning-only and never raises an error
    because of suspicious prediction behavior.
    """

    def __init__(
        self,
        saturation_threshold: float = 0.95,
        clipping_tolerance: float = 1e-12,
        unique_ratio_threshold: float = 0.10,
        near_zero_std_ratio: float = 1e-6,
        low_cardinality_threshold: int = 5,
    ):
        self.saturation_threshold = float(
            saturation_threshold
        )

        self.clipping_tolerance = float(
            clipping_tolerance
        )

        self.unique_ratio_threshold = float(
            unique_ratio_threshold
        )

        self.near_zero_std_ratio = float(
            near_zero_std_ratio
        )

        self.low_cardinality_threshold = int(
            low_cardinality_threshold
        )

    # ==================================================
    # Prediction Conversion
    # ==================================================

    @staticmethod
    def _to_numeric_array(
        predictions: Iterable[Any],
    ) -> np.ndarray:
        """
        Convert valid finite predictions to a NumPy array.

        Non-numeric and non-finite values are ignored.
        """

        values: List[float] = []

        for prediction in predictions:

            try:
                value = float(prediction)

            except (
                TypeError,
                ValueError,
            ):
                continue

            if np.isfinite(value):

                values.append(value)

        return np.asarray(
            values,
            dtype=np.float64,
        )

    # ==================================================
    # Main Analysis
    # ==================================================

    def analyze(
        self,
        predictions: Iterable[Any],
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze baseline predictions.

        Parameters
        ----------
        predictions:
            Existing baseline scalar predictions.

        output_path:
            Optional JSON output path.

        Returns
        -------
        dict
            Decoder health report.

        Notes
        -----
        This method is warning-only.
        It never blocks or stops AutoMR testing.
        """

        values = self._to_numeric_array(
            predictions
        )

        warnings: List[str] = []

        # ----------------------------------------------
        # No valid predictions
        # ----------------------------------------------

        if values.size == 0:

            report = {
                "status": "WARNING",
                "warning_only": True,
                "prediction_count": 0,
                "mean": None,
                "variance": None,
                "std": None,
                "min": None,
                "max": None,
                "range": None,
                "unique_count": 0,
                "unique_ratio": 0.0,
                "saturation_ratio": 0.0,
                "lower_clipping_ratio": 0.0,
                "upper_clipping_ratio": 0.0,
                "distribution_diagnostic":
                    "no_valid_predictions",
                "warnings": [
                    "No finite baseline predictions "
                    "were available for decoder health "
                    "analysis."
                ],
            }

            self._save_report(
                report,
                output_path,
            )

            return report

        # ----------------------------------------------
        # Basic statistics
        # ----------------------------------------------

        minimum = float(
            np.min(values)
        )

        maximum = float(
            np.max(values)
        )

        mean = float(
            np.mean(values)
        )

        variance = float(
            np.var(values)
        )

        std = float(
            np.std(values)
        )

        value_range = float(
            maximum - minimum
        )

        # ----------------------------------------------
        # Unique-value analysis
        # ----------------------------------------------

        unique_count = int(
            np.unique(values).size
        )

        unique_ratio = float(
            unique_count / values.size
        )

        # ----------------------------------------------
        # Quantiles
        # ----------------------------------------------

        q01 = float(
            np.quantile(values, 0.01)
        )

        q99 = float(
            np.quantile(values, 0.99)
        )

        # ----------------------------------------------
        # Saturation detection
        # ----------------------------------------------

        if value_range > 0:

            edge_ratio = (
                1.0
                - self.saturation_threshold
            )

            lower_edge = (
                minimum
                + edge_ratio
                * value_range
            )

            upper_edge = (
                maximum
                - edge_ratio
                * value_range
            )

            lower_extreme = (
                values <= lower_edge
            )

            upper_extreme = (
                values >= upper_edge
            )

        else:

            lower_extreme = np.ones(
                values.size,
                dtype=bool,
            )

            upper_extreme = np.ones(
                values.size,
                dtype=bool,
            )

        lower_saturation_ratio = float(
            np.mean(lower_extreme)
        )

        upper_saturation_ratio = float(
            np.mean(upper_extreme)
        )

        saturation_ratio = max(
            lower_saturation_ratio,
            upper_saturation_ratio,
        )

        # ----------------------------------------------
        # Clipping detection
        # ----------------------------------------------

        lower_clipped = np.isclose(
            values,
            minimum,
            atol=self.clipping_tolerance,
        )

        upper_clipped = np.isclose(
            values,
            maximum,
            atol=self.clipping_tolerance,
        )

        lower_clipping_ratio = float(
            np.mean(lower_clipped)
        )

        upper_clipping_ratio = float(
            np.mean(upper_clipped)
        )

        # ----------------------------------------------
        # Relative standard deviation
        # ----------------------------------------------

        std_ratio = (
            std
            / max(abs(mean), 1.0)
        )

        # ----------------------------------------------
        # Distribution diagnostics
        # ----------------------------------------------

        if (
            value_range == 0.0
            or unique_count == 1
        ):

            distribution_diagnostic = (
                "constant"
            )

            warnings.append(
                "Baseline predictions are constant "
                "or nearly constant. The scalar "
                "decoder may be degenerate."
            )

        elif (
            unique_count
            <= self.low_cardinality_threshold
        ):

            distribution_diagnostic = (
                "low_cardinality"
            )

            warnings.append(
                "Baseline predictions contain very "
                "few unique values. The decoded "
                "quantity may be discrete or "
                "heavily quantized."
            )

        elif (
            unique_ratio
            < self.unique_ratio_threshold
        ):

            distribution_diagnostic = (
                "low_diversity"
            )

            warnings.append(
                "Baseline predictions have low "
                "unique-value diversity."
            )

        else:

            distribution_diagnostic = (
                "continuous_like"
            )

        # ----------------------------------------------
        # Low standard deviation
        # ----------------------------------------------

        if (
            values.size > 1
            and std_ratio
            <= self.near_zero_std_ratio
        ):

            warnings.append(
                "Prediction standard deviation is "
                "extremely small relative to the "
                "prediction scale."
            )

        # ----------------------------------------------
        # Saturation warning
        # ----------------------------------------------

        if (
            saturation_ratio
            >= self.saturation_threshold
        ):

            warnings.append(
                "A large proportion of predictions "
                "lie near an observed boundary. "
                "Possible saturation detected."
            )

        # ----------------------------------------------
        # Clipping warning
        # ----------------------------------------------

        if (
            max(
                lower_clipping_ratio,
                upper_clipping_ratio,
            )
            >= self.saturation_threshold
        ):

            warnings.append(
                "A large proportion of predictions "
                "exactly match an observed minimum "
                "or maximum. Possible clipping "
                "detected."
            )

        # ----------------------------------------------
        # Build report
        # ----------------------------------------------

        report = {
            "status":
                "WARNING"
                if warnings
                else "OK",

            "warning_only": True,

            "prediction_count":
                int(values.size),

            "mean":
                mean,

            "variance":
                variance,

            "std":
                std,

            "min":
                minimum,

            "max":
                maximum,

            "range":
                value_range,

            "unique_count":
                unique_count,

            "unique_ratio":
                unique_ratio,

            "q01":
                q01,

            "q99":
                q99,

            "saturation_ratio":
                saturation_ratio,

            "lower_clipping_ratio":
                lower_clipping_ratio,

            "upper_clipping_ratio":
                upper_clipping_ratio,

            "distribution_diagnostic":
                distribution_diagnostic,

            "warnings":
                warnings,
        }

        self._save_report(
            report,
            output_path,
        )

        return report

    # ==================================================
    # Save Report
    # ==================================================

    @staticmethod
    def _save_report(
        report: Dict[str, Any],
        output_path: Optional[str],
    ) -> None:
        """
        Save decoder health report as JSON.
        """

        if not output_path:
            return

        path = Path(output_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
            )