"""
Epsilon sensitivity summary and diagnostics for AutoMR.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class EpsilonSummary:
    """
    Analyze epsilon sensitivity results and generate
    diagnostic information.

    In addition to the existing recommended epsilon
    analysis, this class provides:

    - Prediction range
    - Normalized epsilon
    - Plateau detection
    - Flat-zero curve detection
    - Diagnostic warnings
    """

    def __init__(
        self,
        stabilization_threshold: float = 0.01,
        plateau_threshold: float = 0.01,
        plateau_min_points: int = 3,
    ):
        self.stabilization_threshold = stabilization_threshold
        self.plateau_threshold = plateau_threshold
        self.plateau_min_points = plateau_min_points

    # ==================================================
    # Public Summary
    # ==================================================

    def summarize(
        self,
        results: List[Dict[str, Any]],
        prediction_range: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate epsilon sensitivity summary.

        Parameters
        ----------
        results:
            Epsilon sensitivity results.

        prediction_range:
            Observed baseline prediction range:

                max_prediction - min_prediction

            Used only for normalized epsilon diagnostics.

        Returns
        -------
        dict
            Summary and diagnostic information.
        """

        if not results:

            return {
                "recommended_epsilon": None,
                "first_failure_epsilon": None,
                "stabilization_epsilon": None,
                "max_failure_rate": 0.0,
                "prediction_range": prediction_range,
                "normalized_epsilon": None,
                "curve_diagnostic": "no_results",
                "warnings": [
                    "No epsilon sensitivity results available."
                ],
            }

        # ----------------------------------------------
        # Sort by epsilon
        # ----------------------------------------------

        sorted_results = sorted(
            results,
            key=lambda item: item.get(
                "epsilon",
                0.0,
            ),
        )

        epsilons = [
            float(item.get("epsilon", 0.0))
            for item in sorted_results
        ]

        failure_rates = [
            float(
                item.get(
                    "failure_rate",
                    0.0,
                )
            )
            for item in sorted_results
        ]

        warnings = []

        # ----------------------------------------------
        # First failure epsilon
        # ----------------------------------------------

        first_failure_epsilon = None

        for epsilon, failure_rate in zip(
            epsilons,
            failure_rates,
        ):

            if failure_rate > 0:

                first_failure_epsilon = epsilon

                break

        # ----------------------------------------------
        # Maximum failure rate
        # ----------------------------------------------

        max_failure_rate = max(
            failure_rates
        )

        # ----------------------------------------------
        # Stabilization epsilon
        # ----------------------------------------------

        stabilization_epsilon = None

        if len(failure_rates) >= 2:

            for index in range(
                1,
                len(failure_rates),
            ):

                difference = abs(
                    failure_rates[index]
                    - failure_rates[index - 1]
                )

                if (
                    difference
                    <= self.stabilization_threshold
                ):

                    stabilization_epsilon = (
                        epsilons[index]
                    )

                    break

        # ----------------------------------------------
        # Recommended epsilon
        #
        # Preserve the existing interpretation:
        # prefer stabilization, otherwise first failure.
        # ----------------------------------------------

        recommended_epsilon = (
            stabilization_epsilon
            if stabilization_epsilon is not None
            else first_failure_epsilon
        )

        # ----------------------------------------------
        # Prediction range diagnostics
        # ----------------------------------------------

        normalized_epsilon = None

        if (
            recommended_epsilon is not None
            and prediction_range is not None
        ):

            if prediction_range > 0:

                normalized_epsilon = (
                    recommended_epsilon
                    / prediction_range
                )

            else:

                warnings.append(
                    "Prediction range is zero or "
                    "invalid. Normalized epsilon "
                    "cannot be calculated."
                )

        # ----------------------------------------------
        # Flat-zero curve detection
        # ----------------------------------------------

        flat_zero_curve = (
            len(failure_rates) > 0
            and all(
                rate == 0
                for rate in failure_rates
            )
        )

        # ----------------------------------------------
        # General plateau detection
        # ----------------------------------------------

        plateau_detected = False

        if (
            len(failure_rates)
            >= self.plateau_min_points
        ):

            consecutive = 1

            for index in range(
                1,
                len(failure_rates),
            ):

                difference = abs(
                    failure_rates[index]
                    - failure_rates[index - 1]
                )

                if (
                    difference
                    <= self.plateau_threshold
                ):

                    consecutive += 1

                    if (
                        consecutive
                        >= self.plateau_min_points
                    ):

                        plateau_detected = True

                        break

                else:

                    consecutive = 1

        # ----------------------------------------------
        # Curve diagnostic
        # ----------------------------------------------

        if flat_zero_curve:

            curve_diagnostic = "flat_zero"

            warnings.append(
                "All epsilon values produced a zero "
                "failure rate. This may indicate "
                "genuine robustness, but baseline "
                "prediction and decoder health should "
                "also be considered."
            )

        elif plateau_detected:

            curve_diagnostic = "plateau"

            warnings.append(
                "The epsilon sensitivity curve "
                "contains a plateau. This may reflect "
                "stable behavior, discrete predictions, "
                "or output saturation."
            )

        else:

            curve_diagnostic = "variable"

        # ----------------------------------------------
        # No observed failures
        # ----------------------------------------------

        if (
            first_failure_epsilon is None
            and not flat_zero_curve
        ):

            warnings.append(
                "No positive failure rate was detected "
                "in the evaluated epsilon range."
            )

        # ----------------------------------------------
        # Final summary
        # ----------------------------------------------

        return {
            "recommended_epsilon":
                recommended_epsilon,

            "first_failure_epsilon":
                first_failure_epsilon,

            "stabilization_epsilon":
                stabilization_epsilon,

            "max_failure_rate":
                max_failure_rate,

            "prediction_range":
                prediction_range,

            "normalized_epsilon":
                normalized_epsilon,

            "plateau_detected":
                plateau_detected,

            "flat_zero_curve":
                flat_zero_curve,

            "curve_diagnostic":
                curve_diagnostic,

            "warnings":
                warnings,
        }