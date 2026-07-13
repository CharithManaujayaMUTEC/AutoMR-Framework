"""
Utility functions for epsilon sensitivity analysis.

This module provides helper functions for generating epsilon values,
updating relation tolerances, applying epsilon values across the
relation registry, and computing basic failure statistics.
"""

import numpy as np
import pandas as pd


def generate_epsilon_values(
    epsilon_min,
    epsilon_max,
    epsilon_count,
):
    """
    Generate evenly spaced epsilon values.

    Parameters
    ----------
    epsilon_min : float
        Minimum epsilon value.
    epsilon_max : float
        Maximum epsilon value.
    epsilon_count : int
        Number of epsilon values to generate.

    Returns
    -------
    numpy.ndarray
        Evenly spaced epsilon values.
    """

    return np.linspace(
        float(epsilon_min),
        float(epsilon_max),
        int(epsilon_count)
    )


def relation_attribute_name(relation):
    """
    Determine which relation attribute controls prediction tolerance.

    Parameters
    ----------
    relation : object
        Metamorphic relation instance.

    Returns
    -------
    str or None
        Name of the tolerance attribute, or None if unsupported.
    """

    # Check for the standard tolerance attribute.
    if hasattr(relation, "tolerance"):
        return "tolerance"

    # Alternative attribute names supported by different relations.
    if hasattr(relation, "epsilon"):
        return "epsilon"

    if hasattr(relation, "delta"):
        return "delta"

    if hasattr(relation, "max_change"):
        return "max_change"

    # No configurable tolerance attribute found.
    return None


def set_relation_epsilon(
    relation,
    epsilon,
):
    """
    Update the tolerance value of a relation dynamically.

    Parameters
    ----------
    relation : object
        Metamorphic relation instance.
    epsilon : float
        New tolerance value.
    """

    # Determine which attribute controls tolerance.
    attr = relation_attribute_name(relation)

    # Update the tolerance attribute if supported.
    if attr is not None:
        setattr(
            relation,
            attr,
            float(epsilon)
        )


def apply_epsilon_to_relations(
    relation_registry,
    epsilon,
):
    """
    Apply a single epsilon value to all registered relations.

    Parameters
    ----------
    relation_registry : object
        Registry containing all metamorphic relations.
    epsilon : float
        Tolerance value to apply.
    """

    # Update every registered relation.
    for name in relation_registry.list():

        relation = relation_registry.get(name)

        set_relation_epsilon(
            relation,
            epsilon
        )


def filter_failed_results(df):
    """
    Return only failed metamorphic test results.

    Parameters
    ----------
    df : pandas.DataFrame
        AutoMR execution results.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing only failed tests.
    """

    # Nothing to filter.
    if df.empty:
        return df

    # Keep only failed rows.
    return df[df["passed"] == False].copy()


def first_failure(df):
    """
    Determine whether any metamorphic test has failed.

    Parameters
    ----------
    df : pandas.DataFrame
        AutoMR execution results.

    Returns
    -------
    bool
        True if at least one failure exists.
    """

    # Empty datasets contain no failures.
    if df.empty:
        return False

    # Check for at least one failed test.
    return (~df["passed"]).any()


def failure_rate(df):
    """
    Calculate the overall failure rate.

    Parameters
    ----------
    df : pandas.DataFrame
        AutoMR execution results.

    Returns
    -------
    float
        Fraction of failed tests.
    """

    # Avoid division on empty datasets.
    if df.empty:
        return 0.0

    # Compute the overall failure rate.
    return 1.0 - df["passed"].mean()