import numpy as np
import pandas as pd


def generate_epsilon_values(
    epsilon_min,
    epsilon_max,
    epsilon_count,
):
    """
    Generate evenly spaced epsilon values.
    """

    return np.linspace(
        float(epsilon_min),
        float(epsilon_max),
        int(epsilon_count)
    )


def relation_attribute_name(relation):
    """
    Returns which attribute controls tolerance.
    """

    if hasattr(relation, "tolerance"):
        return "tolerance"

    if hasattr(relation, "epsilon"):
        return "epsilon"

    if hasattr(relation, "delta"):
        return "delta"

    if hasattr(relation, "max_change"):
        return "max_change"

    return None


def set_relation_epsilon(
    relation,
    epsilon,
):
    """
    Update relation tolerance dynamically.
    """

    attr = relation_attribute_name(relation)

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
    Apply one epsilon to every registered MR.
    """

    for name in relation_registry.list():

        relation = relation_registry.get(name)

        set_relation_epsilon(
            relation,
            epsilon
        )


def filter_failed_results(df):
    """
    Keep only rows that failed.
    """

    if df.empty:
        return df

    return df[df["passed"] == False].copy()


def first_failure(df):
    """
    Returns True if any failure exists.
    """

    if df.empty:
        return False

    return (~df["passed"]).any()


def failure_rate(df):

    if df.empty:
        return 0.0

    return 1.0 - df["passed"].mean()