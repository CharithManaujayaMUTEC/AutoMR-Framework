# automr/dashboard/dashboard_utils.py

import os
import cv2
import pandas as pd


def create_output_dirs(output_dir):

    os.makedirs(output_dir, exist_ok=True)

    os.makedirs(
        os.path.join(
            output_dir,
            "violations"
        ),
        exist_ok=True
    )


def calculate_percent_change(
    original,
    transformed
):

    diff = abs(
        transformed - original
    )

    pct = (
        diff /
        (abs(original) + 1e-6)
    ) * 100

    return diff, pct


def get_severity(diff):

    if diff < 0.001:
        return "LOW"

    elif diff < 0.01:
        return "MEDIUM"

    else:
        return "HIGH"


def evaluate_mr(
    automr,
    mr_name,
    original_pred,
    transformed_pred
):

    relation = (
        automr.relation_registry.get(
            mr_name
        )
    )

    try:

        passed = relation.check(
            original_pred,
            transformed_pred
        )

    except Exception:

        passed = False

    return (
        "PASS"
        if passed
        else "FAIL"
    )


def save_violation_image(
    output_dir,
    mr_name,
    frame_id,
    image
):

    filename = os.path.join(
        output_dir,
        "violations",
        f"{mr_name}_{frame_id}.jpg"
    )

    cv2.imwrite(
        filename,
        image
    )


def save_results_csv(
    results,
    output_dir
):

    if len(results) == 0:
        return

    csv_path = os.path.join(
        output_dir,
        "webcam_results.csv"
    )

    pd.DataFrame(
        results
    ).to_csv(
        csv_path,
        index=False
    )


def update_summary(
    results,
    output_dir
):

    if len(results) == 0:
        return

    df = pd.DataFrame(results)

    summary = (
        df.groupby("mr")
        .agg(
            tests=("mr", "count"),
            failures=(
                "status",
                lambda x:
                (x == "FAIL").sum()
            )
        )
        .reset_index()
    )

    summary["failure_rate"] = (
        summary["failures"]
        /
        summary["tests"]
    ) * 100

    summary.to_csv(
        os.path.join(
            output_dir,
            "dashboard_summary.csv"
        ),
        index=False
    )