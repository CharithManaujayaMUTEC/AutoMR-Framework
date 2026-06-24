# automr/dashboard/graph_panel.py

import cv2
import numpy as np


def draw_summary_panel(
    width,
    height,
    total_tests,
    failures,
    failure_rate
):

    panel = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    cv2.putText(
        panel,
        f"Tests: {total_tests}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.putText(
        panel,
        f"Failures: {failures}",
        (20,100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )

    cv2.putText(
        panel,
        f"Rate: {failure_rate:.2f}%",
        (20,150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,0,255),
        2
    )

    return panel