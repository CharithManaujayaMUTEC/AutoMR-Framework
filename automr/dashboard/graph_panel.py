import cv2
import numpy as np


def draw_summary_panel(
    width,
    height,
    tests,
    failures,
    rate,
    progress=0,
    total=0
):

    panel = np.full(
        (height, width, 3),
        60,
        dtype=np.uint8
    )

    cv2.putText(
        panel,
        f"Tests: {tests}",
        (30,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        panel,
        f"Failures: {failures}",
        (30,140),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2
    )

    cv2.putText(
        panel,
        f"Rate: {rate:.2f}%",
        (30,200),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,100,255),
        2
    )

    if total > 0:

        cv2.rectangle(
            panel,
            (30,260),
            (350,300),
            (100,100,100),
            -1
        )

        w = int(
            (progress / total) * 320
        )

        cv2.rectangle(
            panel,
            (30,260),
            (30+w,300),
            (0,255,0),
            -1
        )

        cv2.putText(
            panel,
            f"{progress}/{total}",
            (120,290),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,0,0),
            2
        )

    return panel