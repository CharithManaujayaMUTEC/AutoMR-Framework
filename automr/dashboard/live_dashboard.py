# automr/dashboard/live_dashboard.py

import cv2
import numpy as np
import pandas as pd
from datetime import datetime

from .dashboard_utils import (
    create_output_dirs,
    calculate_percent_change,
    get_severity,
    evaluate_mr,
    save_violation_image,
    save_results_csv,
    update_summary
)


class LiveDashboard:

    def __init__(
        self,
        automr,
        model,
        selected_mrs=None,
        custom_ranges=None,
        frame_skip=30,
        save_results=True,
        save_violations=True,
        output_dir="results/live_dashboard"
    ):

        self.automr = automr
        self.model = model

        self.selected_mrs = (
            selected_mrs
            if selected_mrs is not None
            else [
                mr
                for mr in automr.list_transforms()
                if mr != "temporal"
            ]
        )

        self.custom_ranges = (
            custom_ranges
            if custom_ranges is not None
            else {}
        )

        self.frame_skip = frame_skip
        self.save_results = save_results
        self.save_violations = save_violations
        self.output_dir = output_dir

        self.results = []

        self.total_tests = 0
        self.total_failures = 0

        self.CELL_W = 320
        self.CELL_H = 240

        create_output_dirs(
            self.output_dir
        )

    def get_parameter(
        self,
        mr_name
    ):

        if mr_name in self.custom_ranges:
            return self.custom_ranges[mr_name]

        start, end = (
            self.automr
            .mr_ranges[mr_name]
        )

        return (
            start + end
        ) / 2

    def get_failure_rate(self):

        if self.total_tests == 0:
            return 0.0

        return (
            self.total_failures
            /
            self.total_tests
        ) * 100

    def process_frame(
        self,
        frame,
        frame_count
    ):

        current_tiles = []

        original_pred = float(
            self.model.predict(frame)
        )

        # ---------------------------
        # ORIGINAL TILE
        # ---------------------------
        original_tile = frame.copy()

        cv2.putText(
            original_tile,
            "ORIGINAL",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            original_tile,
            f"{original_pred:.4f}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        current_tiles.append(
            original_tile
        )

        # ---------------------------
        # TRANSFORMATIONS
        # ---------------------------
        for mr_name in self.selected_mrs:

            try:

                transform = (
                    self.automr
                    .transform_registry
                    .get(mr_name)
                )

                param = self.get_parameter(
                    mr_name
                )

                transformed = transform(
                    frame.copy(),
                    param
                )

                transformed_pred = float(
                    self.model.predict(
                        transformed
                    )
                )

                diff, pct = (
                    calculate_percent_change(
                        original_pred,
                        transformed_pred
                    )
                )

                status = evaluate_mr(
                    self.automr,
                    mr_name,
                    original_pred,
                    transformed_pred
                )

                severity = get_severity(
                    diff
                )

                self.total_tests += 1

                if status == "FAIL":
                    self.total_failures += 1

                    if self.save_violations:

                        save_violation_image(
                            self.output_dir,
                            mr_name,
                            frame_count,
                            transformed
                        )

                self.results.append({

                    "timestamp":
                        datetime.now(),

                    "frame_id":
                        frame_count,

                    "mr":
                        mr_name,

                    "parameter":
                        float(param),

                    "original_prediction":
                        float(original_pred),

                    "transformed_prediction":
                        float(transformed_pred),

                    "difference":
                        float(diff),

                    "percent_change":
                        float(pct),

                    "status":
                        status,

                    "severity":
                        severity
                })

                cv2.putText(
                    transformed,
                    mr_name,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    transformed,
                    f"{transformed_pred:.4f}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    transformed,
                    status,
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (
                        (0, 255, 0)
                        if status == "PASS"
                        else (0, 0, 255)
                    ),
                    2
                )

                current_tiles.append(
                    transformed
                )

            except Exception as e:
                print(
                    f"{mr_name} failed: {e}"
                )

        return current_tiles

    def build_dashboard(
        self,
        tiles
    ):

        resized = []

        for img in tiles:

            resized.append(
                cv2.resize(
                    img,
                    (
                        self.CELL_W,
                        self.CELL_H
                    )
                )
            )

        rows = []

        for i in range(
            0,
            len(resized),
            3
        ):

            row = resized[i:i + 3]

            while len(row) < 3:

                row.append(
                    np.zeros(
                        (
                            self.CELL_H,
                            self.CELL_W,
                            3
                        ),
                        dtype=np.uint8
                    )
                )

            rows.append(
                np.hstack(row)
            )

        return np.vstack(rows)

    def run(
        self,
        video_source=0
    ):

        cap = cv2.VideoCapture(
            video_source
        )

        if not cap.isOpened():

            raise RuntimeError(
                f"Cannot open source: "
                f"{video_source}"
            )

        frame_count = 0

        print("Dashboard started")
        print("ESC = Exit")

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            tiles = self.process_frame(
                frame,
                frame_count
            )

            dashboard = (
                self.build_dashboard(
                    tiles
                )
            )

            failure_rate = (
                self.get_failure_rate()
            )

            cv2.putText(
                dashboard,
                (
                    f"Tests:"
                    f"{self.total_tests} "
                    f"Fails:"
                    f"{self.total_failures} "
                    f"Rate:"
                    f"{failure_rate:.2f}%"
                ),
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "AutoMR Live Dashboard",
                dashboard
            )

            if (
                frame_count
                %
                self.frame_skip
                == 0
            ):

                if self.save_results:

                    save_results_csv(
                        self.results,
                        self.output_dir
                    )

                    update_summary(
                        self.results,
                        self.output_dir
                    )

            key = cv2.waitKey(1)

            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        if self.save_results:

            save_results_csv(
                self.results,
                self.output_dir
            )

            update_summary(
                self.results,
                self.output_dir
            )

        print("Dashboard stopped")