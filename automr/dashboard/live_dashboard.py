import cv2
import numpy as np

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

    def get_test_parameters(
        self,
        mr_name
    ):

        if mr_name in self.custom_ranges:

            start, end, num_tests = (
                self.custom_ranges[mr_name]
            )

        else:

            start, end = (
                self.automr.mr_ranges[mr_name]
            )

            num_tests = 5

        return np.linspace(
            start,
            end,
            num_tests
        )

    def process_frame(
        self,
        frame,
        frame_id
    ):

        tiles = []

        original_pred = float(
            self.model.predict(frame)
        )

        original_tile = frame.copy()

        cv2.putText(
            original_tile,
            f"ORIGINAL {original_pred:.4f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        tiles.append(
            original_tile
        )

        for mr_name in self.selected_mrs:

            try:

                transform = (
                    self.automr
                    .transform_registry
                    .get(mr_name)
                )

                parameters = (
                    self.get_test_parameters(
                        mr_name
                    )
                )

                best_tile = None

                for param in parameters:

                    transformed = transform(
                        frame.copy(),
                        float(param)
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
                                frame_id,
                                transformed
                            )

                    self.results.append({

                        "timestamp":
                            datetime.now(),

                        "frame_id":
                            frame_id,

                        "mr":
                            mr_name,

                        "parameter":
                            float(param),

                        "original_prediction":
                            original_pred,

                        "transformed_prediction":
                            transformed_pred,

                        "difference":
                            diff,

                        "percent_change":
                            pct,

                        "status":
                            status,

                        "severity":
                            severity
                    })

                    best_tile = transformed

                if best_tile is not None:

                    cv2.putText(
                        best_tile,
                        mr_name,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

                    tiles.append(
                        best_tile
                    )

            except Exception as e:

                print(
                    f"{mr_name}: {e}"
                )

        return tiles

    def build_dashboard(
        self,
        tiles
    ):

        resized = []

        for tile in tiles:

            resized.append(
                cv2.resize(
                    tile,
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

        frame_id = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_id += 1

            tiles = self.process_frame(
                frame,
                frame_id
            )

            dashboard = (
                self.build_dashboard(
                    tiles
                )
            )

            cv2.putText(
                dashboard,
                f"Tests:{self.total_tests}  Fails:{self.total_failures}",
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
                frame_id %
                self.frame_skip == 0
            ):

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

        save_results_csv(
            self.results,
            self.output_dir
        )

        update_summary(
            self.results,
            self.output_dir
        )