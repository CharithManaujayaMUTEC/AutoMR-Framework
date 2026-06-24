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

from .control_panel import DashboardConfig
from .graph_panel import draw_summary_panel

class LiveDashboard:

    def __init__(
        self,
        automr,
        model,
        frame_skip=30,
        save_results=True,
        save_violations=True,
        output_dir="results/live_dashboard"
    ):

        self.automr = automr
        self.model = model

        self.config = DashboardConfig(
            automr
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

        settings = (
            self.config
            .mr_ranges[mr_name]
        )

        return np.linspace(
            settings["start"],
            settings["end"],
            settings["tests"]
        )
    
    def update_controls(self):

        tests = cv2.getTrackbarPos(
            "Tests",
            "AutoMR Live Dashboard"
        )

        frame_skip = cv2.getTrackbarPos(
            "FrameSkip",
            "AutoMR Live Dashboard"
        )

        range_scale = cv2.getTrackbarPos(
            "Range %",
            "AutoMR Live Dashboard"
        )

        mr_index = cv2.getTrackbarPos(
            "MR Index",
            "AutoMR Live Dashboard"
        )

        enabled = cv2.getTrackbarPos(
            "Enable",
            "AutoMR Live Dashboard"
        )

        mrs = [
            mr
            for mr in self.automr.list_transforms()
            if mr != "temporal"
        ]

        self.config.current_mr = mrs[mr_index]

        current = self.config.current_mr

        if enabled == 1:

            if current not in self.config.selected_mrs:
                self.config.selected_mrs.append(current)

        else:

            if current in self.config.selected_mrs:
                self.config.selected_mrs.remove(current)

        self.frame_skip = max(
            frame_skip,
            1
        )

        current = (
            self.config.current_mr
        )

        start, end = (
            self.automr.mr_ranges[current]
        )

        scaled_end = (
            start +
            ((end - start) *
            range_scale / 100.0)
        )

        self.config.mr_ranges[
            current
        ]["start"] = start

        self.config.mr_ranges[
            current
        ]["end"] = scaled_end

        self.config.mr_ranges[
            current
        ]["tests"] = max(
            tests,
            1
        )

    def handle_keys(self, key):

        mapping = {
            ord("1"): "brightness",
            ord("2"): "rotation",
            ord("3"): "translation",
            ord("4"): "noise",
            ord("5"): "blur",
            ord("6"): "contrast",
            ord("7"): "rain",
            ord("8"): "snow",
            ord("9"): "fog",
            ord("v"): "visibility",
            ord("d"): "darkness"
        }

        if key in mapping:

            mr = mapping[key]

            self.config.current_mr = mr

            if mr in self.config.selected_mrs:

                self.config.selected_mrs.remove(
                    mr
                )

            else:

                self.config.selected_mrs.append(
                    mr
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

        if frame_id % self.frame_skip != 0:
            return tiles

        for mr_name in self.config.selected_mrs:

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

                    worst_diff = -1

                    if diff > worst_diff:
                        worst_diff = diff
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

        if len(rows) == 0:

            return np.zeros(
                (
                    self.CELL_H,
                    self.CELL_W,
                    3
                ),
                dtype=np.uint8
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
                f"Cannot open video source: {video_source}"
            )

        frame_id = 0

        cv2.namedWindow(
            "AutoMR Live Dashboard"
        )

        cv2.createTrackbar(
            "MR Index",
            "AutoMR Live Dashboard",
            0,
            len([
                mr
                for mr in self.automr.list_transforms()
                if mr != "temporal"
            ]) - 1,
            lambda x: None
        )

        cv2.createTrackbar(
            "Enable",
            "AutoMR Live Dashboard",
            1,
            1,
            lambda x: None
        )

        cv2.createTrackbar(
            "Tests",
            "AutoMR Live Dashboard",
            5,
            100,
            lambda x: None
        )

        cv2.createTrackbar(
            "FrameSkip",
            "AutoMR Live Dashboard",
            30,
            100,
            lambda x: None
        )

        cv2.createTrackbar(
            "Range %",
            "AutoMR Live Dashboard",
            50,
            100,
            lambda x: None
        )

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            self.update_controls()

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

            failure_rate = 0

            if self.total_tests > 0:

                failure_rate = (
                    self.total_failures /
                    self.total_tests
                ) * 100

            summary_panel = draw_summary_panel(
                400,
                dashboard.shape[0],
                self.total_tests,
                self.total_failures,
                failure_rate
            )

            dashboard = np.hstack([
                dashboard,
                summary_panel
            ])

            cv2.putText(
                dashboard,
                "1-9/V/D Toggle MRs",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                1
            )

            cv2.putText(
                dashboard,
                f"Active: {len(self.config.selected_mrs)}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

            cv2.putText(
                dashboard,
                f"Current MR: {self.config.current_mr}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,0),
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

            #self.handle_keys(key)

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