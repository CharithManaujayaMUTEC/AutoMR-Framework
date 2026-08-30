# automr/dashboard/control_panel.py

import cv2


class DashboardConfig:

    def __init__(self, automr):

        self.selected_mrs = [
            mr
            for mr in automr.list_transforms()
            if mr != "temporal"
        ]

        self.current_mr = (
            self.selected_mrs[0]
        )

        self.mr_ranges = {}

        for mr in self.selected_mrs:

            cfg = automr.mr_ranges[mr]

            if isinstance(cfg, dict):
                start = cfg["start"]
                end = cfg["end"]
            else:
                start, end = cfg

            self.mr_ranges[mr] = {
                "start": start,
                "end": end,
                "tests": 5
            }

        self.current_mr = self.selected_mrs[0]

        self.frame_skip = 30

        # Live preview intensity (0-100%)
        self.live_intensity = 50

        self.save_results = True

        self.save_violations = True