"""
Run the AutoMR Live Dashboard with a mock regression model.

This mirrors run_live_dashboard_example.py but replaces the external
model (D:/FYP 78SEm/Modals) with a self-contained dummy model, so the
dashboard can be launched without any external dependencies.

Run:
    python run_live_dashboard_mock.py
"""

import numpy as np

from automr.api import AutoMR
from automr.dashboard import run_live_dashboard


class MockModel:
    """Dummy regression model: predicts mean pixel brightness (0-255)."""

    def predict(self, x):
        return float(np.asarray(x).mean())

    def predict_batch(self, xs):
        return [self.predict(x) for x in xs]


model = MockModel()

automr = AutoMR(
    model=model,
    task="regression",
    input_type="image",
    range_threshold=5.0,
)

video_source = 0  # USB webcam

run_live_dashboard(
    automr=automr,
    model=model,
    video_source=video_source,
    frame_skip=30,
    save_results=True,
    save_violations=True,
    output_dir="results/live_dashboard",
)
