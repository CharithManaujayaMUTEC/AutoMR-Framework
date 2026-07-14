# examples/live_dashboard_example.py

"""
Example: AutoMR Live Dashboard

Run:

    python examples/live_dashboard_example.py

This launches the interactive OpenCV dashboard.

Live dashboard:
    - Tests one transformation intensity per frame.
    - Use the Intensity slider to change the parameter.
    - Press R to run the full benchmark sweep.

"""

import sys

sys.path.append("D:/FYP 78SEm/Modals")

from load_model import get_model
from run_dashboard import RealModel

from automr.api import AutoMR
from automr.dashboard import run_live_dashboard


model = RealModel()

automr = AutoMR(
    model=model,
    task="regression",
    input_type="image",
    range_threshold=5.0,
)

run_live_dashboard(
    automr=automr,
    model=model,
    video_source=0, #0 for default camera, or path to video file
    frame_skip=30, #if needed only, to skip frames for faster processing
    save_results=True,
    save_violations=True,
    output_dir="results/live_dashboard"
)