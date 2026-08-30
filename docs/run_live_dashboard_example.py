# run_live_dashboard_example.py

"""
Example: AutoMR Live Dashboard

Run:

    python run_live_dashboard_example.py

This launches the interactive OpenCV dashboard.

Live dashboard:
    - Tests one transformation intensity per frame.
    - Use the Intensity slider to change the parameter.
    - Press R to run the full benchmark sweep.

Supported video sources:

    USB Webcam
        video_source=0

    Video File
        video_source="road.mp4"

    Android IP Webcam
        video_source=CameraSource.ip_webcam("192.168.1.20")

    RTSP Camera
        video_source=CameraSource.rtsp("192.168.1.50:554/stream")
"""

import sys

sys.path.append("D:/FYP 78SEm/Modals")

from load_model import get_model
from run_dashboard import RealModel

from automr.api import AutoMR
from automr.dashboard import (
    run_live_dashboard,
    CameraSource,
)

model = RealModel()

automr = AutoMR(
    model=model,
    task="regression",
    input_type="image",
    range_threshold=5.0,
)

# ------------------------------------------------------------------
# Select ONE video source
# ------------------------------------------------------------------

# Option 1 - USB webcam
video_source = 0

# Option 2 - Video file
# video_source = "videos/road.mp4"

# Option 3 - Android IP Webcam
# 1. Install "IP Webcam" from Google Play.
# 2. Start Server.
# 3. Note the displayed IP address (e.g. 192.168.1.20:8080).
# 4. Ensure your PC and phone are on the same Wi-Fi network.
#
# video_source = CameraSource.ip_webcam(
#     "192.168.1.20:8080"
# )

# Option 4 - RTSP camera
# video_source = CameraSource.rtsp(
#     "192.168.1.50:554/stream"
# )

# ------------------------------------------------------------------

run_live_dashboard(
    automr=automr,
    model=model,
    video_source=video_source,
    frame_skip=30,
    save_results=True,
    save_violations=True,
    output_dir="results/live_dashboard",
)