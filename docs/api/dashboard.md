# Dashboard API

## Overview

The AutoMR Dashboard provides interactive real-time metamorphic testing using live camera streams, video files, or IP cameras.

Unlike offline testing, the dashboard evaluates one transformation parameter per frame, allowing users to observe prediction changes immediately while adjusting testing parameters.

---

# Import

```python
from automr.dashboard import (
    run_live_dashboard,
    CameraSource
)
```

---

# Running the Dashboard

```python
run_live_dashboard(
    automr=automr,
    model=model,
    video_source=0
)
```

---

# Parameters

| Parameter | Description |
|------------|-------------|
| automr | AutoMR instance |
| model | Wrapped prediction model |
| video_source | Camera, video file, or stream |
| frame_skip | Number of frames between tests |
| save_results | Save CSV reports |
| save_violations | Save failed images |
| output_dir | Output directory |

---

# Supported Video Sources

## USB Camera

```python
video_source = 0
```

---

## Video File

```python
video_source = "road.mp4"
```

---

## Android IP Webcam

```python
video_source = CameraSource.ip_webcam(
    "192.168.1.20"
)
```

---

## RTSP Camera

```python
video_source = CameraSource.rtsp(
    "192.168.1.50:554/stream"
)
```

---

# Dashboard Controls

| Control | Description |
|----------|-------------|
| MR Index | Select active metamorphic relation |
| Enable | Enable or disable selected MR |
| Intensity % | Current transformation intensity |
| Range % | Maximum transformation range |
| FrameSkip | Number of frames between tests |
| Epsilon | Metamorphic tolerance |
| R | Run complete benchmark |
| ESC | Exit |

---

# Live Mode

During live execution the dashboard performs:

```
Frame
    ↓
Current MR
    ↓
Current Intensity
    ↓
Transformation
    ↓
Prediction
    ↓
Relation Check
    ↓
Dashboard Update
```

Only one parameter is evaluated per frame.

---

# Benchmark Mode

Press **R** to execute the complete benchmark.

Benchmark mode evaluates

```
MR
    ↓
Parameter Sweep
    ↓
Prediction
    ↓
Relation Verification
    ↓
CSV Reports
```

This mode uses the configured Start, End and Tests values.

---

# Generated Outputs

The dashboard automatically generates:

- automr_results.csv
- failure_summary.csv
- prediction_trace.csv

Optionally:

- violation images
- summary reports

---

# CameraSource

The CameraSource helper creates compatible OpenCV video sources.

```python
CameraSource.usb(0)

CameraSource.video("road.mp4")

CameraSource.ip_webcam("192.168.1.20")

CameraSource.rtsp("192.168.1.50:554/stream")
```