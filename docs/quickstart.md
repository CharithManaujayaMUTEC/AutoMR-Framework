# Quick Start

## Create an AutoMR Instance

```python
from automr import AutoMR

automr = AutoMR(
    model=model,
    task="regression",
    input_type="image",
    epsilon=0.05,
    range_threshold=5.0,
    transform_ranges={
        "brightness": {"start": 0.5, "end": 2.0, "samples": 8},
        "rotation": {"start": -25, "end": 25, "samples": 11},
    },
)
```

---

## Load a Dataset

```python
dataset = MyDataset("dataset/")
```

---

## Run Metamorphic Testing

```python
df, results = automr.run_full_test(
    dataset=dataset,
    samples_per_mr=5,
    epsilon_min=0.005,
    epsilon_max=0.05,
    epsilon_count=3,
)
```

---

## Launch the Live Dashboard

```python
from automr.dashboard import run_live_dashboard

run_live_dashboard(
    automr=automr,
    model=model,
    video_source=0
)
```

The Live Dashboard performs interactive metamorphic testing on a live video stream.

### Dashboard Controls

| Control | Description |
|----------|-------------|
| MR Index | Select the active metamorphic relation |
| Enable | Enable or disable the selected MR |
| Intensity % | Current transformation intensity |
| Range % | Maximum transformation range |
| FrameSkip | Test every N frames |
| Epsilon | Metamorphic relation tolerance |
| R | Run a complete benchmark sweep |
| ESC | Exit the dashboard |

---

## Using Different Camera Sources

USB webcam

```python
video_source=0
```

Video file

```python
video_source="road.mp4"
```

Android IP Webcam

```python
from automr.dashboard import CameraSource

video_source = CameraSource.ip_webcam(
    "192.168.1.20"
)
```

RTSP stream

```python
from automr.dashboard import CameraSource

video_source = CameraSource.rtsp(
    "192.168.1.50:554/stream"
)
```

---

## Generated Outputs

The framework automatically produces:

- automr_results.csv
- failure_summary.csv
- severity_summary.csv
- prediction_trace.csv
- range_summary.csv
- range_analysis.csv

Optional outputs include:

- epsilon_summary.csv
- epsilon_report.txt

If you omit transform_ranges, AutoMR uses built-in default ranges for all registered relations.

---

## Next Steps

- Configure custom transformations.
- Register custom metamorphic relations.
- Explore HPC execution for large datasets.
- Use the Live Dashboard for real-time testing.
- Review the tutorials for end-to-end examples.