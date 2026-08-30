# Running Tests

## Overview

AutoMR supports:

- Interactive live testing
- Dataset testing
- High Performance Computing (HPC) execution

Each mode is designed for a different workflow.

---

# Live Dashboard

Launch the interactive dashboard.

```python
from automr.dashboard import run_live_dashboard

run_live_dashboard(
    automr=automr,
    model=model,
    video_source=0
)
```

The Live Dashboard performs:

- One metamorphic relation evaluation per frame
- One transformation intensity per frame
- Real-time visualization of predictions
- Interactive control of testing parameters

Press **R** to execute a complete benchmark sweep.

---

# Running a Dataset Test

```python
df, results = automr.run_full_test(
    dataset=dataset
)
```

Dataset testing evaluates every configured parameter for every enabled metamorphic relation.

---

# Limiting Dataset Size

```python
df, results = automr.run_full_test(
    dataset=dataset,
    max_samples=500
)
```

---

# Controlling Transformations

```python
df, results = automr.run_full_test(
    dataset=dataset,
    samples_per_mr=5
)
```

`samples_per_mr` controls how many parameter values are evaluated for each metamorphic relation during benchmark execution.

---

# Epsilon Sensitivity Analysis

```python
df, results = automr.run_full_test(
    dataset=dataset,
    epsilon_min=0.005,
    epsilon_max=0.10,
    epsilon_count=5
)
```

---

# HPC Execution

```python
from automr.hpc import HighPerformanceAutoMR

automr = HighPerformanceAutoMR(
    model=model,
    task="regression",
    input_type="image"
)
```

---

# Camera Sources

The Live Dashboard supports multiple video sources.

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

# Output Directory

```python
automr.run_full_test(
    dataset=dataset,
    output_dir="results"
)
```

---

# Viewing Results

Generated reports are automatically saved to the specified output directory.

The Live Dashboard also provides immediate visual feedback while testing, while benchmark mode produces detailed CSV reports for offline analysis.