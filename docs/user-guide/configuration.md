# Configuration

## Overview

AutoMR behavior can be configured during initialization, dataset testing, HPC execution, and the Live Dashboard.

---

# AutoMR Parameters

```python
automr = AutoMR(
    model=model,
    task="regression",
    input_type="image",
    epsilon=0.05,
    range_threshold=5.0
)
```

---

# Configuration Options

| Parameter | Description |
|-----------|-------------|
| model | Target model |
| task | Prediction task |
| input_type | Input handler |
| epsilon | Comparison tolerance |
| range_threshold | Maximum acceptable deviation |

---

# Dataset Testing Parameters

```python
automr.run_full_test(
    dataset=dataset,
    max_samples=100,
    samples_per_mr=5
)
```

| Parameter | Description |
|-----------|-------------|
| max_samples | Maximum dataset samples |
| samples_per_mr | Number of parameter values evaluated for each MR |

---

# Live Dashboard Controls

The Live Dashboard provides interactive controls for configuring testing in real time.

| Control | Description |
|----------|-------------|
| MR Index | Select the active metamorphic relation |
| Enable | Enable or disable the selected MR |
| Intensity % | Current transformation intensity |
| Range % | Maximum transformation range |
| FrameSkip | Test every N frames |
| Epsilon | Metamorphic relation tolerance |

Unlike benchmark mode, the Live Dashboard evaluates only one parameter per frame.

Press **R** at any time to execute the complete benchmark sweep.

---

# Camera Sources

The dashboard supports multiple video sources.

USB webcam

```python
video_source=0
```

Video file

```python
video_source="video.mp4"
```

Android IP Webcam

```python
from automr.dashboard import CameraSource

video_source = CameraSource.ip_webcam(
    "192.168.1.20"
)
```

RTSP camera

```python
from automr.dashboard import CameraSource

video_source = CameraSource.rtsp(
    "192.168.1.50:554/stream"
)
```

---

# HPC Parameters

```python
HighPerformanceAutoMR(
    batch_size=64,
    num_workers=8
)
```

---

# Epsilon Analysis

```python
automr.run_full_test(
    epsilon_min=0.01,
    epsilon_max=0.10,
    epsilon_count=5
)
```

---

# Output Directory

```python
automr.run_full_test(
    output_dir="results"
)
```

---

# Recommended Settings

| Dataset Size | Batch Size |
|--------------|-----------|
| Small | 16 |
| Medium | 32 |
| Large | 64 |
| HPC GPU | 64–256 |