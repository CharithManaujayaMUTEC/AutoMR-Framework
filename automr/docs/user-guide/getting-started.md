# Getting Started

## Overview

AutoMR is an automated metamorphic testing framework designed to evaluate the robustness and reliability of machine learning and deep learning models using metamorphic testing.

The framework supports both:

- Offline dataset testing
- Interactive live dashboard testing

---

# Workflow

The typical AutoMR workflow consists of the following steps:

1. Load a trained model.
2. Create an AutoMR instance.
3. Load a dataset.
4. Execute metamorphic testing.
5. Review generated reports.

Alternatively, launch the Live Dashboard for real-time testing.

---

# Create an AutoMR Instance

```python
from automr import AutoMR

automr = AutoMR(
    model=model,
    task="classification",
    input_type="image"
)
```

---

# Load a Dataset

```python
dataset = MyDataset("dataset/")
```

---

# Execute Dataset Testing

```python
df, results = automr.run_full_test(
    dataset=dataset
)
```

Dataset testing performs a complete parameter sweep for every enabled metamorphic relation.

---

# Launch the Live Dashboard

```python
from automr.dashboard import run_live_dashboard

run_live_dashboard(
    automr=automr,
    model=model,
    video_source=0
)
```

The Live Dashboard evaluates one transformation intensity per frame, allowing real-time interaction with metamorphic relations.

The benchmark mode can still be executed at any time by pressing **R**, which performs the complete parameter sweep.

---

# Supported Camera Sources

The dashboard supports:

- USB webcams
- Video files
- Android IP Webcam streams
- RTSP camera streams

Example:

```python
from automr.dashboard import CameraSource

run_live_dashboard(
    automr=automr,
    model=model,
    video_source=CameraSource.ip_webcam("192.168.1.20")
)
```

---

# Generated Reports

After execution, AutoMR generates multiple reports including:

- automr_results.csv
- failure_summary.csv
- severity_summary.csv
- prediction_trace.csv
- range_summary.csv
- range_analysis.csv

---

# Next Steps

Continue with:

- Running Tests
- Transformations
- Metamorphic Relations
- Configuration