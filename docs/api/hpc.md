# HighPerformanceAutoMR API

## Overview

`HighPerformanceAutoMR` extends the standard AutoMR engine with optimized execution for large datasets.

---

# Import

```python
from automr.hpc import HighPerformanceAutoMR
```

---

# Constructor

```python
HighPerformanceAutoMR(
    model,
    task="regression",
    input_type="image",
    epsilon=0.05,
    range_threshold=5.0,
    batch_size=64,
    num_workers=8
)
```

---

# Additional Parameters

| Parameter | Description |
|------------|-------------|
| batch_size | GPU inference batch size |
| num_workers | Parallel worker threads |

---

# Main Methods

## run_full_test()

Runs the optimized AutoMR pipeline.

```python
df, results = automr.run_full_test(dataset)
```

---

## run_dataset()

Processes an entire dataset using the HPC engine.

---

## save_baseline()

Generates or reuses cached baseline predictions.

---

## save_results()

Exports all generated reports.

---

# Features

- Parallel execution
- GPU acceleration
- Batch inference
- Prediction cache
- Progress monitoring
- Epsilon analysis