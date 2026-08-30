# High-Performance Computing (HPC) Engine

## Overview

The HighPerformanceAutoMR (HPC) engine accelerates metamorphic testing for large datasets by combining parallel execution, batched inference, prediction caching, and GPU acceleration.

It is intended for evaluating thousands of samples efficiently while producing the same reports as the standard AutoMR engine.

---

# Features

- Parallel dataset processing
- Batched model inference
- GPU acceleration (CUDA)
- Prediction caching
- Concurrent image loading
- Automatic report generation
- Epsilon sensitivity analysis

---

# Execution Pipeline

```
Dataset
   │
   ▼
Image Loading
   │
   ▼
Batch Preparation
   │
   ▼
GPU Inference
   │
   ▼
Transformation Execution
   │
   ▼
Relation Verification
   │
   ▼
Analysis
   │
   ▼
Reports
```

---

# Initialization

```python
from automr.hpc import HighPerformanceAutoMR

automr = HighPerformanceAutoMR(
    model=model,
    task="regression",
    input_type="image",
    batch_size=64,
    num_workers=8
)
```

---

# Main Optimizations

- Multi-threaded execution
- Parallel image loading
- Prediction reuse
- Batched inference
- Efficient memory utilization

---

# Generated Reports

The HPC engine generates the same output reports as the standard AutoMR pipeline.

---

# Best Practices

- Use CUDA when available.
- Select an appropriate batch size for available GPU memory.
- Avoid unnecessary data transfers between CPU and GPU.
- Reuse cached baseline predictions whenever possible.