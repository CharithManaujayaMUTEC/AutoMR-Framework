# Tutorial: Lane Detection

## Overview

This tutorial demonstrates how to evaluate a lane detection model using AutoMR.

---

# Step 1: Load the Model

```python
model = load_model()
```

---

# Step 2: Register Preprocessing

```python
model._automr_preprocess = tusimple_preprocess
model._automr_decoder = lane_decoder
```

---

# Step 3: Create AutoMR

```python
automr = HighPerformanceAutoMR(
    model=model,
    task="regression",
    input_type="image"
)
```

---

# Step 4: Load Dataset

```python
dataset = TuSimpleDataset(DATASET_ROOT)
```

---

# Step 5: Execute Testing

```python
df, results = automr.run_full_test(
    dataset=dataset
)
```

---

# Generated Reports

- Failure summary
- Severity summary
- Prediction trace
- Range analysis
- Epsilon analysis

---

# Notes

This workflow is suitable for evaluating lane detection models using the TuSimple dataset and the HPC execution engine.