# Tutorial: Image Classification

## Overview

This tutorial demonstrates how to evaluate an image classification model using AutoMR.

---

# Step 1: Import AutoMR

```python
from automr import AutoMR
```

---

# Step 2: Load the Model

```python
model = load_model()
```

---

# Step 3: Create AutoMR

```python
automr = AutoMR(
    model=model,
    task="classification",
    input_type="image"
)
```

---

# Step 4: Load Dataset

```python
dataset = ImageDataset("dataset/")
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

- automr_results.csv
- failure_summary.csv
- severity_summary.csv
- prediction_trace.csv

---

# Next Steps

Review the generated reports to identify robustness issues and analyze failure patterns.