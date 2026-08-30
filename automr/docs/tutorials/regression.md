# Tutorial: Regression Models

## Overview

This tutorial demonstrates how to evaluate regression models using AutoMR.

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

# Step 3: Initialize AutoMR

```python
automr = AutoMR(
    model=model,
    task="regression",
    input_type="image",
    epsilon=0.05
)
```

---

# Step 4: Load Dataset

```python
dataset = RegressionDataset("dataset/")
```

---

# Step 5: Run Testing

```python
df, results = automr.run_full_test(
    dataset=dataset
)
```

---

# Optional Epsilon Analysis

```python
automr.run_full_test(
    dataset=dataset,
    epsilon_min=0.005,
    epsilon_max=0.10,
    epsilon_count=5
)
```

---

# Output

Review the generated reports to evaluate prediction stability and metamorphic relation violations.