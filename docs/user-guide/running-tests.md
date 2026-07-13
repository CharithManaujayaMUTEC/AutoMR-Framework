# Running Tests

## Overview

AutoMR supports testing individual samples, datasets, and large-scale evaluations using the High Performance Computing (HPC) engine.

---

# Running a Dataset Test

```python
df, results = automr.run_full_test(
    dataset=dataset
)
```

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

`samples_per_mr` controls how many parameter values are evaluated for each metamorphic relation.

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

# Output Directory

```python
automr.run_full_test(
    dataset=dataset,
    output_dir="results"
)
```

---

# Viewing Results

Generated reports are saved automatically in the specified output directory.