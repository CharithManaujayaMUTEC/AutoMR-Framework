# Getting Started

## Overview

AutoMR is an automated metamorphic testing framework designed to evaluate the robustness and reliability of machine learning and deep learning models using metamorphic testing.

This guide introduces the basic workflow for using the framework.

---

# Workflow

The typical AutoMR workflow consists of the following steps:

1. Load a trained model.
2. Create an AutoMR instance.
3. Load a dataset.
4. Execute metamorphic testing.
5. Review generated reports.

---

# Example

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

# Execute Testing

```python
df, results = automr.run_full_test(
    dataset=dataset
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