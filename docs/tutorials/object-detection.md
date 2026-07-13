# Tutorial: Object Detection

## Overview

This tutorial outlines the general workflow for evaluating object detection models using AutoMR.

---

# Workflow

1. Load the detection model.
2. Create an AutoMR instance.
3. Load the dataset.
4. Execute metamorphic testing.
5. Analyze generated reports.

---

# Example

```python
automr = AutoMR(
    model=model,
    task="detection",
    input_type="image"
)

df, results = automr.run_full_test(
    dataset=dataset
)
```

---

# Evaluation

Generated reports provide:

- Failure rates
- Severity scores
- Prediction traces
- Range analysis

---

# Notes

Support for object detection depends on the availability of an appropriate model wrapper and comparator implementation.