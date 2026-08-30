# Comparators

## Overview

Comparators determine whether the predictions produced by the original and transformed inputs satisfy the expected metamorphic relation.

They provide a task-independent verification layer that enables AutoMR to support different machine learning problem types while maintaining a consistent testing workflow.

---

# Responsibilities

The comparator is responsible for:

- Comparing original and transformed predictions
- Applying epsilon tolerance
- Determining pass or fail status
- Calculating prediction differences
- Returning standardized verification results

---

# Comparator Workflow

```
Original Prediction
         │
         ▼
    Comparator
         ▲
         │
Transformed Prediction
         │
         ▼
Difference Calculation
         │
         ▼
PASS / FAIL
```

---

# Supported Tasks

Current comparator implementations support:

- Regression
- Classification
- Binary Classification

Additional comparators can be implemented for other prediction tasks.

---

# Epsilon

Regression tasks use an epsilon tolerance to determine whether prediction differences are acceptable.

Example:

```python
AutoMR(
    model=model,
    task="regression",
    epsilon=0.05
)
```

---

# Comparator Selection

AutoMR automatically selects the appropriate comparator during initialization.

```python
from automr.comparators import get_comparator

comparator = get_comparator(
    task="regression",
    epsilon=0.05
)
```

---

# Extending Comparators

Custom comparators should implement a consistent comparison interface and return standardized verification results compatible with the AutoMR analysis pipeline.

---

# Best Practices

- Keep comparisons deterministic.
- Avoid framework-specific logic.
- Return consistent output formats.
- Support configurable tolerance values.