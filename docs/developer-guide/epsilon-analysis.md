# Epsilon Analysis

## Overview

AutoMR can automatically evaluate multiple epsilon values to determine an appropriate prediction tolerance for a target model.

Instead of selecting epsilon manually, the framework measures how prediction stability changes across different tolerance values.

---

# Purpose

Epsilon analysis helps to:

- Identify the first failure point.
- Estimate a recommended epsilon.
- Measure prediction robustness.
- Reduce manual parameter tuning.

---

# Running Epsilon Analysis

```python
automr.run_full_test(
    dataset=dataset,
    epsilon_min=0.005,
    epsilon_max=0.10,
    epsilon_count=5
)
```

---

# Process

1. Generate candidate epsilon values.
2. Execute AutoMR for each epsilon.
3. Measure failure rates.
4. Compare stability.
5. Produce summary statistics.

---

# Generated Outputs

- epsilon_summary.csv
- epsilon_report.txt

---

# Report Contents

The generated report includes:

- First failure epsilon
- Recommended epsilon
- Stabilization epsilon
- Maximum failure rate

---

# Recommendations

- Use a sufficiently wide epsilon range.
- Evaluate multiple values.
- Compare results across different datasets.
- Select the recommended epsilon for subsequent experiments.