# Reports

## Overview

AutoMR automatically generates multiple reports after execution.

---

# Generated Files

| File | Description |
|------|-------------|
| automr_results.csv | Complete testing results |
| failure_summary.csv | Failure statistics |
| severity_summary.csv | Severity analysis |
| prediction_trace.csv | Prediction history |
| range_summary.csv | Range statistics |
| range_analysis.csv | Parameter analysis |
| worst_cases.csv | Highest severity failures |
| baseline_metrics.json | Baseline metrics |
| dataset_info.json | Dataset summary |
| model_summary.txt | Model information |
| original_predictions.csv | Baseline predictions |
| epsilon_summary.csv | Epsilon sensitivity |
| epsilon_report.txt | Recommended epsilon |

---

# Failure Summary

Contains:

- Total tests
- Passed tests
- Failed tests
- Failure rate

---

# Severity Report

Provides:

- Average severity
- Maximum severity
- Relation ranking

---

# Prediction Trace

Records:

- Original prediction
- Transformed prediction
- Difference
- Pass/fail status

---

# Export Format

Reports are generated as:

- CSV
- TXT
- JSON

---

# Output Directory

```
results/
```

can be changed using

```python
output_dir="my_results"
```