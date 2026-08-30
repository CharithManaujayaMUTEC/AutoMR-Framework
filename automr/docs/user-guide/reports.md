# Reports

## Overview

AutoMR automatically generates multiple reports after execution.

The framework also includes a graph module (`GraphGenerator`) that saves plot-based analysis outputs automatically.

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

# Generated Graphs

Graph outputs are saved under:

```
results/graphs/
```

Key graph files include:

- overall/pass_fail_pie.png
- overall/prediction_distribution.png
- overall/difference_distribution.png
- overall/failure_heatmap.png
- overall/worst_cases.png
- overall/summary_dashboard.png
- summary/failure_rate.png
- summary/severity.png
- summary/range_analysis.png
- summary/epsilon_curve.png

Graph-ready CSV exports include:

- summary/failure_rate.csv
- summary/severity.csv
- summary/epsilon_curve.csv
- <mr_name>/parameter_vs_prediction.csv
- <mr_name>/testcase_vs_prediction.csv

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
- PNG

---

# Output Directory

```
results/
```

can be changed using

```python
output_dir="my_results"
```