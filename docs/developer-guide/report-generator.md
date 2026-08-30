# Report Generator

## Overview

After execution, AutoMR automatically generates reports summarizing metamorphic testing results.

These reports support debugging, analysis, and experimental evaluation.

---

# Generated Files

- automr_results.csv
- failure_summary.csv
- severity_summary.csv
- prediction_trace.csv
- range_summary.csv
- range_analysis.csv
- worst_cases.csv
- failure_regions.txt
- baseline_metrics.json
- dataset_info.json
- model_summary.txt
- original_predictions.csv
- epsilon_summary.csv
- epsilon_report.txt

---

# Report Categories

## Prediction Reports

- Original predictions
- Prediction trace

## Failure Reports

- Failure summary
- Failure regions
- Worst cases

## Statistical Reports

- Severity summary
- Range summary
- Range analysis

## Metadata

- Dataset information
- Model summary
- Baseline metrics

---

# Output Directory

Reports are written to the directory specified by:

```python
output_dir="results"
```

---

# Export Formats

- CSV
- JSON
- TXT

---

# Recommendations

- Keep generated reports for reproducibility.
- Archive reports together with experiment configurations.
- Use report summaries when comparing multiple model versions.