# Final Evaluation Report

## Overview

`FinalEvaluationReport` consolidates artifacts already produced by an AutoMR
experiment. It is artifact-based and does not rerun model inference or
metamorphic testing.

## Import and Usage

```python
from automr.evaluation import FinalEvaluationReport

report_generator = FinalEvaluationReport(output_dir="results")
report = report_generator.generate_and_save(
    test_summary=test_summary,
    epsilon_summary=epsilon_summary,
    benchmark_results=benchmark_results,
)
```

The constructor defaults to `output_dir="results"`. `generate()` accepts
`test_summary`, `epsilon_summary`, `benchmark_results`, and
`additional_results`. `save()` writes an existing report, and
`generate_and_save()` generates and saves it.

When explicit values are not supplied, `generate()` loads available JSON
artifacts named `test_summary.json`, `baseline_metrics.json`,
`dataset_info.json`, `decoder_health.json`, `epsilon_summary.json`, and
`benchmark_results.json`. It also records other files in the output directory
under `available_artifacts`.

The default output filename is `final_evaluation_report.json`. The report
contains test summary, baseline metrics and dataset information, decoder
health, epsilon analysis, benchmark results, additional analysis, and the
available artifact list when those inputs or files exist.
