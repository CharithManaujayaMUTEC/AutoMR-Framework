# Decoder Health Validation

## Overview

`DecoderHealthAnalyzer` performs warning-only diagnostics on baseline scalar
predictions. It runs during the `run_full_test()` evaluation flow and never
blocks or stops metamorphic testing.

## Import and Analysis

```python
from automr.evaluation import DecoderHealthAnalyzer

analyzer = DecoderHealthAnalyzer()
report = analyzer.analyze(
    predictions=baseline_predictions,
    output_path="results/decoder_health.json",
)
```

The returned report includes prediction count, mean, variance, standard
deviation (`std`), minimum, maximum, range, unique count and ratio, `q01` and
`q99` quantiles, saturation ratios, lower and upper clipping ratios, a
`distribution_diagnostic`, `status`, `warning_only`, and `warnings`.

Distribution diagnostics identify constant, low-cardinality, low-diversity,
or continuous-like predictions. Saturation and clipping checks produce
warnings when applicable. The output file is optional; when requested, the
report is written as JSON to the supplied path.
