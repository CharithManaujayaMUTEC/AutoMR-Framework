# AutoMR API

## Overview

The `AutoMR` class is the primary entry point for executing metamorphic testing.

It manages model wrapping, transformation registration, relation verification, analysis, and report generation.

---

# Import

```python
from automr import AutoMR
```

---

# Constructor

```python
AutoMR(
    model,
    task="regression",
    input_type="image",
    epsilon=0.05,
    range_threshold=5.0
)
```

---

# Parameters

| Parameter | Description |
|------------|-------------|
| model | Target machine learning model |
| task | Prediction task type |
| input_type | Input handler type |
| epsilon | Comparison tolerance |
| range_threshold | Threshold for range testing |

---

# Main Methods

## run_full_test()

Runs the complete AutoMR pipeline.

```python
df, results = automr.run_full_test(dataset)
```

---

## run_dataset()

Executes AutoMR on a dataset.

```python
df = automr.run_dataset(dataset)
```

---

## run_all_mrs()

Executes all registered metamorphic relations for a single sample.

```python
df = automr.run_all_mrs(sample)
```

---

## run_mr()

Executes a single metamorphic relation.

```python
df, summary = automr.run_mr(
    sample,
    "brightness"
)
```

---

## analyze()

Generates analysis summaries.

```python
results = automr.analyze(df)
```

---

## save_results()

Exports reports.

```python
automr.save_results(
    df,
    results,
    output_dir="results"
)
```

---

# Registry Methods

```python
register_transform()
unregister_transform()
list_transforms()
list_relations()
get_transform()
get_relation()
```