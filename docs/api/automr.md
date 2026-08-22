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

## Custom Extension Methods

Custom extensions can be registered at runtime through convenience methods on
an `AutoMR` instance. These methods leave the existing registration APIs and
built-in registries unchanged.

### register_custom_transformation()

```python
automr.register_custom_transformation(
    name,
    transform,
    param_range=None,
)
```

Registers a transformation independently. `name` is the unique registry
name, `transform` is the callable, and the optional `param_range` is a
dictionary containing `start`, `end`, and `samples`. The method returns the
`AutoMR` instance.

### register_custom_relation()

```python
automr.register_custom_relation(
    name,
    relation,
)
```

Registers a relation independently. `relation` may be a callable or relation
object. A relation object used by the framework provides `check(y1, y2)`,
`type()`, and `expected()`. The method returns the `AutoMR` instance.

### register_custom_mr()

```python
automr.register_custom_mr(
    name,
    transform,
    relation,
    param_range,
)
```

Registers a complete custom metamorphic relation using the same `name` in the
transformation registry, relation registry, and `mr_ranges` parameter
configuration. `param_range` is a dictionary containing `start`, `end`, and
`samples`. The method returns the `AutoMR` instance, and the registered MR is
available to the existing `run_mr()` and `run_all_mrs()` pipeline.

For example, the existing `InvarianceRelation` implements the required
relation interface:

```python
from automr import AutoMR
from automr.relations.image_relations import InvarianceRelation

def invert_colors(
    image,
    factor=1.0,
    seed=None,
):
    return 255 - image

automr = AutoMR(model=model, task="regression", input_type="image")

automr.register_custom_mr(
    name="invert_colors",
    transform=invert_colors,
    relation=InvarianceRelation(
        epsilon=0.10,
    ),
    param_range={
        "start": 0.0,
        "end": 1.0,
        "samples": 5,
    },
)

automr.run_mr(
    input_data=image,
    mr_name="invert_colors",
    samples=5,
)
```

---

# Live Dashboard

AutoMR also provides an interactive Live Dashboard for real-time metamorphic testing.

```python
from automr.dashboard import run_live_dashboard

run_live_dashboard(
    automr=automr,
    model=model,
    video_source=0
)
```

For complete dashboard documentation, see **Dashboard API**.