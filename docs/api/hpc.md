# HighPerformanceAutoMR API

## Overview

`HighPerformanceAutoMR` extends the standard AutoMR engine with optimized execution for large datasets.

---

# Import

```python
from automr.hpc import HighPerformanceAutoMR
```

---

# Constructor

```python
HighPerformanceAutoMR(
    model,
    task="regression",
    input_type="image",
    epsilon=0.05,
    range_threshold=5.0,
    batch_size=64,
    num_workers=8
)
```

---

# Additional Parameters

| Parameter | Description |
|------------|-------------|
| batch_size | GPU inference batch size |
| num_workers | Parallel worker threads |

---

# Main Methods

## run_full_test()

Runs the optimized AutoMR pipeline.

```python
df, results = automr.run_full_test(dataset)
```

---

## run_dataset()

Processes an entire dataset using the HPC engine.

---

## save_baseline()

Generates or reuses cached baseline predictions.

---

## save_results()

Exports all generated reports.

---

# Features

- Parallel execution
- GPU acceleration
- Batch inference
- Prediction cache
- Progress monitoring
- Epsilon analysis

## Cache Instrumentation

`HighPerformanceAutoMR` exposes the prediction cache through these methods:

```python
automr.clear_cache()
automr.cache_size()
automr.cache_stats()
automr.reset_cache_stats()
```

`cache_stats()` returns a dictionary with the exact keys `hits`, `misses`,
`requests`, `hit_ratio`, and `cache_size`. `reset_cache_stats()` resets hit,
miss, and request counters without clearing cached predictions. `clear_cache()`
clears cached predictions and resets the cache statistics.

The underlying `PredictionCache` also provides `get_stats()` and
`reset_stats()`, plus the `hits`, `misses`, `requests`, and `hit_ratio`
properties.

## Performance Benchmarking

Use [`AutoMRBenchmark`](benchmark.md) to compare a standard `AutoMR` instance
with a `HighPerformanceAutoMR` instance under identical dataset and execution
settings. The benchmark reports runtime, processed sample count, throughput,
CPU time, memory measurements, GPU availability and memory fields, cache
statistics, and speedup.