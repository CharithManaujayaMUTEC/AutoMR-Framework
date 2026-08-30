# Performance Benchmarking

## Overview

`AutoMRBenchmark` provides a controlled comparison between standard `AutoMR`
execution and `HighPerformanceAutoMR` execution.

## Import and Construction

```python
from automr.benchmark import AutoMRBenchmark

benchmark = AutoMRBenchmark(
    standard_automr=standard_automr,
    hpc_automr=hpc_automr,
)
```

Use the same model, dataset, and execution settings for both engines when
comparing performance.

## run()

```python
results = benchmark.run(
    dataset,
    max_samples=None,
    samples_per_mr=5,
    show_progress=False,
)
```

Both engines receive the same benchmark parameters. The returned dictionary
has `standard`, `hpc`, `cache`, and `speedup` keys. Each engine result reports:

- `runtime_seconds`
- `sample_count`
- `throughput_samples_per_second`
- `cpu_time_seconds`
- `memory_before_mb`, `memory_after_mb`, and `memory_delta_mb`
- `gpu`, containing `gpu_available`, `gpu_memory_allocated_mb`, and `gpu_memory_reserved_mb`
- `results`, containing the engine's execution result

The `cache` value contains the HPC cache statistics when the HPC instance
provides `cache_stats()`. `speedup` is standard runtime divided by HPC runtime.
