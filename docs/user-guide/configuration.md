# Configuration

## Overview

AutoMR behavior can be configured during initialization and execution.

---

# AutoMR Parameters

```python
automr = AutoMR(
    model=model,
    task="regression",
    input_type="image",
    epsilon=0.05,
    range_threshold=5.0
)
```

---

# Configuration Options

| Parameter | Description |
|-----------|-------------|
| model | Target model |
| task | Prediction task |
| input_type | Input handler |
| epsilon | Comparison tolerance |
| range_threshold | Maximum acceptable deviation |

---

# Dataset Parameters

```python
automr.run_full_test(
    dataset=dataset,
    max_samples=100,
    samples_per_mr=5
)
```

---

# HPC Parameters

```python
HighPerformanceAutoMR(
    batch_size=64,
    num_workers=8
)
```

---

# Epsilon Analysis

```python
automr.run_full_test(
    epsilon_min=0.01,
    epsilon_max=0.10,
    epsilon_count=5
)
```

---

# Output Directory

```python
automr.run_full_test(
    output_dir="results"
)
```

---

# Recommended Settings

| Dataset Size | Batch Size |
|--------------|-----------|
| Small | 16 |
| Medium | 32 |
| Large | 64 |
| HPC GPU | 64–256 |