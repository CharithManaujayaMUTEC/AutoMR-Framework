<div align="center">
  <img src="automrlogo.png" width="400"/>

  <br/><br/>

  ![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
  ![PyPI](https://img.shields.io/pypi/v/automr?style=flat-square&color=3776AB)
  ![Downloads](https://img.shields.io/pypi/dm/automr?style=flat-square&color=22c55e)
  ![CI](https://img.shields.io/github/actions/workflow/status/CharithManaujayaMUTEC/AutoMR-Framework/ci.yml?style=flat-square&label=CI)
  ![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)
  ![Status](https://img.shields.io/badge/Status-Active-22c55e?style=flat-square)
  ![Domain](https://img.shields.io/badge/Domain-Autonomous%20Driving-7c6fcd?style=flat-square)
  ![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000?style=flat-square)

  <br/><br/>

  <p>
    A Model-Agnostic Metamorphic Testing Framework for Regression-Based AI/ML Models<br/>
    with Standard and High-Performance Execution Engines.
  </p>
</div>

---

## What's New in v0.7

- **HighPerformanceAutoMR (HPC)** — parallel execution engine for large-scale testing
- **Batch inference** — process multiple samples in a single model call
- **Prediction caching** — automatic baseline prediction reuse across MR sweeps
- **Parallel MR execution** — run multiple metamorphic relations concurrently
- **CPU/GPU backend selection** — automatic or manual backend switching per environment
- **GPU-accelerated transformations** — OpenCV CUDA acceleration for supported operations
- **Expanded model support** — TensorFlow, PyTorch, scikit-learn, XGBoost, ONNX Runtime, Remote API, and custom models
- **Epsilon sensitivity analysis** — automated threshold sweep with first-failure and stabilization detection
- **Automatic epsilon recommendation** — framework selects the optimal comparator threshold
- **New example scripts** — standard, HPC, and live dashboard examples

---

## Overview

AutoMR is a metamorphic testing framework designed to evaluate the robustness and reliability of AI/ML models **without requiring ground-truth labels**.

Instead of checking whether predictions exactly match expected outputs, AutoMR verifies whether a model behaves **consistently under controlled transformations** that should preserve expected behavior.

The framework automatically applies transformations, validates metamorphic relations, analyzes failures, and generates comprehensive reports — all with zero boilerplate.

| Problem | What AutoMR Does |
|---|---|
| No labeled data | Tests models without any ground-truth labels |
| Real-world perturbations | Measures robustness under realistic noise and conditions |
| Silent failures | Pinpoints when and how models begin to fail |

---

## Key Features

- **Model-Agnostic Testing** — works with TensorFlow, Keras, PyTorch, scikit-learn, XGBoost, or any custom model
- **Multi-Framework Model Support** — TensorFlow/Keras, PyTorch, scikit-learn, XGBoost, ONNX Runtime, Remote REST APIs, and custom models
- **Input-Agnostic Architecture** — supports images, time-series, sequential, and tabular data
- **Output-Agnostic Validation** — handles regression, continuous, and numerical outputs
- **Built-in Metamorphic Relations** — 16 ready-to-use relations covering geometric, photometric, weather, behavioral, composite, and temporal transformations
- **Automated Transformation Pipeline** — 23 built-in transformations with configurable parameter ranges
- **CPU/GPU Transformation Backend** — automatically switches between CPU and GPU implementations or allows manual backend selection
- **GPU-Accelerated Transformations** — OpenCV CUDA acceleration for supported image transformations
- **Backend-Agnostic Architecture** — identical API regardless of execution backend
- **HighPerformanceAutoMR (HPC) Engine** — parallel, batched, cache-accelerated execution for large datasets
- **Multi-Threaded Dataset Processing** — concurrent sample loading and transformation across worker pools
- **Native Batch Prediction** — optimized wrappers for supported frameworks
- **Batch Model Inference** — grouped prediction calls to reduce inference overhead
- **Prediction Caching** — baseline predictions computed once and reused across all MR sweeps
- **Shared Baseline Prediction Cache** — single baseline pass feeds all MR sweeps in the same run
- **CPU Optimizations** — configurable threading, prefetch loading, and shared epsilon cache
- **Parallel Metamorphic Testing** — multiple relations evaluated concurrently
- **Parameter Range Testing** — sweep transformation parameters across configurable ranges
- **Epsilon Sensitivity Analysis** — automatically evaluates model robustness across multiple epsilon thresholds
- **Automatic Epsilon Recommendation** — identifies first failure, stabilization, and recommended epsilon values
- **Interactive Live Dashboard** — real-time webcam/video testing with configurable MRs and epsilon
- **Failure Detection and Localization** — pinpoints the exact conditions where models break
- **Severity Analysis** — ranks failures by output deviation magnitude
- **Failure Region Identification** — isolates parameter ranges with highest instability
- **Worst-Case Sample Discovery** — surfaces samples with the largest prediction deviations
- **Graph Module (GraphGenerator)** — automatically exports evaluation plots and graph-ready CSV files under `results/graphs/`
- **Generic Model Wrapper Support** — wrap any callable as an AutoMR-compatible model
- **Plugin Architecture** — register custom transformations and relations at runtime
- **Verification Artifact Generation** — transformed samples saved automatically per relation
- **CSV, JSON, and Text Report Generation** — comprehensive reproducible outputs
- **Progress Tracking** — optional live progress bars for long-running evaluations
- **Decoder Health Validation** — warning-only diagnostics for baseline prediction variance, standard deviation, range, uniqueness, saturation, clipping, and distribution
- **Enhanced Epsilon Diagnostics** — normalized epsilon, prediction range, plateau and flat-zero curve detection, and diagnostic warnings
- **Cache Instrumentation** — inspect cache hits, misses, requests, hit ratio, and cache size, and reset statistics independently
- **Performance Benchmarking** — compare standard and high-performance execution with runtime, throughput, CPU, memory, GPU, cache, and speedup metrics
- **Final Evaluation Reporting** — consolidate available experiment artifacts into `final_evaluation_report.json`

---

## Installation

### Basic Installation

```bash
pip install -r requirements.txt
```

### PyPI

```bash
pip install automr
```

### Source Installation

```bash
git clone https://github.com/CharithManaujayaMUTEC/AutoMR-Framework.git
cd AutoMR-Framework

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

### Optional GPU Support

For CUDA-enabled systems:

```bash
pip install onnxruntime-gpu
```

For PyTorch CUDA, install the appropriate build from:
https://pytorch.org/get-started/locally/

---

## Supported Models

AutoMR automatically detects and wraps supported model frameworks through a unified wrapper factory.

| Framework | Supported |
|---|---|
| TensorFlow / Keras | ✅ |
| PyTorch | ✅ |
| scikit-learn | ✅ |
| XGBoost | ✅ |
| ONNX Runtime | ✅ |
| Remote REST APIs | ✅ |
| Custom `predict()` models | ✅ |

---

## Supported Backends

| Backend | Status |
|---|---|
| CPU | ✅ |
| GPU (CUDA / OpenCV CUDA) | ✅ |
| Automatic selection | ✅ |

---

## Quick Start

### Standard Engine

```python
from automr.api import AutoMR
from automr.transforms.backend import set_backend

# auto | cpu | gpu
set_backend("auto")

automr = AutoMR(
    model=model,
    task="regression",
    input_type="image",
    epsilon=0.05,
      range_threshold=5.0,
      transform_ranges={
            "brightness": {"start": 0.2, "end": 2.5, "samples": 7},
            "rotation": {"start": -45, "end": 45, "samples": 9},
      },
)

df, results = automr.run_full_test(
    dataset=dataset,
    max_samples=500,
    samples_per_mr=5,
    epsilon_min=0.01,
    epsilon_max=0.20,
    epsilon_count=4
)
```

Use `transform_ranges` to override the default transformation range for any built-in metamorphic relation. The keys must match the registered relation names, and each entry can set `start`, `end`, and `samples`.

### High-Performance Engine (HPC)

```python
from automr.hpc import HighPerformanceAutoMR
from automr.transforms.backend import set_backend

set_backend("auto")

automr = HighPerformanceAutoMR(
    model=model,
    task="regression",
    input_type="image",
    epsilon=0.05,
    range_threshold=5.0,
    num_workers=8,
    batch_size=64,
      transform_ranges={
            "brightness": {"start": 0.5, "end": 2.0, "samples": 8},
            "rotation": {"start": -20, "end": 20, "samples": 11},
      },
)

df, results = automr.run_full_test(
    dataset=dataset,
    max_samples=None,
    samples_per_mr=5,
    epsilon_min=0.005,
    epsilon_max=0.05,
    epsilon_count=3,
)
```

---

## Command-Line Interface

AutoMR provides a small CLI for quick access to version and documentation.

Usage examples:

```bash
automr --version   # show installed AutoMR version
automr --info      # print framework feature summary and version
automr --docs      # list available documentation files in the package
```

See the full CLI reference in the documentation: [CLI Reference](docs/cli.md)


## Backend Selection

AutoMR supports three execution modes for transformation processing.

```python
from automr.transforms.backend import set_backend

# Automatic backend selection (recommended)
set_backend("auto")

# Force CPU
set_backend("cpu")

# Force GPU
set_backend("gpu")
```

| Backend | Description |
|---|---|
| `auto` | Automatically selects GPU when available, falls back to CPU |
| `cpu` | Always uses CPU implementations |
| `gpu` | Always uses GPU (CUDA) implementations |

---

## Architecture

```
User Model
      │
      ▼
Wrapper Factory
      │
      ▼
Model Wrapper
      │
      ▼
AutoMR / HighPerformanceAutoMR
      │
      ▼
Backend Selector
      │
 ┌────┴────┐
 │         │
CPU       GPU
 │         │
 └────┬────┘
      ▼
Transformations
      ▼
Range Tester
      ▼
Relations
      ▼
Analyzer
      ▼
Reports
```

```
AutoMR
    │
    ├── Standard Engine
    │       ├── TransformationRegistry
    │       ├── RelationRegistry
    │       ├── FailureAnalyzer
    │       └── EpsilonSensitivity
    │
    └── HighPerformanceAutoMR
            ├── Parallel Executor
            ├── Batch Predictor
            ├── Prediction Cache
            ├── Prefetch Loader
            └── Result Aggregator
```

---

## Execution Engines

AutoMR provides two execution engines depending on the scale and performance requirements of testing.

| Engine | Description |
|---|---|
| **AutoMR** | Standard execution engine suitable for small and medium datasets |
| **HighPerformanceAutoMR** | Optimized engine for large datasets using parallel execution, batch inference, and prediction caching |

---

## HighPerformanceAutoMR

`HighPerformanceAutoMR` extends the standard `AutoMR` engine with a parallel, cache-accelerated execution backend designed for large-scale or latency-sensitive testing.

| Capability | Description |
|---|---|
| Parallel dataset processing | Multiple samples processed concurrently across configurable worker threads |
| Batch inference | Samples grouped into batches and passed to the model in a single call |
| Prediction caching | Baseline predictions computed once and shared across all MR sweeps |
| CPU optimization | Prefetch loading and shared epsilon cache minimize redundant computation |
| GPU acceleration | CUDA-backed transformations via OpenCV CUDA when GPU backend is active |
| Configurable workers | `num_workers` controls the thread pool size |
| Configurable batch sizes | `batch_size` controls how many samples are grouped per inference call |

### AutoMR vs. HighPerformanceAutoMR

| Feature | AutoMR | HighPerformanceAutoMR |
|---|---|---|
| Standard execution | ✅ | ✅ |
| CPU backend | ✅ | ✅ |
| GPU backend | ✅ | ✅ |
| Parallel processing | ❌ | ✅ |
| Batch inference | ❌ | ✅ |
| Prediction caching | Limited | ✅ |
| HPC optimization | ❌ | ✅ |
| Epsilon prediction reuse | ✅ | ✅ |
| Large dataset support | Good | Excellent |

---

## Framework Workflow

```
Load Dataset
      ↓
Load Model
      ↓
Select Backend (CPU / GPU)
      ↓
Generate Transformations
      ↓
Batch Prediction
      ↓
Prediction Cache
      ↓
Metamorphic Validation
      ↓
Failure Analysis
      ↓
Epsilon Sensitivity Analysis
      ↓
Report Generation
      ↓
Interactive Live Dashboard
      ↓
Export Results
```

---

## Supported Metamorphic Relations

| Relation | Purpose |
|---|---|
| `BlurRelation` | Robustness to Gaussian blur |
| `BrightnessRelation` | Robustness to brightness variation |
| `CompositeRelation` | Robustness under combined image perturbations |
| `ContrastRelation` | Robustness to contrast variation |
| `DarkVisibilityRelation` | Robustness under low-light and reduced visibility conditions |
| `DustRelation` | Robustness under dust simulation |
| `FogRelation` | Robustness under fog simulation |
| `HazeRelation` | Robustness under haze simulation |
| `NoiseRelation` | Robustness to Gaussian noise |
| `RainRelation` | Robustness under rain simulation |
| `RotationRelation` | Stability under image rotation |
| `SandstormRelation` | Robustness under sandstorm simulation |
| `SmokeRelation` | Robustness under smoke simulation |
| `SnowRelation` | Robustness under snow simulation |
| `TemporalSmoothnessRelation` | Temporal consistency across sequential frames |
| `TranslationRelation` | Stability under image translation |

---

## Supported Transformations

| Transformation | Description |
|---|---|
| `brightness` | Adjust image brightness |
| `contrast` | Modify image contrast |
| `blur` | Apply Gaussian blur |
| `rotation` | Rotate the image |
| `translation` | Translate the image horizontally or vertically |
| `noise` | Inject Gaussian noise |
| `global_brightness` | Apply brightness adjustment across the full frame |
| `global_contrast` | Apply contrast adjustment across the full frame |
| `global_blur` | Apply blur across the full frame |
| `global_noise` | Apply noise across the full frame |
| `global_rotation` | Apply rotation across the full frame |
| `global_translation` | Apply translation across the full frame |
| `composite` | Apply multiple transformations simultaneously |
| `rain` | Simulate rainy weather |
| `snow` | Simulate snowy weather |
| `fog` | Simulate foggy conditions |
| `haze` | Simulate haze |
| `smoke` | Simulate smoke |
| `dust` | Simulate dusty environments |
| `sandstorm` | Simulate sandstorm conditions |
| `darkness` | Simulate low-light or night-time conditions |
| `visibility` | Reduce scene visibility |
| `temporal` | Generate temporal frame pairs for sequence consistency testing |

---

## Example Results

```
=== AutoMR Results ===

                            total  passed  failed  failure_rate

DarkVisibilityRelation       100      89      11        0.11
TranslationRelation           50      47       3        0.06
RotationRelation              50      49       1        0.02
BrightnessRelation            50      50       0        0.00
ContrastRelation              50      50       0        0.00
BlurRelation                  50      50       0        0.00
FogRelation                   50      50       0        0.00
RainRelation                  50      50       0        0.00
SnowRelation                  50      50       0        0.00
```

---

## Epsilon Sensitivity Analysis

AutoMR can automatically evaluate a model across multiple comparator thresholds. Instead of manually selecting an epsilon value, the framework performs repeated metamorphic testing over a configurable epsilon range and reports:

- First Failure Epsilon
- Recommended Epsilon
- Stabilization Epsilon
- Maximum Failure Rate

```
========== EPSILON ANALYSIS ==========

First Failure Epsilon  : 0.01
Recommended Epsilon    : 0.1367
Stabilization Epsilon  : 0.1367
Maximum Failure Rate   : 6.25%

======================================
```

Generated files:

```
results/
├── epsilon_summary.csv
└── epsilon_report.txt
```

---

## Performance Optimizations

AutoMR and HighPerformanceAutoMR apply a layered optimization stack to maximize throughput:

- **Parallel dataset execution** — concurrent sample processing across a configurable worker thread pool
- **Batch prediction** — samples grouped into batches per inference call
- **Prediction caching** — baseline predictions stored and reused; never recomputed for the same sample
- **Shared baseline prediction cache** — single baseline pass feeds all MR sweeps in the same run
- **Shared epsilon prediction cache** — epsilon sensitivity results shared across MR evaluations
- **Background data prefetching** — next batch prepared while the current batch is being evaluated
- **CPU thread optimization** — configurable threading for multicore processors
- **GPU-accelerated transformations** — CUDA-backed OpenCV operations when GPU backend is active
- **Automatic backend selection** — environment-aware CPU/GPU switching with no API changes
- **Native batch prediction wrappers** — framework-specific batched inference for TensorFlow, PyTorch, ONNX Runtime
- **Parallel metamorphic relation execution** — multiple relations evaluated concurrently where dataset size permits

---

## Generated Reports

All reports are automatically saved to the `results/` directory.

### Core Reports

| File | Description |
|---|---|
| `automr_results.csv` | Full per-sample test log |
| `prediction_trace.csv` | Full prediction trace across all samples and transforms |
| `failure_summary.csv` | Failure rate per metamorphic relation |
| `severity_summary.csv` | Average output deviation per MR |
| `worst_cases.csv` | Samples with the highest deviations |
| `failure_regions.txt` | Parameter ranges where failures cluster |
| `range_summary.csv` | Summary of parametric range sweep results |
| `range_analysis.csv` | Detailed per-range analysis |
| `epsilon_summary.csv` | Failure rate at each evaluated epsilon threshold |
| `epsilon_report.txt` | Human-readable epsilon recommendation report |

### HPC Reports

When using `HighPerformanceAutoMR`, the same core reports are generated with significantly faster execution. Additionally, HPC runs report execution statistics:

- Total runtime
- Images processed
- Average images per second
- Worker count
- Batch size
- Prediction cache utilization

### Graph Reports

AutoMR now includes a dedicated graph module (`GraphGenerator`) that automatically saves visual summaries to `results/graphs/` during `save_results(...)`.

```
results/graphs/
├── overall/
│   ├── pass_fail_pie.png
│   ├── prediction_distribution.png
│   ├── difference_distribution.png
│   ├── failure_heatmap.png
│   ├── worst_cases.png
│   └── summary_dashboard.png
├── summary/
│   ├── failure_rate.png
│   ├── severity.png
│   ├── range_analysis.png
│   ├── epsilon_curve.png
│   ├── failure_rate.csv
│   ├── severity.csv
│   └── epsilon_curve.csv
└── <mr_name>/
      ├── parameter_vs_prediction.png
      ├── testcase_vs_prediction.png
      ├── parameter_vs_prediction.csv
      └── testcase_vs_prediction.csv
```

### Metadata Reports

| File | Description |
|---|---|
| `baseline_metrics.json` | Model baseline performance metrics |
| `dataset_info.json` | Dataset structure and statistics |
| `model_summary.txt` | Model architecture summary |
| `original_predictions.csv` | Unmodified model predictions |

### Live Dashboard Reports

```
results/live_dashboard/
├── webcam_results.csv
├── dashboard_summary.csv
└── violations/
```

Each dashboard record stores the epsilon value used during evaluation, allowing experiments to be reproduced even when the threshold changes interactively.

### Verification Artifacts

Transformation samples are saved per relation under `results/transformation_samples/`:

```
transformation_samples/
├── metadata.csv
├── transformation_summary.csv
├── brightness/
├── contrast/
├── blur/
├── rotation/
├── translation/
├── noise/
├── rain/
├── snow/
├── fog/
├── visibility/
└── darkness/
```

---

## Output Columns

| Column | Description |
|---|---|
| `mr` | Metamorphic relation identifier |
| `param` | Transformation parameter value |
| `original` | Original model prediction |
| `transformed` | Prediction after transformation |
| `difference` | Absolute prediction difference |
| `percent_change` | Relative prediction change (%) |
| `passed` | Boolean pass/fail result |
| `status` | `PASS` or `FAIL` |
| `severity` | Failure severity score |
| `sample_id` | Dataset sample index |
| `expected_behavior` | Expected MR behavioral rule |
| `actual_behavior` | Observed behavior (`Consistent` / `Violation`) |

---

## Built-in Analysis

AutoMR automatically computes the following after each test run:

- **Failure Rate** — per metamorphic relation, across all samples
- **Severity Analysis** — average and maximum output deviation
- **Worst-Case Failures** — samples with the largest prediction deviations
- **Failure Regions** — parameter ranges where the model is most unstable
- **Parameter Sensitivity** — how model behavior shifts with transformation intensity
- **Range Stability Analysis** — identifies safe vs. unstable transformation ranges
- **Prediction Trace Analysis** — tracks prediction drift across all transformations
- **Epsilon Sensitivity Analysis** — failure rate curve across a threshold sweep
- **Automatic Epsilon Recommendation** — heuristic-based optimal threshold selection

### Validation and Diagnostics

Decoder health analysis runs during `run_full_test()` and is warning-only: it
does not block or stop metamorphic testing. The report includes prediction
variance, standard deviation, minimum, maximum, range, unique count and ratio,
quantiles, saturation and clipping ratios, a distribution diagnostic, and any
warnings. When an output directory is used, it is saved as
`decoder_health.json`.

Epsilon sensitivity summaries include prediction range, normalized epsilon,
stabilization and first-failure values, plateau detection, flat-zero curve
detection, a `curve_diagnostic`, and warnings that can identify potentially
uninformative sensitivity behavior.

### Cache Instrumentation

For `HighPerformanceAutoMR`, `cache_stats()` returns `hits`, `misses`,
`requests`, `hit_ratio`, and `cache_size`. Use `reset_cache_stats()` to reset
instrumentation without clearing predictions, or `clear_cache()` to clear the
cache and reset its counters.

### Final Evaluation Report

`run_full_test()` adds `final_evaluation_report` to its results. The
`FinalEvaluationReport` class consolidates available test summary, baseline,
decoder health, epsilon, benchmark, and additional analysis artifacts and
saves `final_evaluation_report.json` in the output directory.

---

## Live Dashboard

AutoMR includes a real-time interactive dashboard for evaluating metamorphic relations on live camera streams, IP cameras, RTSP streams, or video files.

Unlike offline testing, the Live Dashboard evaluates **one transformation parameter per frame**, enabling smooth real-time visualization while maintaining responsive inference. A complete parameter sweep remains available through **Benchmark Mode**.

**Features**

- Live inference from USB webcams, video files, Android IP Webcam, and RTSP streams
- Interactive MR selection using the **MR Index** control
- Enable or disable individual metamorphic relations during execution
- Adjustable transformation intensity using the **Intensity** slider
- Adjustable transformation range using the **Range** slider
- Configurable epsilon threshold with real-time updates
- One transformation intensity evaluated per frame for responsive live testing
- Full benchmark mode (**R**) to execute the complete parameter sweep
- Per-MR result caching with HOLD overlays for easier visualization
- Active MR focus—only the selected MR is evaluated each test cycle while other MR tiles are served from cache
- Real-time prediction display with colour-coded PASS/FAIL result tiles
- Automatic metamorphic relation verification
- Automatic violation image capture
- Continuous CSV result logging
- Live summary statistics panel showing tests, failures, and failure rate
- Benchmark progress indicator during full parameter sweeps
- Compatible with CPU and GPU inference

**Launch**

```python
from automr.dashboard import run_live_dashboard

run_live_dashboard(
    automr,
    model,
    video_source=0
)
```

**Dashboard Controls**

| Control | Purpose |
|---|---|
| MR Index | Select active metamorphic relation |
| Enable | Enable / disable the selected relation |
| Tests | Number of parameter samples per sweep |
| Range % | Scale the transformation range |
| Epsilon x1000 | Comparator threshold |
| FrameSkip | Processing frequency (frames between tests) |
| 1–9 / V / D | Toggle MRs by keyboard |
| R | Run focused benchmark on current MR |
| ESC | Exit dashboard |

---

## API Overview

| Class | Description |
|---|---|
| `AutoMR` | Standard metamorphic testing engine |
| `HighPerformanceAutoMR` | Parallel, batched HPC engine |
| `TransformationRegistry` | Register and retrieve input transformations |
| `RelationRegistry` | Register and retrieve metamorphic relations |
| `FailureAnalyzer` | Compute failure metrics, severity, and worst cases |
| `EpsilonSensitivity` | Run threshold sweeps and generate epsilon reports |
| `EpsilonSummary` | Epsilon reporting utilities |
| `Dashboard` | Real-time live testing interface |

---

## Extending AutoMR

### Custom transformations and relations

Custom transformations and metamorphic relations can be registered at runtime
without modifying built-in framework code. A complete custom MR uses the same
name for its transformation, relation, and parameter range, then runs through
the existing `run_mr()` pipeline.

```python
from automr import AutoMR

def invert_colors(
      image,
      factor=1.0,
      seed=None,
):
      return 255 - image

class CustomRelation:
      def __init__(self, epsilon=0.10):
            self.epsilon = epsilon

      def check(self, y1, y2):
            return abs(y1 - y2) <= self.epsilon

      def type(self):
            return "equality"

      def expected(self):
            return "Prediction should remain approximately invariant."

automr = AutoMR(model=model, task="regression", input_type="image")
automr.register_custom_mr(
      name="invert_colors",
      transform=invert_colors,
      relation=CustomRelation(epsilon=0.10),
      param_range={"start": 0.0, "end": 1.0, "samples": 5},
)

automr.run_mr(
      input_data=image,
      mr_name="invert_colors",
      samples=5,
)
```

The independent methods `register_custom_transformation(name, transform,
param_range=None)` and `register_custom_relation(name, relation)` are also
available. `InvarianceRelation` is an existing relation implementation with
the same `check(y1, y2)`, `type()`, and `expected()` interface.

### Register a custom transformation

```python
from automr.registry import TransformationRegistry

registry = TransformationRegistry()

@registry.register("custom_blur")
def my_blur(image, param):
    return cv2.GaussianBlur(image, (0, 0), param)
```

### Register a custom relation

```python
from automr.registry import RelationRegistry
from automr.relations.base import BaseRelation

class MyRelation(BaseRelation):
    def check(self, original, transformed):
        return abs(original - transformed) < self.epsilon

registry = RelationRegistry()
registry.register("my_relation", MyRelation)
```

### Unregister a plugin

```python
registry.unregister("custom_blur")
```

### Custom model wrapper

```python
from automr.models.wrapper import ModelWrapper

class MyModelWrapper(ModelWrapper):
    def predict(self, x):
        return self.model(x).item()

    def predict_batch(self, batch):
        return self.model(batch).numpy()
```

---

## Design Principles

### Model-Agnostic

Any model implementing a `predict(x)` interface is compatible:

```python
output = model.predict(input)
```

Supported frameworks include TensorFlow, Keras, PyTorch, scikit-learn, XGBoost, ONNX Runtime, and fully custom models.

### Input-Agnostic

AutoMR accepts any input type — images, time-series, sequential data, tabular data, or custom formats. Transformations are applied modularly and do not depend on input structure.

> AutoMR does not perform preprocessing. Users must provide inputs in the format expected by their model. This ensures the original model pipeline is evaluated without modification.

### Output-Agnostic

AutoMR supports regression outputs, continuous predictions, numerical outputs, and custom scalar outputs. No assumptions are made about output scale or range — the comparator is configurable via the `epsilon` parameter.

### Backend-Agnostic

The transformation pipeline exposes an identical API regardless of whether CPU or GPU execution is active. Switching backends requires a single `set_backend()` call and no changes to model or testing code.

### Modular Architecture

| Component | Role |
|---|---|
| `Model` | Generates predictions |
| `Transform` | Modifies input samples |
| `Backend` | Selects CPU or GPU execution path |
| `Relation` | Defines expected behavioral properties |
| `Analyzer` | Computes failure metrics and summaries |
| `Reporter` | Exports CSV, JSON, and artifact files |

---

## Project Structure

```
AutoMR-Framework/
│
├── automr/
│   ├── __init__.py
│   ├── api.py
│   │
│   ├── hpc/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── executor.py
│   │   ├── scheduler.py
│   │   ├── batcher.py
│   │   ├── cache.py
│   │   └── utils.py
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── analyzer.py
│   │
│   ├── comparators/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── regression.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── range_tester.py
│   │   ├── failure_analysis.py
│   │   └── validation_runner.py
│   │
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── live_dashboard.py
│   │   ├── video_runner.py
│   │   ├── control_panel.py
│   │   ├── dashboard_utils.py
│   │   └── graph_panel.py
│   │
│   ├── epsilon/
│   │   ├── __init__.py
│   │   ├── sensitivity.py
│   │   ├── summary.py
│   │   └── utils.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   └── graph_generator.py
│   │
│   ├── input_handlers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── image.py
│   │   ├── tabular.py
│   │   └── sequence.py
│   │
│   ├── logging/
│   │   ├── __init__.py
│   │   └── logger.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── tensorflow_wrapper.py
│   │   ├── pytorch_wrapper.py
│   │   ├── sklearn_wrapper.py
│   │   ├── xgboost_wrapper.py
│   │   ├── onnx_wrapper.py
│   │   ├── remote_wrapper.py
│   │   ├── custom_wrapper.py
│   │   └── wrapper_factory.py
│   │
│   ├── registry/
│   │   ├── __init__.py
│   │   ├── transformation_registry.py
│   │   └── relation_registry.py
│   │
│   ├── relations/
│   │   ├── image_relations.py
│   │   ├── weather_relations.py
│   │   ├── behavioral_relations.py
│   │   └── temporal_relations.py
│   │
│   ├── transforms/
│   │   ├── backend.py
│   │   ├── backend_utils.py
│   │   ├── image_transforms.py
│   │   ├── weather_transforms.py
│   │   ├── behavioral_transforms.py
│   │   ├── temporal_transforms.py
│   │   ├── composite_transforms.py
│   │   ├── translation.py
│   │   ├── cpu/
│   │   ├── gpu/
│   │   └── effects/
│   │
│   └── verification/
│       ├── __init__.py
│       └── transformation_saver.py
│
├── examples/
│   ├── run_test.py
│   ├── hpc_run_test.py
│   ├── custom_model_example.py
│   ├── plugin_example.py
│   ├── classification_example.py
│   ├── dashboard_example.py
│   └── webcam_automr_live.py
│
├── results/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

---

## Current Limitations

- Transformation suite is primarily focused on image-based inputs
- Classification-specific metamorphic relations are still under development
- Automatic epsilon recommendation is heuristic-based and should be validated against domain-specific safety requirements
- Runtime depends on model inference speed
- HPC batch inference requires models that support batched input
- GPU backend requires a CUDA-capable device and compatible OpenCV build

---

## Future Work

- Multi-GPU execution support
- Distributed multi-node execution
- Ray / Dask integration for elastic scaling
- Kubernetes-native execution support
- CUDA kernels for additional transformations
- Apple Metal backend
- ROCm backend
- Cloud execution backend
- Interactive HTML report generation
- Classification-specific metamorphic relation library
- NLP and tabular MR support
- Cross-model comparison testing
- Native web dashboard
- Multi-camera live testing

---

## Research Contributions

AutoMR provides the following contributions for regression-based autonomous driving systems:

- Automated metamorphic testing without ground-truth labels
- Label-free robustness validation under realistic conditions
- Parameterized MR evaluation with range sweep support
- Failure region detection and severity-based ranking
- Epsilon sensitivity analysis with automatic threshold recommendation
- HPC execution engine for scalable parallel metamorphic testing
- CPU/GPU backend abstraction for environment-adaptive execution
- Expanded multi-framework model support via wrapper factory
- Reusable and extensible plugin architecture

---

## Examples

```
examples/
├── run_test.py               Standard AutoMR test run
├── hpc_run_test.py           HighPerformanceAutoMR test run
├── custom_model_example.py   Wrapping a custom model
├── plugin_example.py         Registering custom transforms and relations
├── classification_example.py Classification model testing (preview)
├── dashboard_example.py      Launching the live dashboard
└── webcam_automr_live.py     Webcam-based live MR evaluation
```

---

## Authors

**Charith Manujaya** — [github.com/CharithManaujayaMUTEC](https://github.com/CharithManaujayaMUTEC)

**Raveesha Peiris** — [github.com/RaveeshaPeiris](https://github.com/RaveeshaPeiris)

> Final Year Project — Metamorphic Testing Framework for Regression-Based Autonomous Driving AI/ML Models

---

## Citation

```bibtex
@software{automr2025,
  title={AutoMR: A Metamorphic Testing Framework for Regression-Based Autonomous Driving Models},
  author={Charith Manujaya, Raveesha Peiris, Thurunu Pabasara, Tharika Akurana},
  year={2025}
}
```

---

## License

Released under the [MIT License](LICENSE).