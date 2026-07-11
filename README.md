<div align="center">
  <img src="automrlogo.png" width="400"/>

<br/><br/>

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-22c55e?style=flat-square)
![Domain](https://img.shields.io/badge/Domain-Autonomous%20Driving-7c6fcd?style=flat-square)

<br/><br/>

  <p>
    A Model-Agnostic, Input-Agnostic, and Output-Agnostic Metamorphic Testing Framework<br/>
    for Regression-Based Autonomous Driving AI/ML Models.
  </p>
</div>

---

## Overview

AutoMR is a metamorphic testing framework designed to evaluate the robustness and reliability of AI/ML models **without requiring ground-truth labels**.

Instead of checking whether predictions exactly match expected outputs, AutoMR verifies whether a model behaves **consistently under controlled transformations** that should preserve expected behavior.

The framework automatically applies transformations, validates metamorphic relations, analyzes failures, and generates comprehensive reports — all with zero boilerplate.

| Problem                  | What AutoMR Does                                         |
| ------------------------ | -------------------------------------------------------- |
| No labeled data          | Tests models without any ground-truth labels             |
| Real-world perturbations | Measures robustness under realistic noise and conditions |
| Silent failures          | Pinpoints when and how models begin to fail              |

---

## Key Features

- **Model-Agnostic Testing** — works with TensorFlow, Keras, PyTorch, scikit-learn, XGBoost, or any custom model
- **Input-Agnostic Architecture** — supports images, time-series, sequential, and tabular data
- **Output-Agnostic Validation** — handles regression, continuous, and numerical outputs
- **Built-in Metamorphic Relations** — 16 ready-to-use metamorphic relations covering geometric, photometric, weather, behavioral, composite, and temporal transformations
- **Automated Transformation Pipeline** — 17 built-in transformations with configurable parameter ranges for robustness evaluation.
- **Parameter Range Testing** — sweep transformation parameters across configurable ranges
- **Epsilon Sensitivity Analysis** — automatically evaluates model robustness across multiple epsilon thresholds
- **Automatic Epsilon Recommendation** — identifies first failure, stabilization, and recommended epsilon values
- **Interactive Live Dashboard** — real-time webcam/video testing with configurable metamorphic relations and epsilon
- **Failure Detection and Localization** — pinpoints the exact conditions where models break
- **Severity Analysis** — ranks failures by output deviation magnitude
- **Failure Region Identification** — isolates parameter ranges with highest instability
- **Worst-Case Sample Discovery** — surfaces samples with the largest prediction deviations
- **CSV, JSON, and Text Report Generation** — comprehensive reporting with reproducible outputs
- **Verification Artifact Generation** — transformed samples saved automatically
- **Progress Tracking** — optional live progress bars for long-running evaluations
- **Extensible Plugin Architecture** — easily add custom transformations and relations
- **Standard AutoMR Engine** — easy-to-use metamorphic testing pipeline for AI/ML models
- **HighPerformanceAutoMR (HPC)** — optimized execution engine for large-scale metamorphic testing
- **Parallel Dataset Processing** — concurrent sample execution using configurable worker pools
- **Batch Inference Support** — reduces model inference overhead through batched prediction
- **Prediction Caching** — reuses predictions across epsilon sensitivity analyses to eliminate redundant inference
- **Shared Baseline Prediction Cache** — baseline predictions are computed once and reused across experiments
- **CPU Optimizations** — configurable threading and optimized execution for multicore processors
- **Configurable HPC Execution** — control worker count, chunk size, batch size, and prefetch behavior

---

## Installation

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

---

## Quick Start

```python
from automr.api import AutoMR

automr = AutoMR(
    model=model,
    task="regression",
    input_type="image",
    epsilon=0.05,
    range_threshold=5.0
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
### HighPerformanceAutoMR

```python
from automr.hpc import HighPerformanceAutoMR

automr = HighPerformanceAutoMR(
    model=model,
    task="regression",
    input_type="image",
    epsilon=0.05,
    range_threshold=5.0,
    workers=8,
    batch_size=64,
    chunk_size=64,
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

# Execution Engines

AutoMR provides two execution engines depending on the scale of testing.

| Engine | Description |
|---------|-------------|
| **AutoMR** | Standard execution engine suitable for small and medium datasets. |
| **HighPerformanceAutoMR** | Optimized execution engine for large datasets using parallel execution, batch inference, and prediction caching. |

---

# AutoMR vs HighPerformanceAutoMR

| Feature | AutoMR | HighPerformanceAutoMR |
|----------|--------|-----------------------|
| Standard Testing | ✅ | ✅ |
| Parallel Processing | ❌ | ✅ |
| Batch Inference | ❌ | ✅ |
| Prediction Cache | Limited | ✅ |
| CPU Optimization | Limited | ✅ |
| Large Dataset Support | Good | Excellent |
| Epsilon Prediction Reuse | ✅ | ✅ |

---

## Framework Workflow

```
Load Dataset
      ↓
Load Model
      ↓
Apply Transformations
      ↓
Generate Transformations
      ↓
Batch Prediction
      ↓
Prediction Cache
      ↓
Metamorphic Validation
      ↓
Validate Metamorphic Relations
      ↓
Analyze Failures
      ↓
Run Epsilon Sensitivity Analysis
      ↓
Generate Reports
      ↓
Interactive Live Dashboard
      ↓
Export Results
```
---

# HighPerformanceAutoMR

HighPerformanceAutoMR extends the standard AutoMR framework with high-performance execution capabilities.

## Features

- Parallel dataset processing
- Parallel metamorphic relation execution
- Batch model inference
- Prediction caching
- Shared epsilon prediction reuse
- Configurable worker threads
- Configurable batch sizes
- CPU optimized execution
- Scalable large dataset validation

---

# Performance Optimizations

AutoMR includes several optimization strategies to improve testing efficiency.

- Batch prediction
- Prediction caching
- Baseline prediction reuse
- Parallel dataset execution
- Shared epsilon prediction cache
- Background data prefetching
- CPU thread optimization
- Parallel metamorphic relation execution

---

## Supported Metamorphic Relations

| Relation | Purpose |
|----------|---------|
| `BlurRelation` | Tests robustness to Gaussian blur |
| `BrightnessRelation` | Tests robustness to brightness variation |
| `CompositeRelation` | Tests robustness under combined image perturbations |
| `ContrastRelation` | Tests robustness to contrast variation |
| `DarkVisibilityRelation` | Tests robustness under low-light and reduced visibility conditions |
| `DustRelation` | Tests robustness under dust simulation |
| `FogRelation` | Tests robustness under fog simulation |
| `HazeRelation` | Tests robustness under haze simulation |
| `NoiseRelation` | Tests robustness to Gaussian noise |
| `RainRelation` | Tests robustness under rain simulation |
| `RotationRelation` | Tests prediction stability under image rotation |
| `SandstormRelation` | Tests robustness under sandstorm simulation |
| `SmokeRelation` | Tests robustness under smoke simulation |
| `SnowRelation` | Tests robustness under snow simulation |
| `TemporalSmoothnessRelation` | Tests temporal consistency between sequential frames |
| `TranslationRelation` | Tests prediction stability under image translation |

---

## Supported Transformations

| Transformation | Description |
|---------------|-------------|
| `brightness` | Adjust image brightness |
| `contrast` | Modify image contrast |
| `blur` | Apply Gaussian blur |
| `rotation` | Rotate the image |
| `translation` | Translate the image horizontally or vertically |
| `noise` | Inject Gaussian noise |
| `composite` | Apply multiple image transformations simultaneously |
| `rain` | Simulate rainy weather |
| `snow` | Simulate snowy weather |
| `fog` | Simulate foggy conditions |
| `sandstorm` | Simulate sandstorm conditions |
| `dust` | Simulate dusty environments |
| `haze` | Simulate haze |
| `smoke` | Simulate smoke |
| `visibility` | Reduce scene visibility |
| `darkness` | Simulate low-light/night-time conditions |
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

```

AutoMR can automatically evaluate a model across multiple comparator thresholds.

Instead of manually selecting an epsilon value, the framework performs repeated metamorphic testing over a configurable epsilon range and reports:

- First Failure Epsilon
- Recommended Epsilon
- Stabilization Epsilon
- Maximum Failure Rate

Example console output
========== EPSILON ANALYSIS ==========

First Failure Epsilon : 0.01
Recommended Epsilon : 0.1367
Stabilization Epsilon : 0.1367
Maximum Failure Rate : 6.25%

======================================


Generated files
results/
├── epsilon_summary.csv
└── epsilon_report.txt
```

---

## Generated Reports

All reports are automatically saved to the `results/` directory.

### Core Reports

| File                   | Description                                             |
| ---------------------- | ------------------------------------------------------- |
| `automr_results.csv`   | Full per-sample test log                                |
| `failure_summary.csv`  | Failure rate per metamorphic relation                   |
| `severity_summary.csv` | Average output deviation per MR                         |
| `worst_cases.csv`      | Samples with the highest deviations                     |
| `failure_regions.txt`  | Parameter ranges where failures cluster                 |
| `range_summary.csv`    | Summary of parametric range sweep results               |
| `range_analysis.csv`   | Detailed per-range analysis                             |
| `prediction_trace.csv` | Full prediction trace across all samples and transforms |

### HPC Reports

When using HighPerformanceAutoMR, the same reports are generated with significantly faster execution on large datasets.

HighPerformanceAutoMR also reports execution statistics including:

- Total runtime
- Images processed
- Average images per second
- Worker count
- Batch size
- Prediction cache utilization

### Metadata Reports

| File                       | Description                        |
| -------------------------- | ---------------------------------- |
| `baseline_metrics.json`    | Model baseline performance metrics |
| `dataset_info.json`        | Dataset structure and statistics   |
| `model_summary.txt`        | Model architecture summary         |
| `original_predictions.csv` | Unmodified model predictions       |

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

| Column              | Description                                    |
| ------------------- | ---------------------------------------------- |
| `mr`                | Metamorphic relation identifier                |
| `param`             | Transformation parameter value                 |
| `original`          | Original model prediction                      |
| `transformed`       | Prediction after transformation                |
| `difference`        | Absolute prediction difference                 |
| `percent_change`    | Relative prediction change (%)                 |
| `passed`            | Boolean pass/fail result                       |
| `status`            | `PASS` or `FAIL`                               |
| `severity`          | Failure severity score                         |
| `sample_id`         | Dataset sample index                           |
| `expected_behavior` | Expected MR behavioral rule                    |
| `actual_behavior`   | Observed behavior (`Consistent` / `Violation`) |

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
- Epsilon Sensitivity Analysis
- Automatic Epsilon Recommendation

---

# API Overview

| Class | Description |
|--------|-------------|
| AutoMR | Standard metamorphic testing engine |
| HighPerformanceAutoMR | High-performance execution engine |
| TransformationRegistry | Transformation management |
| RelationRegistry | Metamorphic relation management |
| FailureAnalyzer | Failure analysis utilities |
| EpsilonSensitivity | Automatic epsilon evaluation |
| EpsilonSummary | Epsilon reporting utilities |

---

## Live Dashboard

````
AutoMR includes a real-time dashboard for evaluating metamorphic relations on webcam or video streams.

Features

- Live webcam/video inference
- Adjustable epsilon threshold
- Configurable metamorphic relations
- Interactive parameter range selection
- Real-time failure detection
- Automatic violation image capture
- Continuous CSV logging
- Summary statistics during execution

Launch

```python
from automr.dashboard import run_live_dashboard

run_live_dashboard(
    automr,
    model,
    video_source=0
)

Dashboard Controls

Control	Purpose
MR Index	Select relation
Enable	Enable/Disable relation
Tests	Number of parameter samples
Range %	Scale transformation range
Epsilon	Comparator threshold
Frame Skip	Processing frequency
R	Run benchmark
ESC	Exit

---

## 5. Update Generated Reports

Add the new files.

### Core Reports
automr_results.csv
prediction_trace.csv
failure_summary.csv
severity_summary.csv
worst_cases.csv
failure_regions.txt
range_summary.csv
range_analysis.csv
epsilon_summary.csv
epsilon_report.txt


### Dashboard Reports

Add a new subsection.

```markdown
### Live Dashboard Reports
results/live_dashboard/
├── dashboard_results.csv
├── dashboard_summary.csv
└── violations/


Each dashboard record stores the epsilon value used during evaluation, allowing experiments to be reproduced even when the threshold changes interactively.

````

---

## Design Principles

### Model-Agnostic

Any model implementing a `predict(x)` interface is compatible:

```python
output = model.predict(input)
```

Supported frameworks include TensorFlow, Keras, PyTorch, scikit-learn, XGBoost, and fully custom models.

### Input-Agnostic

AutoMR accepts any input type — images, time-series, sequential data, tabular data, or custom formats. Transformations are applied modularly and do not depend on input structure.

> AutoMR does not perform preprocessing. Users must provide inputs in the format expected by their model. This ensures the original model pipeline is evaluated without modification.

### Output-Agnostic

AutoMR supports regression outputs, continuous predictions, numerical outputs, and custom scalar outputs. No assumptions are made about output scale or range — the comparator is configurable via the `epsilon` parameter.

### Modular Architecture

| Component   | Role                                   |
| ----------- | -------------------------------------- |
| `Model`     | Generates predictions                  |
| `Transform` | Modifies input samples                 |
| `Relation`  | Defines expected behavioral properties |
| `Analyzer`  | Computes failure metrics and summaries |
| `Reporter`  | Exports CSV, JSON, and artifact files  |

---

## Project Structure

```
AutoMR-Framework/
│
├── automr/
│   ├── __init__.py
│   ├── api.py
|   ├── hpc/
|   │   ├── __init__.py
|   │   ├── automr.py
|   │   ├── executor.py
|   │   ├── batch_predictor.py
|   │   ├── cache.py
|   │   └── scheduler.py
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
│   │   └── baseline.py
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
│   │   └── wrapper.py
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
│   │   ├── image_transforms.py
│   │   ├── weather_transforms.py
│   │   ├── behavioral_transforms.py
│   │   └── temporal_transforms.py
│   │
│   └── verification/
│       ├── __init__.py
│       └── transformation_saver.py
│
├── examples/
│   ├── run_test.py
│   ├── webcam_automr_live.py
│   └── custom_relation_example.py
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
- Automatic (`epsilon`) recommendation is heuristic-based and should be validated for domain-specific safety requirements.
- Runtime depends on model inference speed
- Large datasets may require longer execution times

---

## Future Work

- NLP and text transformation extensions
- Tabular data transformation support
- Classification-specific metamorphic relations
- Native web dashboard
- Automatic epsilon optimization strategies
- Multi-camera live testing
- GPU-accelerated batch validation
- NLP and tabular metamorphic relations
- Distributed testing across multiple machines
- Cross-model MR comparison
- Automated result visualizations (plots and charts)
- Distributed and parallel testing support
- Web-based reporting interface

---

## Research Contributions

AutoMR provides the following contributions for regression-based autonomous driving systems:

- Automated metamorphic testing without ground-truth labels
- Label-free robustness validation under realistic conditions
- Parameterized MR evaluation with range sweep support
- Failure region detection and severity-based ranking
- Reusable and extensible testing infrastructure

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
  author={Charith Manujaya and Raveesha Peiris},
  year={2025}
}
```

---

## License

Released under the [MIT License](LICENSE).
