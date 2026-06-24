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
- **Built-in Metamorphic Relations** — 11 ready-to-use relations covering weather, geometric, and temporal properties
- **Automated Transformation Pipeline** — end-to-end execution with configurable parameters
- **Parameter Range Testing** — sweep transformation parameters across configurable ranges
- **Failure Detection and Localization** — pinpoints the exact conditions where models break
- **Severity Analysis** — ranks failures by output deviation magnitude
- **Failure Region Identification** — isolates parameter ranges with highest instability
- **Worst-Case Sample Discovery** — surfaces samples with the largest prediction deviations
- **CSV and JSON Report Generation** — full results persisted automatically
- **Verification Artifact Generation** — transformation samples saved per relation
- **Progress Tracking** — optional live progress bars for long-running tests
- **Extensible Transformation Framework** — add custom transforms and relations with minimal code

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
    strict=True
)

df, results = automr.run_full_test(
    dataset=dataset,
    max_samples=2000,
    samples_per_mr=5,
    show_progress=True
)
```

---

## Framework Workflow

```
Load Dataset
      ↓
Load Model
      ↓
Apply Transformations
      ↓
Generate Predictions
      ↓
Validate Metamorphic Relations
      ↓
Analyze Failures
      ↓
Generate Reports
      ↓
Export Results
```

---

## Supported Metamorphic Relations

| Relation                     | Purpose                                 |
| ---------------------------- | --------------------------------------- |
| `BrightnessRelation`         | Robustness to brightness variation      |
| `ContrastRelation`           | Robustness to contrast variation        |
| `BlurRelation`               | Robustness to blur                      |
| `RotationRelation`           | Stability under rotation                |
| `TranslationRelation`        | Stability under translation             |
| `NoiseRelation`              | Robustness to Gaussian noise            |
| `RainRelation`               | Robustness under rain simulation        |
| `SnowRelation`               | Robustness under snow simulation        |
| `FogRelation`                | Robustness under fog simulation         |
| `DarkVisibilityRelation`     | Robustness under visibility degradation |
| `TemporalSmoothnessRelation` | Temporal consistency across frames      |

---

## Supported Transformations

| Transformation | Description                  |
| -------------- | ---------------------------- |
| Brightness     | Pixel intensity modification |
| Contrast       | Contrast adjustment          |
| Blur           | Gaussian blur                |
| Rotation       | Image rotation               |
| Translation    | Spatial translation          |
| Noise          | Gaussian noise injection     |
| Rain           | Rain simulation              |
| Snow           | Snow simulation              |
| Fog            | Fog simulation               |
| Visibility     | Visibility degradation       |
| Darkness       | Low-light simulation         |
| Temporal       | Sequential frame analysis    |

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
│   ├── api.py
│   ├── comparator.py
│   │
│   ├── core/
│   │   ├── range_tester.py
│   │   ├── failure_analysis.py
│   │   └── validation_runner.py
│   │
│   ├── transforms/
│   ├── relations/
│   ├── analysis/
│   ├── reports/
│   └── dashboard/
│
├── examples/
├── results/
│
├── README.md
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

---

## Current Limitations

- Transformation suite is primarily focused on image-based inputs
- Classification-specific metamorphic relations are still under development
- Comparator thresholds (`epsilon`) require manual tuning per task
- Runtime depends on model inference speed
- Large datasets may require longer execution times

---

## Future Work

- NLP and text transformation extensions
- Tabular data transformation support
- Classification-specific metamorphic relations
- Interactive Streamlit dashboard
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
