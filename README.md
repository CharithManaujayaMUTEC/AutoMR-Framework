<div align="center">
  <img src="automrlogo.png" width="400"/>
  <br/><br/>
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=flat-square"/>
  <img src="https://img.shields.io/badge/Status-Active-22c55e?style=flat-square"/>
  <img src="https://img.shields.io/badge/Domain-Autonomous%20Driving-7c6fcd?style=flat-square"/>
  <br/><br/>
  <p>Model agnostic + Input agnostic + Output agnostic Metamorphic testing framework for regressional autonomous driving AI/ML models.</p>
</div>

---

## Overview

AutoMR evaluates ML models by verifying **metamorphic relations (MRs)** — expected behavioral properties that must hold under controlled input transformations. Instead of checking exact outputs against ground truth, AutoMR checks whether the model behaves _consistently_ when inputs are perturbed in predictable ways.

| Problem                  | What AutoMR does                                         |
| ------------------------ | -------------------------------------------------------- |
| No labeled data          | Tests models without any ground-truth labels             |
| Real-world perturbations | Measures robustness under realistic noise and conditions |
| Silent failures          | Pinpoints when and how models begin to fail              |

---

## Key Features

- **Model-agnostic** — works with TensorFlow, PyTorch, sklearn, or any custom model
- **Input-agnostic** — supports images, text, tabular, and sequential data
- **Output-agnostic** — handles regression and classification outputs
- **Built-in MR pipeline** — end-to-end execution with zero boilerplate
- **Parametric testing** — sweep transformation parameters across configurable ranges
- **Automated analysis** — failure rate, severity scores, and worst-case detection
- **Automatic CSV export** — all results persisted without manual intervention
- **Optional progress tracking** — live progress bars for long-running tests

---

## Installation

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

automr = AutoMR(model)

df, results = automr.run_full_test(
    dataset,
    max_samples=2000,
    samples_per_mr=5,
    show_progress=True
)
```

---

## Execution Flow

```
1. Load dataset       →  user-defined input source
2. Load model         →  any model exposing predict(x)
3. Apply transforms   →  brightness, rotation, noise, fog, ...
4. Generate outputs   →  original vs. transformed predictions
5. Validate MRs       →  check expected behavioral properties
6. Analyze results    →  failure rate, severity, worst cases
7. Export             →  CSV files written to /results
```

---

## Output Files

All results are saved automatically to the `/results` directory.

| File                   | Description                                |
| ---------------------- | ------------------------------------------ |
| `automr_results.csv`   | Full per-sample test log                   |
| `failure_summary.csv`  | Failure rate per metamorphic relation      |
| `severity_summary.csv` | Average output deviation per MR            |
| `worst_cases.csv`      | Samples with highest deviation             |
| `failure_regions.txt`  | Parametric boundaries where failures occur |

### Output columns

| Column              | Description                       |
| ------------------- | --------------------------------- |
| `mr`                | Metamorphic relation identifier   |
| `param`             | Transformation parameter value    |
| `original`          | Original model prediction         |
| `transformed`       | Prediction after transformation   |
| `difference`        | Raw output difference             |
| `percent_change`    | Percentage change between outputs |
| `status`            | `PASS` / `FAIL`                   |
| `expected_behavior` | Expected MR rule                  |
| `actual_behavior`   | `Consistent` / `Violation`        |
| `sample_id`         | Input sample index                |

---

## Metamorphic Relations

| Relation              | Description                          | Type       |
| --------------------- | ------------------------------------ | ---------- |
| `BrightnessRelation`  | Output invariant to lighting changes | Invariance |
| `RotationRelation`    | Stable under small rotations         | Invariance |
| `TranslationRelation` | Stable under image shifts            | Invariance |
| `NoiseRelation`       | Robust to random noise               | Robustness |
| `FogRelation`         | Robust to visibility degradation     | Robustness |
| `TemporalSmoothness`  | Consistent outputs across frames     | Monotonic  |

---

## Transformations

| Transform   | Description               |
| ----------- | ------------------------- |
| Brightness  | Adjust pixel intensity    |
| Rotation    | Rotate image by angle     |
| Translation | Shift image spatially     |
| Noise       | Add random Gaussian noise |
| Fog / Rain  | Simulate adverse weather  |
| Blur        | Apply smoothing filter    |

---

## Design Principles

### Model-agnostic

Works with any model that implements a `predict(x)` interface:

```python
# TensorFlow, PyTorch, sklearn, or fully custom — all compatible
output = model.predict(input)
```

### Input-agnostic

Accepts any input type — images, sequences, tabular data, or custom formats. Transformations are applied modularly and do not depend on input structure.
AutoMR does not perform preprocessing.

Users must provide inputs in the format expected by their model.

This ensures AutoMR evaluates the original model pipeline without modification.

### Modular architecture

| Component   | Role                                   |
| ----------- | -------------------------------------- |
| `Model`     | Generates predictions                  |
| `Transform` | Modifies input samples                 |
| `Relation`  | Defines expected behavioral properties |
| `Analyzer`  | Computes failure metrics and summaries |

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
│   │   └── failure_analysis.py
│   │
│   ├── relations/
│   ├── transforms/
│   └── analysis/
│
├── run_test_example.py
├── requirements.txt
├── .gitignore
└── automrlogo.png
```

---

## Example Run

```
Running AutoMR: ██████████████ 100%

=== AutoMR Results ===
Failure Summary:
  BrightnessRelation   →  12.4% failure rate  |  avg deviation: 0.031
  RotationRelation     →   8.7% failure rate  |  avg deviation: 0.019
  NoiseRelation        →  21.1% failure rate  |  avg deviation: 0.074
  FogRelation          →  34.2% failure rate  |  avg deviation: 0.112

DONE: Results saved in /results
```

---

## Built-in Analysis

AutoMR automatically computes the following after each test run:

- **Failure rate** per metamorphic relation
- **Severity** — average output deviation across failures
- **Worst-case failures** — samples with the largest deviations
- **Failure regions** — parameter ranges where the model is most unstable

---

## Limitations

- Current transformation suite is image-focused
- Comparator thresholds require manual tuning per task
- End-to-end performance depends on model inference speed

---

## Future Work

- NLP and tabular transformation extensions
- Classification-specific comparators
- Streamlit dashboard for interactive analysis
- Cross-model MR testing
- Automated result visualizations (plots and charts)

---

## Authors

**CharithManaujayaMUTEC** — [github.com/CharithManaujayaMUTEC](https://github.com/CharithManaujayaMUTEC)  
**RaveeshaPeiris** — [github.com/RaveeshaPeiris](https://github.com/RaveeshaPeiris)

> Final Year Project — Metamorphic Testing Framework for Regressional Based Autonomous Driving AI/ML Models

---

## License

Released under the [MIT License](LICENSE).
