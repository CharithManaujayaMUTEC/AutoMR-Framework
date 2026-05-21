# 🚗 AutoMR

![AutoMR Logo](automrlogo.png)

AutoMR is a **model-agnostic, input-agnostic, and output-agnostic metamorphic testing framework** designed to evaluate rgressional based autonomous driving machine learning/AI models **without requiring ground-truth labels**.

Instead of checking exact outputs, AutoMR verifies **metamorphic relations (MRs)** expected behaviors under controlled input transformations.

---

# 🎯 Objective

This project addresses:

- How to test ML models **without labeled data**
- How robust models are under **real-world perturbations**
- When and how models **start to fail**

---

# ✨ Key Features

- Model-agnostic (TensorFlow, PyTorch, sklearn, custom)
- Input-agnostic (images, text, tabular)
- Output-agnostic (regression, classification)
- Built-in MR execution pipeline
- Parametric testing (range-based MR sweeps)
- Automated analysis (failure rate, severity, worst cases)
- Automatic CSV export
- Optional progress tracking

---

# 🏗️ Project Structure

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
│   │
│   ├── relations/
│   ├── transforms/
│   ├── analysis/
│
├── run_test_example.py
├── requirements.txt
├── .gitignore
├── automrlogo.png
```

---

# ⚙️ Installation

```bash
git clone https://github.com/CharithManaujayaMUTEC/AutoMR-Framework.git
cd AutoMR-Framework

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

---

# 🚀 Quick Start (Recommended)

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

# 🔄 Execution Flow

1. Load dataset (user-defined)
2. Load model (user-defined)
3. AutoMR:
   - Applies transformations
   - Generates predictions
   - Validates metamorphic relations
4. Computes:
   - Failure rate
   - Severity
   - Worst-case failures
5. Saves results automatically

---

# 📊 Output

Generated files (in `/results` folder):

```
automr_results.csv
failure_summary.csv
severity_summary.csv
worst_cases.csv
failure_regions.txt
```

---

## 📋 Output Columns

| Column            | Description            |
| ----------------- | ---------------------- |
| mr                | Metamorphic relation   |
| param             | Transformation value   |
| original          | Original prediction    |
| transformed       | Transformed output     |
| difference        | Output difference      |
| percent_change    | % change               |
| status            | PASS / FAIL            |
| expected_behavior | Expected MR rule       |
| actual_behavior   | Consistent / Violation |
| sample_id         | Input index            |

---

# 🧪 Built-in Analysis

AutoMR automatically computes:

- Failure rate per MR
- Severity (average deviation)
- Worst-case failures
- Failure regions

---

# 🔁 Metamorphic Relations (Examples)

| MR                  | Description                  |
| ------------------- | ---------------------------- |
| BrightnessRelation  | Output invariant to lighting |
| RotationRelation    | Stable under small rotations |
| TranslationRelation | Stable under shifts          |
| NoiseRelation       | Robust to noise              |
| FogRelation         | Robust to visibility changes |
| TemporalSmoothness  | Consistency across frames    |

---

# 🧩 Transformations

| Transform   | Description            |
| ----------- | ---------------------- |
| Brightness  | Adjust pixel intensity |
| Rotation    | Rotate image           |
| Translation | Shift image            |
| Noise       | Add random noise       |
| Fog/Rain    | Simulate weather       |
| Blur        | Apply smoothing        |

---

# 🧠 Design Principles

### ✔ Model Agnostic

Works with any model implementing:

```python
predict(x)
```

### ✔ Input Agnostic

Supports any input type (images, sequences, etc.)

### ✔ Modular Architecture

| Component | Role               |
| --------- | ------------------ |
| Model     | Prediction         |
| Transform | Input modification |
| Relation  | Expected behavior  |
| Analyzer  | Failure analysis   |

---

# ⚠️ Limitations

- Current transformations are image-focused
- Comparator tuning required per task
- Performance depends on model speed

---

# 🔮 Future Work

- NLP and tabular extensions
- Classification-specific comparators
- Streamlit dashboard
- Cross-model MR testing
- Automated visualization (plots)

---

# 🧪 Example Run

```
Running AutoMR: ██████████████ 100%

=== AutoMR Results ===
Failure Summary:
...

DONE: Results saved in /results
```

---

# 👨‍💻 Authors

by
CharithManaujayaMUTEC - https://github.com/CharithManaujayaMUTEC
RaveeshaPeiris - https://github.com/RaveeshaPeiris

for our Final Year Project — Metamorphic Testing Framework for Regressional Based Autonomous Driving AI/ML Models

---

# 📜 License

MIT License
