# 🚗 AutoMR — Metamorphic Testing Framework

AutoMR is a **model-agnostic, input-agnostic, and output-agnostic metamorphic testing framework** designed to evaluate machine learning models **without requiring ground-truth labels**.

Instead of checking exact outputs, AutoMR verifies **metamorphic relations (MRs)** — expected behaviors under controlled input transformations.

---

# 🎯 Objective

This project addresses:

- How to test ML models **without labeled data**
- How robust models are under **real-world perturbations**
- When and how models **start to fail**

---

# Key Features

- Model-agnostic (TensorFlow, PyTorch, sklearn, custom)
- Input-agnostic (images, text, tabular)
- Output-agnostic (regression, classification)
- Comparator-based evaluation
- Parametric testing (range-based MR sweeps)
- CSV/JSON export
- Progress tracking (tqdm)

---

# 🧠 Core Idea

Instead of:

```
f(x) == expected_output ❌
```

We test:

```
f(x) ≈ f(T(x))
```

Where `T(x)` is a transformed input.

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
│   │   ├── tester.py
│   │   ├── range_tester.py
│   │
│   ├── relations/
│   │   └── image_relations.py
│   │
│   ├── transforms/
│   │   └── image_transforms.py
│   │
│   ├── analysis/
│   │   └── analyzer.py
│
├── run_test.py
├── requirements.txt
├── .env
├── .env.example
```

---

# ⚙️ Installation

## 1. Clone repository

```bash
git clone <your-repo-url>
cd AutoMR-Framework
```

## 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Setup

Create `.env` file:

```env
DATASET_PATH=D:/FYP 78SEm/Datasets/archive/trafic_data/train/images
MODEL_PATH=D:/FYP 78SEm/Modals/nvidia_model.h5
SAMPLES=5
EPSILON=0.1
```

---

# 🚀 How to Run

```bash
python run_test.py
```

---

# 🔄 Execution Flow

1. Load dataset
2. Load model
3. For each input:
   - Apply transformations
   - Predict outputs
   - Compare results
   - Evaluate MR
4. Store results
5. Export CSV

---

# 📊 Output

Generated file:

```
automr_results_detailed.csv
```

---

## 📋 Output Columns

| Column            | Description            |
| ----------------- | ---------------------- |
| mr                | Metamorphic relation   |
| param             | Transformation value   |
| original          | Original prediction    |
| transformed       | Transformed prediction |
| difference        | Output difference      |
| percent_change    | % change               |
| status            | PASS / FAIL            |
| expected_behavior | Expected MR rule       |
| actual_behavior   | Consistent / Violation |
| sample_id         | Input index            |

---

# 🧪 Comparator (Core Design)

```python
class RegressionComparator:
    def __init__(self, epsilon=0.05):
        self.epsilon = epsilon

    def compare(self, y1, y2):
        diff = abs(y1 - y2)
        passed = diff < self.epsilon
        return diff, passed
```

---

# 🔁 Metamorphic Relations

| MR                  | Description                    |
| ------------------- | ------------------------------ |
| BrightnessRelation  | Output invariant to brightness |
| RotationRelation    | Stable under rotation          |
| TranslationRelation | Stable under shift             |
| NoiseRelation       | Robust to noise                |

---

# 🧩 Transformations

| Transform   | Description            |
| ----------- | ---------------------- |
| Brightness  | Adjust pixel intensity |
| Rotation    | Rotate image           |
| Translation | Shift image            |
| Noise       | Add random noise       |

---

# 🧠 Design Principles

### Input Agnostic

Supports any input type.

### Output Agnostic

Supports regression, classification, etc.

### Modular Architecture

| Component  | Role               |
| ---------- | ------------------ |
| Model      | Prediction         |
| Transform  | Input modification |
| Relation   | Expected behavior  |
| Comparator | Pass/fail decision |

---

# ⚠️ Limitations

- Current transforms are image-focused
- Comparator must be defined per task
- Performance depends on model speed

---

# 🔮 Future Work

- NLP support
- Classification comparator
- Streamlit dashboard
- Cross-model analysis
- Failure region visualization

---

# 🧪 Example Run

```
Running AutoMR: ██████████████ 100%
DONE: automr_results_detailed.csv generated
```

---

# 👨‍💻 Authors

Final Year Project — Metamorphic Testing Framework

---

# 📜 License

MIT License
