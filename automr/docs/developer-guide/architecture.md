# Architecture

## Overview

AutoMR is designed as a modular and extensible metamorphic testing framework for machine learning and deep learning models. The architecture separates model interaction, input preprocessing, metamorphic transformations, relation verification, execution, analysis, and reporting into independent components.

---

# High-Level Architecture

```
                  +----------------------+
                  |      User Model      |
                  +----------+-----------+
                             |
                             v
                    +------------------+
                    |  Model Wrapper   |
                    +------------------+
                             |
                             v
                    +------------------+
                    | Input Handler    |
                    +------------------+
                             |
                             v
                  +-----------------------+
                  | AutoMR Core Engine    |
                  +-----------------------+
                    |      |        |
                    |      |        |
                    v      v        v
            Transform   Relation  Comparator
             Registry   Registry
                    |
                    v
              Range Tester
                    |
                    v
               Result Analyzer
                    |
                    v
              Report Generator
```

---

# Live Dashboard Architecture

The interactive dashboard evaluates one transformation parameter per frame for responsive visualization. Full parameter sweeps remain available through Benchmark Mode.

```
Camera / Video
       │
       ▼
CameraSource
       │
       ▼
Frame Capture
       │
       ▼
Current MR
       │
       ▼
Current Intensity
       │
       ▼
Transformation
       │
       ▼
Prediction
       │
       ▼
Relation Evaluation
       │
       ▼
Dashboard Rendering
```

---

# Core Components

- AutoMR Engine
- Model Wrapper
- Input Handler
- Transformation Registry
- Relation Registry
- Comparator
- Range Tester
- Analyzer
- Report Generator
- Graph Generator
- HPC Execution Engine

---

# Design Principles

- Modular
- Extensible
- Plugin-based
- Framework-independent
- GPU compatible
- Dataset scalable

---

# Execution Pipeline

1. Load model
2. Wrap model
3. Register transformations
4. Register relations
5. Generate transformed inputs
6. Predict outputs
7. Verify metamorphic relations
8. Compute statistics
9. Generate reports and graphs