# Model Wrappers API

## Overview

Model wrappers provide a unified interface between AutoMR and different machine learning frameworks. They abstract framework-specific prediction logic, allowing the AutoMR engine to interact with all supported models using a consistent API.

---

# Purpose

The wrapper layer is responsible for:

- Standardizing model inference
- Managing CPU/GPU execution
- Performing batch inference
- Returning normalized prediction outputs

---

# Wrapper Responsibilities

Every wrapper should:

- Accept framework-specific models
- Execute inference
- Handle preprocessing if required
- Return predictions in a consistent format

---

# Basic Usage

```python
from automr.models import get_wrapper

wrapped_model = get_wrapper(model)
prediction = wrapped_model.predict(image)
```

---

# Batch Prediction

```python
predictions = wrapped_model.predict_batch(images)
```

---

# Supported Frameworks

Current implementations support:

- PyTorch

Future wrappers may include:

- TensorFlow
- Keras
- ONNX Runtime
- Scikit-learn

---

# Custom Wrapper Example

```python
class CustomWrapper:

    def predict(self, input_data):
        ...

    def predict_batch(self, batch):
        ...
```

---

# Best Practices

- Keep wrapper logic framework-specific.
- Return consistent prediction formats.
- Support batched inference whenever possible.