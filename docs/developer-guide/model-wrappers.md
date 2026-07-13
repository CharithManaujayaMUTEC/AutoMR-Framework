# Model Wrappers

## Purpose

Model wrappers provide a unified prediction interface across different machine learning frameworks.

---

# Supported Frameworks

- PyTorch
- TensorFlow
- Keras
- ONNX Runtime
- Scikit-learn

---

# Required Interface

```python
predict(input_data)
```

or

```python
predict_batch(inputs)
```

---

# Wrapper Responsibilities

- Input preprocessing
- Device management
- Batch prediction
- Output normalization

---

# Custom Wrapper Example

```python
class MyWrapper:

    def predict(self, image):
        ...

    def predict_batch(self, images):
        ...
```

---

# Registration

```python
wrapped = get_wrapper(model)
```

---

# Best Practices

- Support batched inference.
- Avoid framework-specific logic outside the wrapper.
- Return consistent prediction formats.