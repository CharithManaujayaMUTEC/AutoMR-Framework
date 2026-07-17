# Supported Models

## Overview

AutoMR is designed to work with a variety of machine learning and deep learning models through a unified wrapper interface.

---

# Currently Supported

## PyTorch

```python
model = load_model()

automr = AutoMR(
    model=model,
    task="classification"
)
```

---

## Regression Models

Supported through the regression comparator.

Examples include:

- Regression CNNs
- Lane detection networks
- Steering angle prediction
- Coordinate prediction models

---

## Classification Models

Examples include:

- ResNet
- VGG
- DenseNet
- MobileNet
- EfficientNet

---

## High Performance Testing

Large-scale dataset evaluation is supported using:

```python
HighPerformanceAutoMR(...)
```

---

# Planned Support

Future versions may include support for:

- TensorFlow
- Keras
- ONNX Runtime
- Scikit-learn
- Hugging Face models

---

# Wrapper Requirement

Models should expose prediction functionality through an AutoMR wrapper to ensure compatibility with the testing engine.