# Transformations

## Overview

Transformations modify input data while preserving expected model behavior according to a corresponding metamorphic relation.

AutoMR includes a collection of built-in image transformations and allows developers to register custom transformations.

---

# Built-in Transformations

## Image Processing

- Brightness
- Contrast
- Blur
- Rotation
- Translation
- Noise
- Composite

---

## Weather Effects

- Rain
- Snow
- Fog
- Haze
- Smoke
- Dust
- Sandstorm

---

## Visibility

- Darkness
- Visibility reduction

---

## Temporal

Temporal transformations evaluate consistency across sequences of related inputs.

---

# Registering a Custom Transformation

```python
automr.register_transform(
    name="custom",
    transform=CustomTransformation(),
    relation=CustomRelation(),
    param_range=(0.0, 1.0)
)
```

---

# Listing Available Transformations

```python
print(automr.list_transforms())
```

---

# Parameter Ranges

Each transformation defines a parameter range that determines the intensity or magnitude of the applied transformation during testing.

---

# Best Practices

- Choose realistic parameter ranges.
- Avoid transformations that destroy semantic information.
- Ensure the associated metamorphic relation matches the intended behavior.
- Validate custom transformations before large-scale evaluation.