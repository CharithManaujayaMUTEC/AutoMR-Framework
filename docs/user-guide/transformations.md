# Transformations

## Overview

Transformations modify input data while preserving expected model behavior according to a corresponding metamorphic relation.

AutoMR includes 23 built-in transformations across image, global image, weather, behavioral, composite, and temporal categories, and also allows developers to register custom transformations.

---

# Built-in Transformations

The registered names below match `automr.list_transforms()`.

## Image

- `brightness`
- `contrast`
- `blur`
- `rotation`
- `translation`
- `noise`

## Global Image

- `global_brightness`
- `global_contrast`
- `global_blur`
- `global_noise`
- `global_rotation`
- `global_translation`

## Composite

- `composite`

## Weather

- `rain`
- `snow`
- `fog`
- `haze`
- `smoke`
- `dust`
- `sandstorm`

## Behavioral

- `darkness`
- `visibility`

## Temporal

- `temporal`

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