# Relations API

## Overview

Metamorphic relations define the expected relationship between predictions made on the original input and its transformed version.

---

# Required Interface

```python
class ExampleRelation:

    def verify(
        self,
        original_prediction,
        transformed_prediction
    ):
        ...
```

---

# Registration

```python
automr.register_transform(
    name="brightness",
    transform=BrightnessTransform(),
    relation=BrightnessRelation(),
    param_range=(0.1, 3.0)
)
```

---

# Built-in Relation Categories

## Invariant Relations

Prediction should remain approximately unchanged.

Examples:

- Blur
- Translation
- Contrast

---

## Monotonic Relations

Prediction should consistently increase or decrease.

Examples:

- Brightness
- Visibility
- Darkness

---

## Temporal Relations

Prediction consistency across sequential frames.

---

# Expected Output

Each relation returns:

- PASS
- FAIL

along with prediction difference and severity.