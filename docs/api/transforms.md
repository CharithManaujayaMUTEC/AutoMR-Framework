# Transform API

## Overview

Transformations generate follow-up inputs for metamorphic testing.

---

# Required Interface

```python
class MyTransform:

    def __call__(
        self,
        input_data,
        parameter
    ):
        return transformed
```

---

# Registration

```python
automr.register_transform(
    name="example",
    transform=MyTransform(),
    relation=ExampleRelation(),
    param_range=(0.0, 1.0)
)
```

---

# Built-in Categories

- Image
- Weather
- Behaviour
- Temporal

---

# Recommendations

- Preserve semantic meaning.
- Keep output dimensions valid.
- Return compatible model inputs.