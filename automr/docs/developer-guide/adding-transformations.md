# Adding Transformations

## Overview

AutoMR allows developers to register custom metamorphic transformations without modifying the framework core.

Each transformation generates a follow-up input that will later be verified using a corresponding metamorphic relation.

---

# Step 1: Create the Transformation

```python
class MyTransformation:

    def __call__(self, input_data, parameter):
        return transformed_input
```

---

# Step 2: Create the Relation

```python
class MyRelation:

    def verify(self, original, transformed):
        ...
```

---

# Step 3: Register

```python
automr.register_transform(
    name="my_transform",
    transform=MyTransformation(),
    relation=MyRelation(),
    param_range=(0.0, 1.0)
)
```

---

# Parameter Range

Each transformation requires an operating range.

Example:

```python
param_range=(0.0, 1.0)
```

During testing, AutoMR samples parameter values from this interval.

---

# Verification

After registration:

```python
automr.list_transforms()
```

should include the new transformation.

---

# Recommendations

- Preserve semantic meaning whenever possible.
- Avoid changing image dimensions unless required.
- Keep execution deterministic.
- Define realistic parameter ranges.