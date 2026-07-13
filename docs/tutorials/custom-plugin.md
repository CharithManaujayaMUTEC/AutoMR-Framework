# Tutorial: Creating a Custom Plugin

## Overview

AutoMR allows developers to extend the framework by registering custom transformations and metamorphic relations.

---

# Step 1: Create a Transformation

```python
class InvertTransformation:

    def __call__(self, image, strength):
        return 255 - image
```

---

# Step 2: Create a Relation

```python
class InvertRelation:

    def expected(self):
        return "Prediction should remain invariant."

    def verify(self, original, transformed):
        return abs(original - transformed) <= 0.05
```

---

# Step 3: Register the Plugin

```python
automr.register_transform(
    name="invert",
    transform=InvertTransformation(),
    relation=InvertRelation(),
    param_range=(0.0, 1.0)
)
```

---

# Step 4: Execute Testing

```python
df, results = automr.run_full_test(
    dataset=dataset
)
```

---

# Verify Registration

```python
print(automr.list_transforms())
print(automr.list_relations())
```

---

# Best Practices

- Register both the transformation and relation together.
- Define meaningful parameter ranges.
- Keep transformations deterministic where possible.
- Ensure relation verification matches the intended metamorphic property.