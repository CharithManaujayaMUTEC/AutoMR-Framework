# Adding Relations

## Overview

Every transformation should be paired with a corresponding metamorphic relation that defines the expected behavior of the model after transformation.

---

# Create a Relation

```python
class MyRelation:

    def verify(self, original, transformed):
        ...

    def expected(self):
        return "Expected behavior"
```

---

# Required Methods

## verify()

Determines whether the transformed prediction satisfies the expected behavior.

## expected()

Returns a human-readable description used in generated reports.

---

# Registration

Relations are registered together with their transformations.

```python
automr.register_transform(
    name="example",
    transform=ExampleTransform(),
    relation=ExampleRelation(),
    param_range=(0.0, 1.0)
)
```

---

# Validation

Registered relations can be listed using

```python
automr.list_relations()
```

or retrieved individually

```python
automr.get_relation("example")
```

---

# Best Practices

- Keep relations independent.
- Make expected behavior explicit.
- Support configurable epsilon where appropriate.
- Ensure deterministic verification.