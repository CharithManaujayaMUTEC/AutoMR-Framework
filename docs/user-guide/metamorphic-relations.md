# Metamorphic Relations

## Overview

Metamorphic relations (MRs) define the expected relationship between a model's prediction on an original input and its prediction after a transformation.

Unlike traditional testing, AutoMR verifies these relationships without requiring ground-truth labels.

---

# Built-in Relations

| Relation | Description |
|----------|-------------|
| BrightnessRelation | Prediction should remain consistent under brightness changes. |
| ContrastRelation | Prediction should remain consistent under contrast adjustments. |
| BlurRelation | Moderate blur should not significantly affect predictions. |
| NoiseRelation | Small amounts of noise should preserve prediction stability. |
| RotationRelation | Small rotations should not substantially alter predictions. |
| TranslationRelation | Small translations should preserve prediction consistency. |
| CompositeRelation | Combined transformations should satisfy expected behavior. |
| FogRelation | Moderate fog should maintain acceptable prediction quality. |
| RainRelation | Rain effects should remain within tolerance. |
| SnowRelation | Snow effects should satisfy expected prediction behavior. |
| DustRelation | Dust conditions should not excessively degrade predictions. |
| HazeRelation | Haze should preserve acceptable outputs. |
| SmokeRelation | Smoke should satisfy defined tolerance. |
| SandstormRelation | Sandstorm effects are evaluated against configured thresholds. |
| DarkVisibilityRelation | Reduced visibility should exhibit predictable degradation. |
| TemporalRelation | Sequential frames should produce temporally consistent predictions. |

---

# Relation Types

- Invariant
- Monotonic
- Bounded
- Temporal

---

# Listing Registered Relations

```python
automr.list_relations()
```

---

# Accessing a Relation

```python
relation = automr.get_relation("brightness")
```

---

# Registering a Custom Relation

```python
automr.register_transform(
    name="custom",
    transform=transform_fn,
    relation=relation_fn,
    param_range=(0.0, 1.0)
)
```

---

# Best Practices

- Define clear expected behavior.
- Use realistic tolerances.
- Match parameter ranges to the transformation.
- Validate custom relations before deployment.