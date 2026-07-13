# Registry System

## Overview

The registry system provides a plugin architecture for metamorphic transformations and relations.

---

# Transformation Registry

Stores transformation implementations.

```python
registry.register(
    "brightness",
    BrightnessTransform()
)
```

---

# Relation Registry

Stores relation implementations.

```python
registry.register(
    "brightness",
    BrightnessRelation()
)
```

---

# Listing Components

```python
registry.list()
```

---

# Retrieving Components

```python
registry.get("brightness")
```

---

# Removing Components

```python
del registry.transforms["brightness"]
```

---

# Advantages

- Dynamic loading
- Plugin architecture
- Runtime extension
- Easy customization