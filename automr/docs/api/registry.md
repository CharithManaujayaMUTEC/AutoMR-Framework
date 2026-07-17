# Registry API

## Overview

The registry system manages transformations and metamorphic relations.

---

# Transformation Registry

## register()

```python
registry.register(
    name,
    transformation
)
```

---

## get()

```python
registry.get(name)
```

---

## list()

```python
registry.list()
```

---

# Relation Registry

The relation registry exposes the same interface.

```python
register()

get()

list()
```

---

# AutoMR Helpers

```python
automr.register_transform()

automr.unregister_transform()

automr.get_transform()

automr.get_relation()

automr.list_transforms()

automr.list_relations()
```

---

# Purpose

The registry architecture enables AutoMR to support runtime extensibility without modifying the framework core.