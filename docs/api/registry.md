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

## Custom Extension Convenience Methods

`AutoMR` provides convenience methods on top of the existing transformation
and relation registry architecture:

```python
automr.register_custom_transformation(name, transform, param_range=None)
automr.register_custom_relation(name, relation)
automr.register_custom_mr(name, transform, relation, param_range)
```

The independent methods register only the requested transformation or
relation, with an optional range for a transformation. `register_custom_mr()`
registers the transformation in the transformation registry, the relation in
the relation registry, and the parameter configuration in `mr_ranges` under
the same MR name. This makes it available to the existing `run_mr()` and
`run_all_mrs()` execution logic without changing built-in transformations,
relations, registries, or execution behavior.