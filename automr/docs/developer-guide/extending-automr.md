# Extending AutoMR

## Overview

AutoMR is designed with a modular architecture that allows developers to extend functionality without modifying the core framework.

---

# Extension Points

Developers can extend:

- Model wrappers
- Input handlers
- Transformations
- Metamorphic relations
- Comparators
- Report generation

---

# Typical Extension Workflow

1. Implement the new component.
2. Register the component.
3. Verify functionality.
4. Execute metamorphic testing.
5. Review generated reports.

---

# Plugin Architecture

AutoMR uses registries to dynamically discover framework components during runtime.

This enables new functionality to be added without changing the execution engine.

---

# Development Guidelines

- Follow the existing module structure.
- Reuse registry APIs.
- Keep components independent.
- Maintain backward compatibility.

---

# Testing Extensions

New extensions should be validated by:

- Unit testing
- Small dataset execution
- Full AutoMR validation
- Report verification

---

# Recommended Directory

```
automr/
    transforms/
    registry/
    models/
    input_handlers/
```