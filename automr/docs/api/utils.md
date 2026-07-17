# Utility API

## Overview

The utility module contains helper functions used throughout the AutoMR framework.

These utilities simplify common tasks such as configuration loading, parameter generation, data conversion, and execution support.

---

# Common Utilities

The utilities module provides functionality for:

- Configuration management
- Parameter generation
- Data conversion
- Logging support
- File handling
- Execution helpers

---

# Configuration Utilities

Example:

```python
from automr.utils import merge_config
```

---

# Parameter Utilities

Utilities are used internally for generating parameter ranges used during metamorphic testing.

---

# Logging Utilities

Helper functions provide standardized logging during execution.

---

# File Utilities

Utility functions assist with:

- Saving reports
- Loading datasets
- Managing output directories

---

# Internal Usage

Most utility functions are used internally by the framework and normally do not require direct interaction by end users.

---

# Best Practices

- Reuse existing utilities whenever possible.
- Avoid duplicating helper functions.
- Keep utility functions independent and reusable.