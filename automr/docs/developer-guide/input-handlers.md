# Input Handlers

## Overview

Input handlers convert raw user inputs into the format required by the target model.

---

# Responsibilities

- Image preprocessing
- Tensor conversion
- Batch preparation
- Data normalization

---

# Supported Types

- Image
- Tensor
- NumPy array
- Custom formats

---

# Registering a Handler

```python
handler = get_handler("image")
```

---

# Typical Workflow

1. Load input
2. Resize
3. Normalize
4. Convert to tensor
5. Forward to model

---

# Custom Handler

```python
class MyInputHandler:

    def preprocess(self, data):
        return processed
```

---

# Recommendations

- Keep preprocessing deterministic.
- Ensure consistent output shapes.
- Minimize unnecessary copies.