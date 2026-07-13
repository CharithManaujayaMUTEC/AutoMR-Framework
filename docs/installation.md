# Installation Guide

## Overview

This guide explains how to install and configure AutoMR for both CPU and GPU environments.

---

# System Requirements

## Operating Systems

- Windows 10/11
- Ubuntu 20.04+
- macOS 13+

---

## Python

- Python 3.10 or newer

---

## Hardware

### Minimum

- Dual-core CPU
- 8 GB RAM
- 5 GB available storage

### Recommended

- 8+ CPU cores
- 16 GB RAM or higher
- NVIDIA GPU with CUDA support
- SSD storage

---

# Clone the Repository

```bash
git clone <repository-url>
cd AutoMR
```

---

# Create a Virtual Environment

## Windows

```bash
python -m venv venv

venv\Scripts\activate
```

## Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Optional GPU Support

Install the appropriate PyTorch version for your CUDA installation.

Example:

```bash
pip install torch torchvision torchaudio
```

Verify CUDA availability:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

# Verify Installation

Run:

```bash
python -c "import automr; print('AutoMR installed successfully.')"
```

---

# Verify Framework

Execute one of the example scripts:

```bash
python examples/basic_example.py
```

A successful execution indicates that the framework has been installed correctly.

---

# Common Installation Issues

## ModuleNotFoundError

Install missing dependencies:

```bash
pip install -r requirements.txt
```

---

## CUDA Not Detected

Verify:

- NVIDIA drivers
- CUDA Toolkit
- Compatible PyTorch version

---

## OpenCV Errors

Reinstall OpenCV:

```bash
pip install --upgrade opencv-python
```

---

# Next Step

Continue with the **Quick Start Guide** to execute your first AutoMR validation.