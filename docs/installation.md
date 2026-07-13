# Installation

## Requirements

- Python 3.10+
- pip
- Git

Recommended:

- NVIDIA GPU with CUDA support
- PyTorch with CUDA

---

# Clone Repository

```bash
git clone <repository-url>
cd AutoMR-Framework
```

---

# Create Virtual Environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Verify Installation

```bash
python -c "import automr; print('AutoMR installed successfully.')"
```

---

# Optional GPU Support

Install the CUDA-compatible version of PyTorch according to your CUDA version.

Refer to the official PyTorch installation guide for compatible installation commands.

---

# Troubleshooting

- Verify Python version.
- Ensure all dependencies are installed.
- Confirm CUDA is available if GPU execution is required.
- Check that model weights and datasets are correctly configured.