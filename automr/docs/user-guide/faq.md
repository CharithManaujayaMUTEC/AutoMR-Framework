# Frequently Asked Questions

## Does AutoMR require labels?

No. AutoMR validates model behavior using metamorphic relations instead of ground-truth labels.

---

## Which frameworks are supported?

- PyTorch
- TensorFlow
- Keras
- ONNX Runtime
- Scikit-learn
- Custom Python models

---

## Can I create custom transformations?

Yes.

Use:

```python
automr.register_transform(...)
```

---

## Can AutoMR evaluate datasets?

Yes.

Use:

```python
automr.run_dataset(...)
```

or

```python
automr.run_full_test(...)
```

---

## What is epsilon?

Epsilon defines the allowable prediction difference before a violation is reported.

---

## What is samples_per_mr?

It specifies how many parameter values are sampled within a transformation's configured range for each metamorphic relation. Larger values provide finer coverage but increase execution time.

---

## Does AutoMR support GPU execution?

Yes.

GPU acceleration is available through the HighPerformanceAutoMR engine.

---

## Where are reports stored?

By default:

```
results/
```

---

## How do I extend AutoMR?

See the **Developer Guide** for architecture and extension points.