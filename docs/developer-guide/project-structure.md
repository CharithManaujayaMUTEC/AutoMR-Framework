# Project Structure

```
automr/
│
├── analysis/
├── comparators/
├── core/
├── dashboard/
├── epsilon/
├── evaluation/
├── hpc/
├── input_handlers/
├── logging/
├── models/
├── registry/
├── reporting/
├── transforms/
├── verification/
└── utils/
```

---

# Main Modules

## core

Core execution engine.

## hpc

High-performance parallel execution.

## registry

Transformation and relation registration.

## transforms

Built-in metamorphic transformations.

## comparators

Prediction comparison logic.

## models

Framework-specific model wrappers.

## input_handlers

Input preprocessing.

## evaluation

Baseline prediction generation.

## analysis

Failure and severity analysis.

## reporting

Result generation.

## verification

Transformation saving and verification.

## epsilon

Sensitivity analysis.

## dashboard

Interactive real-time testing components.

---

# Recommended Extension Strategy

- Add transformations in `transforms/`
- Add relations in `registry/`
- Add wrappers in `models/`
- Add handlers in `input_handlers/`