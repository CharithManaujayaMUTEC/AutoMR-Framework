# Quick Start

## Create an AutoMR Instance

```python
from automr import AutoMR

automr = AutoMR(
    model=model,
    task="classification",
    input_type="image"
)
```

---

## Load a Dataset

```python
dataset = MyDataset("dataset/")
```

---

## Run Metamorphic Testing

```python
df, results = automr.run_full_test(
    dataset=dataset
)
```

---

## Generated Outputs

The framework automatically produces:

- automr_results.csv
- failure_summary.csv
- severity_summary.csv
- prediction_trace.csv
- range_summary.csv
- range_analysis.csv

Optional outputs include:

- epsilon_summary.csv
- epsilon_report.txt

---

## Next Steps

- Configure custom transformations.
- Register custom metamorphic relations.
- Explore HPC execution for large datasets.
- Review the tutorials for end-to-end examples.