# Correct but Fragile

## Numerical Stability Risks in Machine Learning Systems

![CI](https://img.shields.io/badge/CI-GitHub%20Actions-success)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue)
![DVC](https://img.shields.io/badge/Data%20Versioning-DVC-purple)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Executive Summary

This project explores a critical reliability question in modern machine learning systems:

> Can a model be correct yet operationally fragile?

While traditional evaluation focuses on accuracy and loss, this system quantifies **numerical instability and prediction drift under operational perturbations** such as random seed variation, batch size changes, and precision mode differences.

The result is a reproducible ML pipeline that treats numerical behavior as a first-class reliability signal — aligning machine learning with DevOps and AIOps principles.

---

## Why This Matters

In production systems:

* Small configuration changes can alter predictions.
* Hardware or precision differences can introduce silent drift.
* CI pipelines validate correctness but rarely validate stability.

This project demonstrates how to:

* Detect prediction-level drift.
* Measure reproducibility gaps.
* Log numerical behavior as an operational signal.
* Enforce selected stability checks inside CI environments.

---

## Technical Highlights

### Deterministic Reproducibility Controls

* Controlled random seeds
* `torch.use_deterministic_algorithms`
* Fixed evaluation subset
* Config hashing for traceability

---

### Stability Metrics

#### 1. Perturbation Sensitivity (Within-Run)

Small numerical perturbation:

[
x' = x + \epsilon
]

Metrics logged:

* Prediction disagreement rate
* Mean logit variance

The training run can fail immediately when perturbation disagreement exceeds
the configured threshold:

```
MAX_PERTURB_DISAGREE
```

---

#### 2. Cross-Run Reproducibility Drift (Between-Run)

Sweeps across:

* Random seeds
* Batch sizes
* Precision mode (`fp32` vs `amp`, when CUDA is available)

For each sweep:

[
\text{Disagreement} =
\frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}(\hat{y}*{baseline} \neq \hat{y}*{compare})
]

Outputs exported to:

```
artifacts/comparisons_week4.csv
```

This quantifies prediction drift even when accuracy remains stable.

Cross-run gating is also supported through:

```
MAX_CROSS_RUN_DISAGREE
```

This gate is opt-in so that configuration sweeps can be observed without
forcing all intentionally different runs to satisfy the same tolerance.

---

## Architecture Overview

```
src/
 ├── model.py
 ├── train.py
 ├── config.py
 └── stability logic

artifacts/
 ├── comparisons_week4.csv
 ├── seed_disagreement.png
 ├── batch_disagreement.png
 ├── precision_disagreement.png
 └── predictions/
       └── <cfg_hash>/
            ├── pred.npy
            ├── logits.npy
            └── summary.json
```

### Tooling Stack

* **PyTorch** — deterministic model training
* **MLflow** — experiment + artifact tracking
* **DVC** — dataset version control
* **GitHub Actions** — clean-room validation
* **Docker** — environment reproducibility foundation

---

## Key Engineering Features

### Artifact-Backed Experimentation

Each run stores:

```
artifacts/predictions/<cfg_hash>/
 ├── pred.npy
 ├── logits.npy
 └── summary.json
```

This enables:

* Offline reproducibility checks
* Cross-run comparison without retraining
* Auditable experiment lineage
* Plot generation directly from saved comparison artifacts

---

### Operational Perturbation Matrix

The configuration sweep includes:

* Randomized (reproducible) seed sweep
* Expanded batch size sweep
* GPU precision sweep for `fp32` vs `amp`
* Deterministic baseline selection

This simulates real-world configuration drift scenarios.

---

### CI-Ready Stability Infrastructure

The system is structured to support:

* Perturbation stability thresholds
* Optional cross-run gating
* Build gating on numerical drift
* SQLite-backed local and CI experiment tracking

The CI workflow currently enforces perturbation stability while leaving
cross-run drift available as an explicit policy choice.

---

## Running the Project

### Install Dependencies

```
python3 -m pip install -r requirements.txt
```

Or with Conda:

```
conda env create -f environment.yml
conda activate numerical-fragility-mlops
```

---

### Execute Training Sweep

```
python3 src/train.py
```

This will:

* Execute configuration matrix
* Log MLflow experiments
* Save prediction artifacts
* Compute cross-run disagreement
* Export comparison CSV

By default, local runs use:

```
sqlite:///mlflow.db
```

You can still override the backend explicitly with `MLFLOW_TRACKING_URI`.

---

### Stability Thresholds

```
export MAX_PERTURB_DISAGREE="0.05"
export MAX_CROSS_RUN_DISAGREE="0.10"
python3 src/train.py
```

`MAX_PERTURB_DISAGREE` is active by default.
`MAX_CROSS_RUN_DISAGREE` is optional and is best used for tightly controlled
like-for-like comparisons rather than broad exploratory sweeps.

---

### Generate Plots

```
python3 src/plot_results.py
```

This writes:

* `artifacts/seed_disagreement.png`
* `artifacts/batch_disagreement.png`
* `artifacts/device_disagreement.png` when a CUDA-backed device sweep is present
* `artifacts/precision_disagreement.png` when a GPU precision sweep is present

### Generate Summary Tables

```
python3 src/generate_results_table.py
```

This writes:

* `artifacts/results_summary_table.csv`
* `artifacts/results_summary_table.md`
* `artifacts/results_detail_table.csv`
* `artifacts/results_detail_table.md`

### Reproduce With DVC

```
dvc repro
```

This runs the training sweep, regenerates plots, and refreshes the summary tables.

### Generate CUDA Device / Precision Data

On a CUDA-enabled machine, run:

```bash
RUN_MODE=gpu_only python3 src/train.py
python3 src/plot_results.py
python3 src/generate_results_table.py
```

This adds:

* `device_sweep` comparisons for CPU vs CUDA in `artifacts/comparisons_week4.csv`
* `precision_sweep` comparisons for `fp32` vs `amp`
* `artifacts/device_disagreement.png`
* `artifacts/precision_disagreement.png`

---

### Launch MLflow UI

```
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
mlflow ui
```

Open:

```
http://127.0.0.1:5000
```

---

## Results Interpretation

Key insight observed:

* Accuracy remains relatively stable across seeds and batch sizes.
* Precision changes can also induce prediction drift when AMP is enabled.
* Prediction-level disagreement is non-zero.
* Numerical perturbations introduce measurable instability.
* Stability thresholds turn those signals into enforceable engineering policy.
* Correctness alone does not capture operational robustness.

---

## Professional Value

This project demonstrates:

* Systems-level thinking in ML engineering
* Experiment reproducibility best practices
* CI-aligned reliability mindset
* Artifact-based experiment design
* Quantitative drift analysis
* ML observability implementation

It bridges:

Machine Learning
DevOps
Reliability Engineering
AIOps

---

## Future Extensions

* CPU vs GPU reproducibility comparison
* CI gating tuned to hardware-specific precision sweeps
* Gradient instability tracking
* Drift visualization dashboards
* Model registry stability metadata

---

## Author

Cordell Stonecipher
Machine Learning Engineer
Oakland University
