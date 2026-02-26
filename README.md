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

While traditional evaluation focuses on accuracy and loss, this system quantifies **numerical instability and prediction drift under operational perturbations** such as random seed variation and batch size changes.

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
* Prepare models for stability gating in CI environments.

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

---

#### 2. Cross-Run Reproducibility Drift (Between-Run)

Sweeps across:

* Random seeds
* Batch sizes

For each sweep:

[
\text{Disagreement} =
\frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}(\hat{y}*{baseline} \neq \hat{y}*{compare})
]

Outputs exported to:

```
artifacts/comparisons_week2.csv
```

This quantifies prediction drift even when accuracy remains stable.

---

## Architecture Overview

```
src/
 ├── model.py
 ├── train.py
 ├── config.py
 └── stability logic

artifacts/
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

---

### Operational Perturbation Matrix

The configuration sweep includes:

* Randomized (reproducible) seed sweep
* Expanded batch size sweep
* Deterministic baseline selection

This simulates real-world configuration drift scenarios.

---

### CI-Ready Stability Infrastructure

The system is structured to support:

* Stability thresholds
* Build gating on numerical drift
* Environment-agnostic tracking backends

Future integration can fail CI when instability exceeds tolerance.

---

## Running the Project

### Install Dependencies

```
pip install -r requirements.txt
```

---

### Execute Training Sweep

```
python src/train.py
```

This will:

* Execute configuration matrix
* Log MLflow experiments
* Save prediction artifacts
* Compute cross-run disagreement
* Export comparison CSV

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
* Prediction-level disagreement is non-zero.
* Numerical perturbations introduce measurable instability.
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
* FP32 vs AMP precision sensitivity
* CI stability gating thresholds
* Gradient instability tracking
* Drift visualization dashboards
* Model registry stability metadata

---

## Author

Cordell Stonecipher
Machine Learning Engineer
Oakland University
