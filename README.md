# Correct but Fragile

### Numerical Stability Risks in Machine Learning Systems

**Author:** Cordell Stonecipher

---

## Overview

Machine learning systems are commonly evaluated using accuracy and loss metrics. While these confirm functional correctness, they do not guarantee numerical robustness, reproducibility, or deployment reliability.

This project investigates how machine learning models that appear correct can still exhibit fragile numerical behavior under realistic operational conditions such as:

* Hardware changes (CPU vs GPU)
* Precision changes (FP32 vs AMP)
* Batch size variation
* Controlled input perturbations
* Seed variation

The objective is to treat numerical instability as an **operational reliability issue**, not merely a modeling concern.

---

## Research Focus

This repository implements a controlled ML pipeline that:

1. Fixes dataset, architecture, and hyperparameters.
2. Executes training under explicit configuration matrices.
3. Tracks experiments using MLflow (SQLAlchemy backend).
4. Versions datasets and artifacts with DVC.
5. Logs stability metrics beyond accuracy.
6. Enables CI-style reproducibility testing.

The goal is to demonstrate that a model may maintain stable accuracy while exhibiting measurable numerical fragility.

---

## Architecture

Dataset → Preprocessing → Training → Stability Evaluation → Experiment Tracking → CI Validation

Core components:

* **PyTorch** – Model implementation and controlled precision execution
* **MLflow (SQLAlchemy backend)** – Experiment tracking and artifact management
* **DVC** – Dataset versioning and artifact traceability
* **Git / GitHub Actions** – CI-based reproducibility checks
* **Docker** – Environment standardization

---

## Numerical Stability Metrics (Week 1)

Beyond accuracy and loss, the pipeline logs:

* `stability_disagree_rate_eps1e-3`
  Fraction of predictions that change under small input perturbations.

* `stability_logit_var_mean_eps1e-3`
  Mean variance of model logits under perturbation.

* `train_seconds`
  Runtime variability signal.

* `cfg_hash`
  Configuration fingerprint for reproducibility verification.

These metrics quantify sensitivity to small numerical changes — a proxy for operational fragility.

---

## Repository Structure

```
numerical-fragility-mlops/
│
├── data/
│   └── raw/                 # Tracked via DVC
│
├── src/
│   ├── train.py             # Training + logging + stability metrics
│   ├── model.py             # TinyNet architecture
│   ├── stability.py         # Stability metric utilities
│   └── config.py            # Configuration matrix
│
├── .github/workflows/
│   └── ci.yml               # CI workflow
│
├── Dockerfile
├── requirements.txt
├── dvc.yaml
└── README.md
```

---

## Setup Instructions

### 1. Install dependencies

```
pip install -r requirements.txt
```

---

### 2. Start MLflow Server (SQLAlchemy backend)

```
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 127.0.0.1 \
  --port 5000
```

In another terminal:

```
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
```

---

### 3. Run Training

```
python src/train.py
```

Open browser:

```
http://127.0.0.1:5000
```

You should see experiment runs with stability metrics logged.

---

## CI Integration

The GitHub Actions workflow:

* Installs dependencies
* Runs a short baseline training
* Ensures reproducibility
* Validates pipeline execution

This enables automated regression checks for numerical stability.

---

## DVC Dataset Versioning

Dataset snapshots are tracked via DVC:

```
dvc add data/raw
git add data/raw.dvc
git commit -m "Track dataset snapshot"
```

This ensures dataset state is tied to specific Git commits.

---

## Reproducibility Goals (Next Phases)

Planned extensions include:

* Multi-seed reproducibility matrix
* CPU vs GPU comparison
* FP32 vs AMP stability comparison
* Batch size sensitivity analysis
* CI gates based on stability thresholds
* Cross-environment Docker validation

---

## Why This Matters

In production environments, silent numerical instability can lead to:

* Non-deterministic predictions
* Reproducibility gaps
* Deployment drift
* Model disagreement across hardware

Accuracy alone does not capture these risks.

This project reframes numerical behavior as an **AI operations reliability concern**.

---

## Status

Week 1 Complete:

* Deterministic baseline model
* MLflow SQLAlchemy backend
* DVC dataset versioning
* Stability metrics logging
* CI pipeline stub
* Docker reproducibility stub

---
