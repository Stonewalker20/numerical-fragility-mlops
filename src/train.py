# src/train.py
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import mlflow
import time, hashlib, json, os, csv
from pathlib import Path
from model import TinyNet
from config import CONFIG_MATRIX

# Use whatever the environment provides.
# - Local dev: export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
# - CI:        MLFLOW_TRACKING_URI="file:./mlruns_ci"
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
else:
    mlflow.set_tracking_uri("file:./mlruns")  # safe fallback

mlflow.set_experiment("numerical-fragility-week1")

def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Determinism controls
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Helps determinism for cuBLAS on CUDA (no-op on CPU/mac)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def get_data(batch_size: int) -> DataLoader:
    tfm = transforms.ToTensor()
    ds = datasets.MNIST(root="data/raw", train=True, download=True, transform=tfm)

    # Keep Week 1 runtime short: small fixed subset
    idx = list(range(0, 5000))
    sub = Subset(ds, idx)

    return DataLoader(sub, batch_size=batch_size, shuffle=True, num_workers=0)

def infer_fixed_eval(model, device: str, batch_size: int):
    """
    Runs inference on the fixed eval loader (same samples, same order).
    Returns:
      pred: (N,) predicted class indices
      logits: (N, C) raw logits
    """
    model.eval()
    dl = get_data(batch_size)

    all_logits = []
    all_pred = []

    with torch.no_grad():
        for x, _y in dl:
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.detach().cpu())
            all_pred.append(logits.argmax(dim=1).detach().cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    pred = torch.cat(all_pred, dim=0).numpy()
    return pred, logits

def stability_eval(model, device: str, batch_size: int, eps: float = 1e-3) -> tuple[float, float]:
    """
    Numerical stability signal via small input perturbation.

    Computes:
      - disagree_rate: fraction of argmax predictions that change between x and x+eps
      - logit_var_mean: mean variance of logits across the two passes

    Uses the same fixed eval set and order as infer_fixed_eval (via get_eval_data).
    """
    model.eval()
    dl = get_data(batch_size)

    all_logits_a = []
    all_logits_b = []
    all_pred_a = []
    all_pred_b = []

    with torch.no_grad():
        for x, _y in dl:
            x = x.to(device)

            logits_a = model(x)

            x_pert = (x + eps).clamp(0.0, 1.0)
            logits_b = model(x_pert)

            all_logits_a.append(logits_a.detach().cpu())
            all_logits_b.append(logits_b.detach().cpu())

            all_pred_a.append(logits_a.argmax(dim=1).detach().cpu())
            all_pred_b.append(logits_b.argmax(dim=1).detach().cpu())

    logits_a = torch.cat(all_logits_a, dim=0).numpy()  # (N, C)
    logits_b = torch.cat(all_logits_b, dim=0).numpy()
    pred_a = torch.cat(all_pred_a, dim=0).numpy()      # (N,)
    pred_b = torch.cat(all_pred_b, dim=0).numpy()

    disagree_rate = float((pred_a != pred_b).mean())

    stack = np.stack([logits_a, logits_b], axis=0)  # (2, N, C)
    logit_var_mean = float(np.var(stack, axis=0).mean())

    return disagree_rate, logit_var_mean

def log_prediction_artifacts(cfg_hash: str, cfg: dict, pred: np.ndarray, logits: np.ndarray, metrics: dict):
    """
    Writes local artifacts under artifacts/predictions/<cfg_hash>/ and logs them to MLflow.
    """
    run_dir = os.path.join("artifacts", "predictions", cfg_hash)
    os.makedirs(run_dir, exist_ok=True)

    np.save(os.path.join(run_dir, "pred.npy"), pred)
    np.save(os.path.join(run_dir, "logits.npy"), logits)

    summary = {
        "cfg_hash": cfg_hash,
        "cfg": cfg,
        "n_eval": int(pred.shape[0]),
        "metrics": metrics,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Log the whole directory to MLflow
    mlflow.log_artifacts(run_dir, artifact_path=f"predictions/{cfg_hash}")

def train_one(cfg: dict) -> tuple[float, float]:
    seed = int(cfg["seed"])
    device = str(cfg["device"])
    precision = str(cfg["precision"])
    batch_size = int(cfg["batch_size"])

    start_time = time.time()
    set_determinism(seed)

    dl = get_data(batch_size)
    model = TinyNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Unified AMP logic:
    # - Autocast exists for cpu/cuda, but GradScaler is mainly useful for fp16 CUDA.
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    use_amp = (precision == "amp") and (device_type == "cuda")

    # IMPORTANT: device type is positional for broad compatibility
    # Docs: torch.amp.GradScaler("cuda", ...) / torch.amp.GradScaler("cpu", ...) :contentReference[oaicite:2]{index=2}
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0

    for step, (x, y) in enumerate(dl):
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)

        # Docs use torch.autocast(device_type="cuda") :contentReference[oaicite:3]{index=3}
        with torch.autocast(device_type=device_type, enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        seen += int(x.size(0))

        if step >= 30:  # cap steps for Week 1
            break

    avg_loss = total_loss / max(seen, 1)
    acc = correct / max(seen, 1)
    elapsed = time.time() - start_time
    mlflow.log_metric("train_seconds", elapsed)
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    mlflow.log_param("cfg_hash", cfg_hash)
    
    disagree_rate, logit_var_mean = stability_eval(model, device, batch_size, eps=1e-3)
    mlflow.log_metric("stability_disagree_rate_eps1e-3", disagree_rate)
    mlflow.log_metric("stability_logit_var_mean_eps1e-3", logit_var_mean)

    pred_fixed, logits_fixed = infer_fixed_eval(model, device, batch_size)

    artifact_metrics = {
        "train_loss": float(avg_loss),
        "train_acc": float(acc),
        "train_seconds": float(elapsed),
        "stability_disagree_rate_eps1e-3": float(disagree_rate),
        "stability_logit_var_mean_eps1e-3": float(logit_var_mean),
    }

    log_prediction_artifacts(cfg_hash, cfg, pred_fixed, logits_fixed, artifact_metrics)
    return avg_loss, acc

def compute_cross_run_disagreement():
    """
    Compares prediction snapshots across runs.
    Creates artifacts/comparisons_week2.csv
    """
    pred_root = Path("artifacts/predictions")
    if not pred_root.exists():
        print("No prediction artifacts found.")
        return

    cfg_map = {}
    for cfg in CONFIG_MATRIX:
        cfg_hash = hashlib.sha256(
            json.dumps(cfg, sort_keys=True).encode()
        ).hexdigest()[:12]
        cfg_map[cfg_hash] = cfg

    # Group by tag
    grouped = {"seed_sweep": [], "batch_sweep": []}

    for run_dir in pred_root.iterdir():
        if run_dir.is_dir() and run_dir.name in cfg_map:
            cfg = cfg_map[run_dir.name]
            grouped[cfg["tag"]].append((run_dir.name, cfg))

    results = []

    for tag, runs in grouped.items():
        if len(runs) < 2:
            continue

        # Sort deterministically
        if tag == "seed_sweep":
            runs = sorted(runs, key=lambda x: x[1]["seed"])
        else:
            runs = sorted(runs, key=lambda x: x[1]["batch_size"])

        baseline_hash, baseline_cfg = runs[0]
        baseline_pred = np.load(pred_root / baseline_hash / "pred.npy")

        for other_hash, other_cfg in runs[1:]:
            other_pred = np.load(pred_root / other_hash / "pred.npy")

            disagree = float((baseline_pred != other_pred).mean())

            results.append({
                "sweep_type": tag,
                "baseline_hash": baseline_hash,
                "compare_hash": other_hash,
                "baseline_seed": baseline_cfg["seed"],
                "compare_seed": other_cfg["seed"],
                "baseline_batch": baseline_cfg["batch_size"],
                "compare_batch": other_cfg["batch_size"],
                "disagreement_rate": disagree,
            })

    out_path = Path("artifacts/comparisons_week3.csv")
    out_path.parent.mkdir(exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved cross-run comparison to {out_path}")

    # Log to MLflow (single artifact under current run)
    mlflow.log_artifact(str(out_path))

def main() -> None:
    mlflow.set_experiment("numerical-fragility-week3")

    for cfg in CONFIG_MATRIX:
        run_name = (
            f"seed={cfg['seed']}_dev={cfg['device']}_prec={cfg['precision']}_bs={cfg['batch_size']}"
        )
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(cfg)
            loss, acc = train_one(cfg)
            mlflow.log_metric("train_loss", loss)
            mlflow.log_metric("train_acc", acc)

            print(run_name, "loss=", loss, "acc=", acc)
    
    compute_cross_run_disagreement()
    
if __name__ == "__main__":
    main()
