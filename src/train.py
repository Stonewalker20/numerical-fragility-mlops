# src/train.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import mlflow

from model import TinyNet
from config import CONFIG_MATRIX


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


def train_one(cfg: dict) -> tuple[float, float]:
    seed = int(cfg["seed"])
    device = str(cfg["device"])
    precision = str(cfg["precision"])
    batch_size = int(cfg["batch_size"])

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
    return avg_loss, acc


def main() -> None:
    mlflow.set_experiment("numerical-fragility-week1")

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


if __name__ == "__main__":
    main()
