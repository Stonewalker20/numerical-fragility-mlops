import random
import torch

# src/config.py
# -------------------------------
# Controlled randomness
# -------------------------------
# This ensures that the randomly generated seeds
# are reproducible across runs of the config file.
random.seed(2026)

NUM_SEEDS = 20
SEEDS = random.sample(range(1, 10000), NUM_SEEDS)

# Expanded batch size sweep
BATCH_SIZES = [32, 64, 128, 256, 512]

DEVICE = "cpu"
PRECISION = "fp32"
PRECISION_ORDER = ["fp32", "amp"]

CONFIG_MATRIX = []

# -------------------------------
# Seed sweep (fixed batch size)
# -------------------------------
for s in SEEDS:
    CONFIG_MATRIX.append(
        {
            "tag": "seed_sweep",
            "seed": s,
            "device": DEVICE,
            "precision": PRECISION,
            "batch_size": 128,
        }
    )

# -------------------------------
# Batch size sweep (fixed seed)
# -------------------------------
BASELINE_SEED = SEEDS[0]

for bs in BATCH_SIZES:
    CONFIG_MATRIX.append(
        {
            "tag": "batch_sweep",
            "seed": BASELINE_SEED,
            "device": DEVICE,
            "precision": PRECISION,
            "batch_size": bs,
        }
    )

# -------------------------------
# Precision sweep (GPU-only)
# -------------------------------
# AMP is only meaningfully enabled in train.py for CUDA runs.
if torch.cuda.is_available():
    for precision in PRECISION_ORDER:
        CONFIG_MATRIX.append(
            {
                "tag": "precision_sweep",
                "seed": BASELINE_SEED,
                "device": "cuda",
                "precision": precision,
                "batch_size": 128,
            }
        )

if __name__ == "__main__":
    print("Generated SEEDS:", SEEDS)
    print("Total configs:", len(CONFIG_MATRIX))
