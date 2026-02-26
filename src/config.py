# src/config.py

import random

# -------------------------------
# Controlled randomness
# -------------------------------
# This ensures that the randomly generated seeds
# are reproducible across runs of the config file.
random.seed(2026)

NUM_SEEDS = 100
SEEDS = random.sample(range(1, 10000), NUM_SEEDS)

# Expanded batch size sweep
BATCH_SIZES = [32, 64, 128, 256, 512]

DEVICE = "cpu"
PRECISION = "fp32"

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

if __name__ == "__main__":
    print("Generated SEEDS:", SEEDS)
    print("Total configs:", len(CONFIG_MATRIX))