from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
COMPARISON_CSV = ARTIFACTS_DIR / "comparisons_week4.csv"


def main() -> None:
    df = pd.read_csv(COMPARISON_CSV)

    seed_df = df[df["sweep_type"] == "seed_sweep"]
    batch_df = df[df["sweep_type"] == "batch_sweep"]
    precision_df = df[df["sweep_type"] == "precision_sweep"]

    plt.figure()
    plt.plot(seed_df["compare_seed"], seed_df["disagreement_rate"], marker="o")
    plt.xlabel("Seed")
    plt.ylabel("Disagreement Rate")
    plt.title("Prediction Drift Across Seeds")
    plt.grid()
    plt.savefig(ARTIFACTS_DIR / "seed_disagreement.png")
    plt.close()

    plt.figure()
    plt.plot(batch_df["compare_batch"], batch_df["disagreement_rate"], marker="o")
    plt.xlabel("Batch Size")
    plt.ylabel("Disagreement Rate")
    plt.title("Prediction Drift Across Batch Sizes")
    plt.grid()
    plt.savefig(ARTIFACTS_DIR / "batch_disagreement.png")
    plt.close()

    if not precision_df.empty:
        plt.figure()
        labels = [
            f"{row['baseline_precision']}->{row['compare_precision']}"
            for _, row in precision_df.iterrows()
        ]
        plt.bar(labels, precision_df["disagreement_rate"])
        plt.xlabel("Precision Comparison")
        plt.ylabel("Disagreement Rate")
        plt.title("Prediction Drift Across Precision Modes")
        plt.grid(axis="y")
        plt.savefig(ARTIFACTS_DIR / "precision_disagreement.png")
        plt.close()

    print(f"Plots saved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
