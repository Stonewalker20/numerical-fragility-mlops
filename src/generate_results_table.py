from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
COMPARISON_CSV = ARTIFACTS_DIR / "comparisons_week4.csv"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
SUMMARY_CSV = ARTIFACTS_DIR / "results_summary_table.csv"
SUMMARY_MD = ARTIFACTS_DIR / "results_summary_table.md"
DETAIL_CSV = ARTIFACTS_DIR / "results_detail_table.csv"
DETAIL_MD = ARTIFACTS_DIR / "results_detail_table.md"


def load_run_summaries() -> pd.DataFrame:
    rows = []
    for summary_path in sorted(PREDICTIONS_DIR.glob("*/summary.json")):
        payload = json.loads(summary_path.read_text())
        cfg = payload["cfg"]
        metrics = payload["metrics"]
        rows.append(
            {
                "cfg_hash": payload["cfg_hash"],
                "tag": cfg["tag"],
                "seed": cfg["seed"],
                "device": cfg["device"],
                "precision": cfg["precision"],
                "batch_size": cfg["batch_size"],
                "train_loss": metrics["train_loss"],
                "train_acc": metrics["train_acc"],
                "train_seconds": metrics["train_seconds"],
                "stability_disagree_rate_eps1e-3": metrics["stability_disagree_rate_eps1e-3"],
                "stability_logit_var_mean_eps1e-3": metrics["stability_logit_var_mean_eps1e-3"],
            }
        )

    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame, float_digits: int = 4) -> pd.DataFrame:
    table = df.copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: f"{x:.{float_digits}f}")
    return table


def main() -> None:
    comparisons = pd.read_csv(COMPARISON_CSV)
    runs = load_run_summaries()
    cfg_maps = {
        "seed": runs.set_index("cfg_hash")["seed"].to_dict(),
        "batch": runs.set_index("cfg_hash")["batch_size"].to_dict(),
        "precision": runs.set_index("cfg_hash")["precision"].to_dict(),
        "device": runs.set_index("cfg_hash")["device"].to_dict(),
    }
    run_metrics = runs[
        [
            "cfg_hash",
            "train_loss",
            "train_acc",
            "train_seconds",
            "stability_disagree_rate_eps1e-3",
            "stability_logit_var_mean_eps1e-3",
        ]
    ]

    detail = comparisons.merge(
        run_metrics.add_prefix("baseline_"),
        left_on="baseline_hash",
        right_on="baseline_cfg_hash",
        how="left",
    ).merge(
        run_metrics.add_prefix("compare_"),
        left_on="compare_hash",
        right_on="compare_cfg_hash",
        how="left",
    )

    for side in ("baseline", "compare"):
        hash_col = f"{side}_hash"
        for field, source in (
            ("seed", "seed"),
            ("batch", "batch"),
            ("precision", "precision"),
            ("device", "device"),
        ):
            col = f"{side}_{field}"
            if col not in detail.columns:
                detail[col] = detail[hash_col].map(cfg_maps[source])
            else:
                fill_mask = detail[col].isna() | (detail[col] == "")
                detail.loc[fill_mask, col] = detail.loc[fill_mask, hash_col].map(cfg_maps[source])

    summary = (
        detail.groupby("sweep_type", dropna=False)
        .agg(
            comparisons=("disagreement_rate", "count"),
            mean_disagreement=("disagreement_rate", "mean"),
            max_disagreement=("disagreement_rate", "max"),
            baseline_train_acc_mean=("baseline_train_acc", "mean"),
            compare_train_acc_mean=("compare_train_acc", "mean"),
            baseline_stability_mean=("baseline_stability_disagree_rate_eps1e-3", "mean"),
            compare_stability_mean=("compare_stability_disagree_rate_eps1e-3", "mean"),
            baseline_train_seconds_mean=("baseline_train_seconds", "mean"),
            compare_train_seconds_mean=("compare_train_seconds", "mean"),
        )
        .reset_index()
        .sort_values("sweep_type")
    )

    detail_columns = [
        "sweep_type",
        "baseline_seed",
        "compare_seed",
        "baseline_batch",
        "compare_batch",
        "baseline_precision",
        "compare_precision",
        "disagreement_rate",
        "baseline_train_acc",
        "compare_train_acc",
        "baseline_stability_disagree_rate_eps1e-3",
        "compare_stability_disagree_rate_eps1e-3",
    ]
    detail = detail[detail_columns].sort_values(
        ["sweep_type", "compare_seed", "compare_batch", "compare_precision"],
        na_position="last",
    )

    summary.to_csv(SUMMARY_CSV, index=False)
    detail.to_csv(DETAIL_CSV, index=False)

    format_table(summary).to_markdown(SUMMARY_MD, index=False)
    format_table(detail).to_markdown(DETAIL_MD, index=False)

    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {SUMMARY_MD}")
    print(f"Wrote {DETAIL_CSV}")
    print(f"Wrote {DETAIL_MD}")


if __name__ == "__main__":
    main()
