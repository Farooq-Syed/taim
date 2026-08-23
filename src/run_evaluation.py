"""Run the Phase 5 evaluation: regular split vs walk-forward, on the default
42-day simulated dataset. Saves numeric results and plots to results/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data_gen import default_config, generate_dataset
from src.detector import DetectorConfig
from src.evaluate import regular_eval, summarize, walk_forward_eval
from src.plot_utils import (
    plot_aggregate_signal,
    plot_device_stages,
    plot_fold_metrics,
)
from src.fast_detector import FastTaimDetector

DATA_CSV = PROJECT_ROOT / "data" / "dataset_42d.csv"
RESULT_DIR = PROJECT_ROOT / "results"


def load_or_generate() -> tuple[pd.DataFrame, list]:
    if DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
        cfg = default_config()
        return df, list(cfg.attack_windows)
    df, windows = generate_dataset(default_config())
    DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_CSV, index=False)
    return df, windows


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df, windows = load_or_generate()
    step = 15

    regular = regular_eval(df, windows, split_frac=0.5, step=step)
    walk = walk_forward_eval(df, windows, train_days=14, step=step)

    summarize(regular)
    summarize(walk)

    # ---- numeric comparison table ----
    rows = []
    for res in (regular, walk):
        r = res["row"]
        rows.append(
            {
                "regime": res["regime"],
                "test_days": f"{res['test_days'][0]}-{res['test_days'][1]}",
                "rows": r["n"],
                "tpr": round(r["tpr"], 4),
                "fpr": round(r["fpr"], 4),
                "precision": round(r["precision"], 4),
                "f1": round(r["f1"], 4),
            }
        )
    comp = pd.DataFrame(rows)
    comp.to_csv(RESULT_DIR / "regular_vs_walkforward.csv", index=False)
    print("\nComparison table saved to results/regular_vs_walkforward.csv")
    print(comp.to_string(index=False))

    # ---- window-level table ----
    win_rows = []
    for res in (regular, walk):
        for w in res["windows"]["windows"]:
            win_rows.append(
                {
                    "regime": res["regime"],
                    "kind": w["kind"],
                    "detected": w["detected"],
                    "max_stage": w["max_stage"],
                    "ttd_min": w["ttd_minutes"],
                    "to_stage4_min": w["to_stage4_minutes"],
                }
            )
    wtab = pd.DataFrame(win_rows)
    wtab.to_csv(RESULT_DIR / "window_metrics.csv", index=False)
    print("\nWindow-level metrics saved to results/window_metrics.csv")
    print(wtab.to_string(index=False))

    # ---- plots ----
    plot_aggregate_signal(
        df, "bandwidth_mbps", windows,
        "Aggregate bandwidth with attack windows",
        RESULT_DIR / "eval_aggregate_bandwidth.png",
    )

    fold_df = pd.DataFrame(walk["fold_details"])
    fold_df.to_csv(RESULT_DIR / "fold_metrics.csv", index=False)
    plot_fold_metrics(walk["fold_details"], RESULT_DIR / "eval_fold_performance.png")

    # ladder behaviour for the flood@[2] device around day 21
    det_out = FastTaimDetector(DetectorConfig()).run(df)
    plot_device_stages(
        det_out, 2, windows, RESULT_DIR / "eval_ladder_device2.png",
        df["timestamp"].min(),
    )
    print("Plots saved to results/.")


if __name__ == "__main__":
    main()
