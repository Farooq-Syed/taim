"""Phase 6: Final validation on an UNSEEN environment.

Simulates deploying the system (with its Phase 5 default config, unchanged)
onto a different network: new seed, more devices, longer history, a different
sampling interval, different noise and a fresh attack schedule. If the
detector still detects all attack types with low false positives, the design
generalizes rather than overfitting the Phase 5 data generator.

Runs the same regular-split and walk-forward evaluations as Phase 5 and
writes a comparison report.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\Farooq Syed\taim")
from src.data_gen import AttackWindow, SimulationConfig, generate_dataset
from src.evaluate import regular_eval, summarize, walk_forward_eval

RESULT_DIR = r"C:\Users\Farooq Syed\taim\results"


def unseen_config() -> SimulationConfig:
    return SimulationConfig(
        n_devices=15,
        n_days=56,
        interval_min=10,  # different granularity (144 steps/day)
        seed=1234,
        attack_windows=[
            AttackWindow(5.0, 5.5, "flood", devices=[3, 9]),   # daytime flood
            AttackWindow(12.2, 12.6, "syn"),                   # all devices
            AttackWindow(20.8, 21.4, "volumetric"),            # all devices
            AttackWindow(33.5, 34.0, "flood", devices=[0]),    # single device
            AttackWindow(44.1, 44.9, "lowslow"),               # all devices
            AttackWindow(51.0, 51.3, "flood", devices=[7]),
        ],
    )


def main() -> None:
    cfg = unseen_config()
    df, windows = generate_dataset(cfg)
    step = 10
    print(f"Unseen environment: {df.device_id.nunique()} devices, "
          f"{cfg.n_days} days, {step}-min intervals, seed={cfg.seed}")
    print(f"attack windows: {[w.kind for w in windows]}\n")

    regular = regular_eval(df, windows, split_frac=0.5, step=step)
    walk = walk_forward_eval(df, windows, train_days=14, step=step)

    summarize(regular)
    summarize(walk)

    # ---- comparison table: recompute Phase 5 fresh (same code, no tuning) ----
    df5, win5 = generate_dataset(
        SimulationConfig(n_devices=10, n_days=42, interval_min=15, seed=42,
                         attack_windows=[
                             AttackWindow(7.1, 7.4, "volumetric"),
                             AttackWindow(14.3, 14.7, "syn"),
                             AttackWindow(21.1, 21.4, "flood", devices=[2]),
                             AttackWindow(28.5, 29.2, "lowslow"),
                             AttackWindow(35.0, 35.3, "flood", devices=[5, 7]),
                         ])
    )
    r5 = regular_eval(df5, win5, split_frac=0.5, step=15)
    w5 = walk_forward_eval(df5, win5, train_days=14, step=15)
    rows = [
        {"label": "phase5_regular", "rows": r5["row"]["n"], "tpr": round(r5["row"]["tpr"], 4),
         "fpr": round(r5["row"]["fpr"], 4), "precision": round(r5["row"]["precision"], 4),
         "f1": round(r5["row"]["f1"], 4)},
        {"label": "phase5_walkforward", "rows": w5["row"]["n"], "tpr": round(w5["row"]["tpr"], 4),
         "fpr": round(w5["row"]["fpr"], 4), "precision": round(w5["row"]["precision"], 4),
         "f1": round(w5["row"]["f1"], 4)},
    ]
    for label, res in (("unseen_regular", regular), ("unseen_walkforward", walk)):
        r = res["row"]
        rows.append(
            {
                "label": label,
                "rows": r["n"],
                "tpr": round(r["tpr"], 4),
                "fpr": round(r["fpr"], 4),
                "precision": round(r["precision"], 4),
                "f1": round(r["f1"], 4),
            }
        )
    comp = pd.DataFrame(rows)
    comp.to_csv(rf"{RESULT_DIR}\phase6_unseen_comparison.csv", index=False)
    print("\nPhase 6 comparison saved to results/phase6_unseen_comparison.csv")
    print(comp.to_string(index=False))

    # window-level detail for the unseen run
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
    wtab.to_csv(rf"{RESULT_DIR}\phase6_unseen_windows.csv", index=False)
    print("\nUnseen window metrics:")
    print(wtab.to_string(index=False))


if __name__ == "__main__":
    main()