"""Robustness / anti-overfitting sweep.

Runs the detector (default config, no per-environment tuning) over N random
simulated environments: random seed, device count, sampling interval, days,
noise and a random attack schedule spanning all four attack types. If
performance (window detection rate, TPR/FPR/F1) stays high and stable across
all of them, the detector generalises rather than overfitting the single
Phase-5 dataset the config was developed on.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\Farooq Syed\taim")
from src.data_gen import AttackWindow, SimulationConfig, generate_dataset
from src.detector import DetectorConfig
from src.fast_detector import FastTaimDetector

RESULT_DIR = r"C:\Users\Farooq Syed\taim\results"

ATTACK_TYPES = ("volumetric", "flood", "syn", "lowslow")


def random_env(rng: np.random.Generator, n_dev_bounds=(8, 25)) -> SimulationConfig:
    n_dev = int(rng.integers(*n_dev_bounds))
    n_days = int(rng.integers(28, 70))
    interval = int(rng.choice([5, 10, 15]))
    noise = float(rng.uniform(0.2, 0.5))          # quieter vs noisier fleets
    peak = float(rng.uniform(9.0, 20.0))          # different activity time zones
    wf = float(rng.uniform(0.4, 0.9))             # different weekend behaviour
    n_attacks = int(rng.integers(4, 7))
    windows = []
    for _ in range(n_attacks):
        kind = ATTACK_TYPES[int(rng.integers(len(ATTACK_TYPES)))]
        start = float(rng.uniform(2.0, n_days - 2.0))
        dur = float(rng.uniform(0.2, 1.0))
        devices = None
        if rng.random() < 0.55:  # targeted subset sometimes
            devices = [int(rng.integers(n_dev)) for _ in range(int(rng.integers(1, 3)))]
        windows.append(AttackWindow(start, min(start + dur, n_days - 0.1), kind, devices=devices))
    return SimulationConfig(
        n_devices=n_dev, n_days=n_days, interval_min=interval,
        noise_sigma=noise, diurnal_peak=peak, weekend_factor=wf,
        seed=int(rng.integers(100000)), attack_windows=windows,
    )


def eval_env(df: pd.DataFrame, windows: list) -> dict:
    out = FastTaimDetector(DetectorConfig()).run(df)
    y = out["is_attack"].astype(bool)
    p = out["flagged"].astype(bool)
    tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum())
    tpr = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + int((~p & ~y).sum())) if (fp + int((~p & ~y).sum())) else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * tpr / (prec + tpr) if prec + tpr else 0.0

    start = df["timestamp"].min()
    detected = {k: 0 for k in ATTACK_TYPES}
    total = {k: 0 for k in ATTACK_TYPES}
    max_stages = {k: 0 for k in ATTACK_TYPES}
    for w in windows:
        lo = start + pd.Timedelta(days=w.start_day)
        hi = start + pd.Timedelta(days=w.end_day)
        win = out[out["timestamp"].between(lo, hi, inclusive="left")]
        if len(win) == 0:
            continue
        total[w.kind] += 1
        det = bool(win["flagged"].any())
        detected[w.kind] += int(det)
        max_stages[w.kind] = max(max_stages[w.kind], int(win["stage"].max()))
    return {
        "tpr": tpr, "fpr": fpr, "f1": f1, "precision": prec,
        "n_windows": len(windows),
        "detected": detected, "total": total,
        "max_stages": max_stages,
    }


def main(n_envs: int = 60, seed: int = 2026) -> None:
    rng = np.random.default_rng(seed)
    results = []
    t0 = time.time()
    for i in range(n_envs):
        cfg = random_env(rng)
        df, windows = generate_dataset(cfg)
        r = eval_env(df, windows)
        r["env"] = i
        r["n_dev"] = cfg.n_devices
        r["n_days"] = cfg.n_days
        r["interval"] = cfg.interval_min
        r["noise"] = cfg.noise_sigma
        r["diurnal_peak"] = cfg.diurnal_peak
        r["weekend"] = cfg.weekend_factor
        r["seed"] = cfg.seed
        results.append(r)
        print(f"[{i + 1}/{n_envs}] seed={cfg.seed} dev={cfg.n_devices} "
              f"days={cfg.n_days} int={cfg.interval_min}m f1={r['f1']:.3f} "
              f"fpr={r['fpr']:.4f} det={r['detected']}/{r['total']}",
              flush=True)

    res = pd.DataFrame(results)
    res.to_csv(rf"{RESULT_DIR}\robustness_sweep.csv", index=False)

    print("\n================= ROBUSTNESS SWEEP SUMMARY =================")
    for m in ["tpr", "fpr", "f1", "precision"]:
        col = res[m].replace([np.inf], np.nan).dropna()
        print(f"{m:10s} mean={col.mean():.3f} median={col.median():.3f} "
              f"min={col.min():.3f} max={col.max():.3f} std={col.std():.3f}")

    print("\nWindow detection rate by attack type (across all envs):")
    for k in ATTACK_TYPES:
        tot = int(res["total"].apply(lambda d: d[k]).sum())
        det = int(res["detected"].apply(lambda d: d[k]).sum())
        ms = max(int(v) for v in res["max_stages"].apply(lambda d: d[k]))
        rate = det / tot if tot else float("nan")
        print(f"  {k:10s} {det}/{tot} ({rate:.0%})  max_stage={ms}")

    print(f"\nelapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()