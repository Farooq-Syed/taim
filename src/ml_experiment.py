"""A/B/C ML experiment.

A: current detector (control)
B: + windowed-mean temporal scorer (non-ML temporal baseline)
C: + PCA-autoencoder temporal scorer (the ML idea)

Runs all three on the same set of environments and compares window-level
detection (especially lowslow) plus F1/FPR. This is the decision test: keep
the ML idea only if C meaningfully beats A *and* beats B.
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
from src.temporal import PCAWindowScorer, WindowedMeanScorer

ATTACK_TYPES = ("volumetric", "flood", "syn", "lowslow")


def run_with_scorer(df, scorer_cls, train_days=14, **kw):
    cfg = DetectorConfig()
    interval = int(round((df['timestamp'].diff().dropna().dt.total_seconds().median()) / 60))
    # ~2-hour window
    W = max(4, int(120 / interval))
    scorer = scorer_cls(window_steps=W, train_days=train_days, step_min=interval, **kw)
    out = FastTaimDetector(cfg, temporal=scorer).run(df)
    return out


def eval_env(out, windows):
    y = out["is_attack"].astype(bool)
    p = out["flagged"].astype(bool)
    tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum())
    tpr = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + int((~p & ~y).sum())) if (fp + int((~p & ~y).sum())) else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * tpr / (prec + tpr) if prec + tpr else 0.0
    start = out["timestamp"].min()
    det = {k: 0 for k in ATTACK_TYPES}
    tot = {k: 0 for k in ATTACK_TYPES}
    for w in windows:
        lo = start + pd.Timedelta(days=w.start_day)
        hi = start + pd.Timedelta(days=w.end_day)
        win = out[out["timestamp"].between(lo, hi, inclusive="left")]
        if len(win) == 0:
            continue
        tot[w.kind] += 1
        det[w.kind] += int(bool(win["flagged"].any()))
    return {"tpr": tpr, "fpr": fpr, "f1": f1, "det": det, "tot": tot}


def make_envs(n, seed):
    from src.robustness import random_env
    rng = np.random.default_rng(seed)
    return [random_env(rng) for _ in range(n)]


def make_strict_envs(n, seed):
    """Much stricter / more realistic environments:
    - shorter warm-up opportunity (fewer days)
    - noisier fleets, more targeted + weaker attacks
    - legitimate flash crowds that must NOT be flagged
    """
    from src.robustness import random_env
    rng = np.random.default_rng(seed)
    envs = []
    for _ in range(n):
        cfg = random_env(rng)
        cfg.n_days = int(rng.integers(21, 40))           # less history
        cfg.noise_sigma = float(rng.uniform(0.3, 0.5))   # noisier
        # weaken + shorten attacks
        for w in cfg.attack_windows:
            w.duration = min(w.end_day - w.start_day, float(rng.uniform(0.15, 0.5)))
            w.end_day = w.start_day + w.duration
            w.intensity = float(rng.uniform(0.6, 1.0))
        # legit flash crowds (normal, must not trigger mitigation)
        n_fc = int(rng.integers(1, 4))
        fc = []
        for _ in range(n_fc):
            s = float(rng.uniform(2.0, cfg.n_days - 2.0))
            fc.append((s, min(s + 0.4, cfg.n_days - 0.1), float(rng.uniform(1.8, 3.0))))
        cfg.flash_crowds = fc
        envs.append(cfg)
    return envs


def main(n_envs: int = 30, seed: int = 777, strict: bool = False) -> None:
    envs = make_strict_envs(n_envs, seed) if strict else make_envs(n_envs, seed)
    systems = {
        "A_current": lambda df: FastTaimDetector(DetectorConfig()).run(df),
        "B_windowmean": lambda df: run_with_scorer(df, WindowedMeanScorer),
        "C_pca_ae": lambda df: run_with_scorer(df, PCAWindowScorer),
    }
    results = {k: [] for k in systems}
    t0 = time.time()
    for i, cfg in enumerate(envs):
        df, windows = generate_dataset(cfg)
        for name, fn in systems.items():
            out = fn(df)
            r = eval_env(out, windows)
            r["env"] = i
            results[name].append(r)
        if (i + 1) % 5 == 0:
            print(f"[{i+1}/{n_envs}] {time.time()-t0:.0f}s", flush=True)

    print("\n================ ML EXPERIMENT ================")
    print(f"mode: {'STRICT (flash crowds, weaker/noisier)' if strict else 'standard'}")
    for name, rs in results.items():
        d = pd.DataFrame(rs)
        print(f"\n{name}:")
        for m in ["tpr", "fpr", "f1"]:
            col = d[m].replace([np.inf], np.nan).dropna()
            print(f"  {m:5s} mean={col.mean():.3f} median={col.median():.3f} min={col.min():.3f}")
        for k in ATTACK_TYPES:
            tot = int(sum(r["tot"][k] for r in rs))
            det = int(sum(r["det"][k] for r in rs))
            print(f"  {k:10s} {det}/{tot} ({det/max(tot,1):.0%})")
    print(f"\nelapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--envs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()
    main(n_envs=args.envs, seed=args.seed, strict=args.strict)