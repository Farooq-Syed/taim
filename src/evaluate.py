"""Phase 5b: Evaluation harness - regular split vs walk-forward.

Both regimes run the SAME causal online detector (score-then-update; no
look-ahead). They differ in how the evaluation is sliced:

  * regular split  - one detector instance over the whole series; metrics are
    computed only on the final `split_frac` of the timeline (the deployed
    window, after long warm-up).
  * walk-forward   - a FRESH detector is warmed up on the preceding
    `train_days` and evaluated on the next single day, then the window rolls
    forward. Every test day is scored against a model that has only seen the
    past - which is what a live deployment actually experiences, including
    the early, less-warmed days.

Expectation being tested: the regular split looks better than the honest
walk-forward picture because (a) it only evaluates late well-warmed data and
(b) it tests fewer attack episodes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.detector import DetectorConfig
from src.fast_detector import FastTaimDetector


def run_detector(df: pd.DataFrame, cfg: DetectorConfig | None = None) -> pd.DataFrame:
    return FastTaimDetector(cfg or DetectorConfig()).run(df)


def _row_metrics(out: pd.DataFrame) -> dict[str, float]:
    y = out["is_attack"].astype(bool)
    pred = out["flagged"].astype(bool)
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())

    def safe(a: float, b: float) -> float:
        return a / b if b > 0 else float("nan")

    tpr = safe(tp, tp + fn)
    fpr = safe(fp, fp + tn)
    prec = safe(tp, tp + fp)
    f1 = safe(2 * prec * tpr, prec + tpr)
    return {
        "n": len(out),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "tpr": tpr,
        "fpr": fpr,
        "precision": prec,
        "f1": f1,
        "attack_rate": y.mean(),
        "flag_rate": pred.mean(),
    }


def _window_metrics(
    out: pd.DataFrame, windows: list, start: pd.Timestamp, step: int, min_fraction: float = 0.0
) -> dict[str, object]:
    """Per-attack-window detection stats. min_fraction: rows must overlap the
    window enough for it to count as "tested"."""
    rows = []
    for w in windows:
        lo = start + pd.Timedelta(days=w.start_day)
        hi = start + pd.Timedelta(days=w.end_day)
        win = out[out["timestamp"].between(lo, hi, inclusive="left")]
        if len(win) == 0:
            continue
        detected = bool(win["flagged"].any())
        max_stage = int(win["stage"].max()) if detected else 0
        ttd_steps = None
        stage4_steps = None
        if detected:
            first = win.index[win["flagged"]].min()
            ttd_steps = int((out.loc[first, "timestamp"] - lo) / pd.Timedelta(minutes=step))
            s4 = win.index[win["stage"] == 4]
            if len(s4):
                stage4_steps = int((out.loc[s4.min(), "timestamp"] - lo) / pd.Timedelta(minutes=step))
        rows.append(
            {
                "kind": w.kind,
                "start": lo,
                "end": hi,
                "rows_in_window": len(win),
                "detected": detected,
                "max_stage": max_stage,
                "ttd_steps": ttd_steps,
                "ttd_minutes": None if ttd_steps is None else ttd_steps * step,
                "to_stage4_steps": stage4_steps,
                "to_stage4_minutes": None if stage4_steps is None else stage4_steps * step,
            }
        )
    return {"windows": rows, "detected_count": sum(1 for r in rows if r["detected"]),
            "tested_count": len(rows)}


def regular_eval(
    df: pd.DataFrame, windows: list, split_frac: float = 0.5, step: int = 15,
    cfg: DetectorConfig | None = None,
) -> dict:
    out = run_detector(df, cfg)
    days = out["timestamp"].dt.dayofyear - out["timestamp"].dt.dayofyear.min()
    n_days = int(days.max()) + 1
    cut = split_frac * n_days
    test = out[days >= cut]
    return {
        "regime": "regular",
        "split_frac": split_frac,
        "test_days": [int(cut), n_days],
        "row": _row_metrics(test),
        "windows": _window_metrics(test, windows, df["timestamp"].min(), step),
    }


def walk_forward_eval(
    df: pd.DataFrame,
    windows: list,
    train_days: int = 14,
    step: int = 15,
    cfg: DetectorConfig | None = None,
) -> dict:
    out = df.copy()
    start = df["timestamp"].min()
    days = (df["timestamp"] - start).dt.days
    n_days = int(days.max()) + 1

    folds = []
    fold_details = []
    for test_day in range(train_days, n_days):
        region = df[days.between(test_day - train_days, test_day)]
        det_out = run_detector(region, cfg)
        # keep only the test day's rows (region is scored, only test day counts)
        day_of_row = (det_out["timestamp"] - start).dt.days
        fold_test = det_out[day_of_row == test_day]
        folds.append(fold_test)
        m = _row_metrics(fold_test)
        fold_details.append(
            {"test_day": int(test_day), "tpr": m["tpr"], "fpr": m["fpr"]}
        )

    if not folds:
        raise ValueError("no test days; reduce train_days")
    wf = pd.concat(folds, ignore_index=True)

    return {
        "regime": "walk_forward",
        "train_days": train_days,
        "test_days": [train_days, n_days],
        "row": _row_metrics(wf),
        "per_fold_tpr": [ _row_metrics(f)["tpr"] for f in folds ],
        "per_fold_fpr": [ _row_metrics(f)["fpr"] for f in folds ],
        "fold_details": fold_details,
        "n_folds": len(folds),
        "windows": _window_metrics(wf, windows, start, step),
    }


def summarize(result: dict) -> None:
    print(f"\n=== {result['regime'].upper()}  (test days {result['test_days']}) ===")
    r = result["row"]
    print(
        f"rows={r['n']}  TPR={r['tpr']:.3f}  FPR={r['fpr']:.4f}  "
        f"precision={r['precision']:.3f}  F1={r['f1']:.3f}"
    )
    wm = result["windows"]
    print(f"attack windows tested={wm['tested_count']} detected={wm['detected_count']}")
    for w in wm["windows"]:
        ttd = w["ttd_minutes"]
        s4 = w["to_stage4_minutes"]
        print(
            f"  {w['kind']:10s} detected={w['detected']} max_stage={w['max_stage']} "
            f"TTD={ttd if ttd is not None else '-'}min "
            f"stage4={s4 if s4 is not None else '-'}min"
        )
    if result["regime"] == "walk_forward":
        tprs = np.array(result["per_fold_tpr"])
        fprs = np.array(result["per_fold_fpr"])
        print(
            f"fold TPR: mean={np.nanmean(tprs):.3f} min={np.nanmin(tprs):.3f} max={np.nanmax(tprs):.3f} "
            f"(folds={result['n_folds']})"
        )
        print(
            f"fold FPR: mean={np.nanmean(fprs):.4f} min={np.nanmin(fprs):.4f} max={np.nanmax(fprs):.4f}"
        )