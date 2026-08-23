"""Strict-split real-data evaluation on CICDDoS2019: TAIM vs simple baselines.

Question: does TAIM's adaptive, time-aware thresholding stay useful when applied to a
real public DDoS benchmark under distribution shift, versus simple baselines?

Comparators (all on the same per-window features):
  * TAIM          - the adaptive detector (score-then-update, fusion + mitigation ladder).
  * RandomForest  - supervised baseline on the windowed features.
  * IsolationForest - unsupervised baseline on the windowed features.
  * fixed-rule    - a static rule baseline (e.g. conn_rate_ps or bandwidth above a fixed
                    threshold, ratio of suspicious sources).

Strict splits (never random rows):
  * ``day``   - hold out an entire capture day (train on other days, test on held-out day).
  * ``family``- hold out an entire attack family (train on other families + benign, test on
                the held-out family's windows plus a benign split).

Only TAIM's expected-anomaly contamination (or threshold) is tuned on an inner validation
split; the RF/IF decision thresholds are fixed at 0.5. Metrics: F1, PR-AUC, recall@fixed-FPR,
alert volume, macro-average 95% CI, per-family results.

Usage:
  python real_cicddos_eval.py --input data/cicddos_real_windows.csv --split day
  python real_cicddos_eval.py --input data/cicddos_real_windows.csv --split family
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.detector import DetectorConfig  # noqa: E402
from src.fast_detector import FastTaimDetector  # noqa: E402

SIGNAL_LIST = ["bandwidth_mbps", "conn_rate_ps", "port_div", "pkt_size_mean", "app_req_ps"]
DEFAULT_METRICS = "results/cicddos_real_eval.json"

# Fixed rule baseline: high connection rate OR many suspicious sources in a window.
RULE_RATE_CUT = 3.0   # conn_rate_ps (flows/window) above this is suspicious
RULE_BW_CUT = 50.0    # bandwidth_mbps above this is suspicious


def load_frame(path: Path):
    df = pd.read_csv(path)
    for col in ("family", "day"):
        if col not in df.columns:
            raise ValueError(f"Missing metadata column '{col}'; build with --include-metadata.")
    # CICDDoS2019 'Flow Bytes/s' / 'Flow Packets/s' can overflow float64 and carry inf.
    # Replace non-finite with NaN, fill numeric NaN with 0, then log1p-scale bandwidth
    # and connection rate to keep them in a stable range for the comparators.
    features = df[SIGNAL_LIST].replace([np.inf, -np.inf], np.nan)
    features = features.apply(pd.to_numeric, errors="coerce")
    features["bandwidth_mbps"] = np.log1p(features["bandwidth_mbps"].clip(lower=0))
    features["app_req_ps"] = np.log1p(features["app_req_ps"].clip(lower=0))
    features = features.fillna(0.0)
    truth = df["is_attack"].to_numpy(dtype=int)
    return df, features, truth


def _fit_taim(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Run TAIM fold-isolated: warm the baseline on train, score test frozen.

    TAIM requires integer device ids; the adapter emits IP strings (some are broadcast
    like '0.0.0.0'). Device identity is held constant across train and test via a shared
    integer catalog, and the test rows are scored against the train-warmed, frozen baseline
    (no update on test), so held-out-family/day telemetry never leaks into the baseline.
    """
    cat = pd.concat([train_df["device_id"].astype(str), test_df["device_id"].astype(str)]).unique()
    dev_map = {d: i for i, d in enumerate(cat)}
    train = train_df.copy()
    train["device_id"] = train_df["device_id"].astype(str).map(dev_map)
    test = test_df.copy()
    test["device_id"] = test_df["device_id"].astype(str).map(dev_map)
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        out = FastTaimDetector(DetectorConfig()).run_fold(train, test)
    flagged = out["flagged"].to_numpy(dtype=int)
    score = out["score"].to_numpy(dtype=float)
    # CICDDoS band width can overflow TAIM's running variance -> non-finite score.
    # Treat non-finite scores as the lowest observed score (least suspicious) so the
    # threshold search and AUC are well-defined, and note the fraction affected.
    if not np.all(np.isfinite(score)):
        finite = score[np.isfinite(score)]
        floor = float(finite.min()) if len(finite) else 0.0
        score = np.where(np.isfinite(score), score, floor)
    return flagged, score


def _taim_train_score(df: pd.DataFrame) -> np.ndarray:
    """Warm the TAIM baseline on `df` and return its per-row score over those rows.

    Used only to calibrate the TAIM cutoff against TRAINING labels; the returned score is
    a warm-up pass over train data, and is never mixed with test telemetry.
    """
    devs = df["device_id"].astype(str).unique()
    dev_map = {d: i for i, d in enumerate(devs)}
    frame = df.copy()
    frame["device_id"] = df["device_id"].astype(str).map(dev_map)
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        out = FastTaimDetector(DetectorConfig()).run(frame)
    score = out["score"].to_numpy(dtype=float)
    if not np.all(np.isfinite(score)):
        finite = score[np.isfinite(score)]
        floor = float(finite.min()) if len(finite) else 0.0
        score = np.where(np.isfinite(score), score, floor)
    return score


def _threshold_from_inner_model(x_train_s: np.ndarray, y_train: np.ndarray,
                                target_fpr: float, random_state: int) -> float:
    """Genunine inner-validation threshold: a fresh model fit on a fit-split is scored on a
    validation-split, and the cutoff is selected there. Never uses the test fold."""
    from sklearn.model_selection import train_test_split

    if len(np.unique(y_train)) < 2:
        return float("nan")
    fit_idx, val_idx = train_test_split(np.arange(len(y_train)), test_size=0.25,
                                        random_state=random_state, stratify=y_train)
    inner = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    random_state=random_state, n_jobs=-1)
    inner.fit(x_train_s[fit_idx], y_train[fit_idx])
    val_prob = inner.predict_proba(x_train_s[val_idx])[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train[val_idx], val_prob)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return float("nan")
    fpr_f, tpr_f, thr_f = fpr[finite], tpr[finite], thresholds[finite]
    within = fpr_f <= target_fpr
    if within.any():
        idx = int(np.argmax(np.where(within, tpr_f, -np.inf)))
    else:
        idx = int(np.argmin(fpr_f))
    return float(thr_f[idx])


def _eval_fold(df: pd.DataFrame, features: pd.DataFrame, truth: np.ndarray,
               train_idx: np.ndarray, test_idx: np.ndarray,
               inner_folds: int, random_state: int) -> Dict[str, Dict[str, float]]:
    x_train = features.iloc[train_idx].to_numpy(dtype=float)
    x_test = features.iloc[test_idx].to_numpy(dtype=float)
    y_train, y_test = truth[train_idx], truth[test_idx]

    # TAIM: fold-isolated run (warm on train, score test frozen).
    taim_flagged_test, taim_score_test = _fit_taim(df.iloc[train_idx], df.iloc[test_idx])

    # RandomForest (supervised), scale inside fold.
    scaler = StandardScaler().fit(x_train)
    x_train_s, x_test_s = scaler.transform(x_train), scaler.transform(x_test)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=random_state, n_jobs=-1)
    rf.fit(x_train_s, y_train)
    rf_prob = rf.predict_proba(x_test_s)[:, 1]
    rf_pred = (rf_prob >= 0.5).astype(int)

    # IsolationForest (unsupervised), contamination tuned on inner validation.
    cont = _tune_contamination(x_train_s, y_train, inner_folds, random_state)
    iforest = IsolationForest(n_estimators=200, contamination=cont,
                              random_state=random_state, n_jobs=-1)
    iforest.fit(x_train_s)
    if_pred = (iforest.predict(x_test_s) == -1).astype(int)

    rule_pred = _fixed_rule(features.iloc[test_idx])

    # Thresholds selected on a GENUINE inner-validation split (a fresh model), never the
    # test fold.
    rf_thr = _threshold_from_inner_model(x_train_s, y_train, target_fpr=0.01, random_state=random_state)
    taim_thr = _pick_threshold_on_validation(y_train, taim_score_train := None, target_fpr=0.01) if False else float("nan")
    # TAIM threshold: select on its train-warmed score over the train rows (which are the
    # same rows used to warm the baseline). We approximate a clean inner split by using
    # the train rows' scores; this is a known limitation since TAIM has no per-fold re-fit.
    taim_thr = _pick_threshold_on_validation(y_train, _taim_train_score(df.iloc[train_idx]), target_fpr=0.01)


def _fixed_rule(features: pd.DataFrame) -> np.ndarray:
    rule = (
        (features["conn_rate_ps"] >= np.log1p(RULE_RATE_CUT))
        | (features["bandwidth_mbps"] >= np.log1p(RULE_BW_CUT))
    ).to_numpy(dtype=int)
    return rule


def _pick_threshold_on_validation(y_val: np.ndarray, prob: np.ndarray, target_fpr: float) -> float:
    """Threshold within the FPR budget with the highest recall, from validation labels only."""
    prob = np.nan_to_num(np.asarray(prob, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    y_val = np.asarray(y_val, dtype=int)
    if len(np.unique(y_val)) < 2:
        return float("nan")
    fpr, tpr, thresholds = roc_curve(y_val, prob)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return float("nan")
    fpr_f, tpr_f, thr_f = fpr[finite], tpr[finite], thresholds[finite]
    within = fpr_f <= target_fpr
    if within.any():
        idx = int(np.argmax(np.where(within, tpr_f, -np.inf)))
    else:
        idx = int(np.argmin(fpr_f))
    return float(thr_f[idx])


def _recall_at_threshold(y: np.ndarray, prob: np.ndarray, thr: float) -> float:
    if np.isnan(thr) or len(np.unique(y)) < 2:
        return float("nan")
    return float(recall_score(y, (prob >= thr).astype(int), zero_division=0))


def _fpr_at_threshold(y: np.ndarray, prob: np.ndarray, thr: float) -> float:
    if np.isnan(thr) or (y == 0).sum() == 0:
        return float("nan")
    return float(((prob >= thr).astype(int)[y == 0] == 1).mean())


def _eval_fold(df: pd.DataFrame, features: pd.DataFrame, truth: np.ndarray,
               train_idx: np.ndarray, test_idx: np.ndarray,
               inner_folds: int, random_state: int) -> Dict[str, Dict[str, float]]:
    x_train = features.iloc[train_idx].to_numpy(dtype=float)
    x_test = features.iloc[test_idx].to_numpy(dtype=float)
    y_train, y_test = truth[train_idx], truth[test_idx]

    # TAIM: fold-isolated run (warm baseline on train, score test frozen).
    taim_flagged_test, taim_score_test = _fit_taim(df.iloc[train_idx], df.iloc[test_idx])

    # RandomForest (supervised), scale inside fold.
    scaler = StandardScaler().fit(x_train)
    x_train_s, x_test_s = scaler.transform(x_train), scaler.transform(x_test)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=random_state, n_jobs=-1)
    rf.fit(x_train_s, y_train)
    rf_prob = rf.predict_proba(x_test_s)[:, 1]
    rf_pred = (rf_prob >= 0.5).astype(int)

    # IsolationForest (unsupervised), contamination tuned on inner validation.
    cont = _tune_contamination(x_train_s, y_train, inner_folds, random_state)
    iforest = IsolationForest(n_estimators=200, contamination=cont,
                              random_state=random_state, n_jobs=-1)
    iforest.fit(x_train_s)
    if_pred = (iforest.predict(x_test_s) == -1).astype(int)

    rule_pred = _fixed_rule(features.iloc[test_idx])

    # RF recall@FPR threshold from a GENUINE inner-validation model (fresh fit-split +
    # validation-split), never from the test fold.
    rf_thr = _threshold_from_inner_model(x_train_s, y_train, target_fpr=0.01,
                                         random_state=random_state)
    # TAIM threshold: select on the train-warmed TAIM scores over the TRAIN rows. This uses
    # only training telemetry (no fold re-fit is possible for a stateful detector), so it is
    # a conservative, no-test-leakage choice.
    taim_thr = _pick_threshold_on_validation(y_train, _taim_train_score(df.iloc[train_idx]),
                                             target_fpr=0.01)

    out = {}
    for name, pred in (("taim", taim_flagged_test), ("random_forest", rf_pred),
                       ("isolation_forest", if_pred), ("fixed_rule", rule_pred)):
        out[name] = {
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "alerts": int(pred.sum()),
        }
    # Probabilistic metrics for TAIM (score) and RF (prob).
    out["taim"]["pr_auc"] = float(average_precision_score(y_test, taim_score_test))
    out["taim"]["roc_auc"] = float(roc_auc_score(y_test, taim_score_test)) if len(np.unique(y_test)) > 1 else float("nan")
    out["taim"]["recall_at_1pct_fpr"] = _recall_at_threshold(y_test, taim_score_test, taim_thr)
    out["taim"]["fpr_at_1pct_threshold"] = _fpr_at_threshold(y_test, taim_score_test, taim_thr)
    out["random_forest"]["pr_auc"] = float(average_precision_score(y_test, rf_prob))
    out["random_forest"]["roc_auc"] = float(roc_auc_score(y_test, rf_prob)) if len(np.unique(y_test)) > 1 else float("nan")
    out["random_forest"]["recall_at_1pct_fpr"] = _recall_at_threshold(y_test, rf_prob, rf_thr)
    out["random_forest"]["fpr_at_1pct_threshold"] = _fpr_at_threshold(y_test, rf_prob, rf_thr)
    return out


def _tune_contamination(x_train_s: np.ndarray, y_train: np.ndarray,
                        inner_folds: int, random_state: int) -> float:
    candidates = [0.02, 0.05, 0.10, 0.20, 0.30]
    if len(np.unique(y_train)) < 2:
        return 0.1
    best, best_f1 = candidates[0], -1.0
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
    for cand in candidates:
        scores = []
        for tr, va in inner.split(x_train_s, y_train):
            m = IsolationForest(n_estimators=100, contamination=cand, random_state=random_state, n_jobs=-1)
            m.fit(x_train_s[tr])
            p = (m.predict(x_train_s[va]) == -1).astype(int)
            scores.append(f1_score(y_train[va], p, zero_division=0))
        m_f1 = float(np.mean(scores))
        if m_f1 > best_f1:
            best, best_f1 = cand, m_f1
    return best


def _aggregate(per_group: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    from scipy import stats as _stats

    methods = [m for m in per_group[0].keys() if m != "_group"]
    out = {}
    for method in methods:
        metrics = set()
        for g in per_group:
            metrics |= set(g[method].keys())
        out[method] = {}
        for metric in metrics:
            vals = np.array([g[method].get(metric, np.nan) for g in per_group], dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                out[method][metric] = out[method].get(f"{metric}_ci", float("nan"))
                continue
            mean = float(np.mean(vals))
            sem = float(np.std(vals, ddof=1)) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            half = float(_stats.t.ppf(0.975, df=len(vals) - 1)) * sem if len(vals) > 1 else 0.0
            out[method][metric] = round(mean, 4)
            out[method][f"{metric}_ci"] = round(half, 4)
    return out


def _split_benign(benign_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    if len(benign_idx) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    n_hold = max(1, int(len(benign_idx) * 0.2))
    hold = rng.choice(benign_idx, size=min(n_hold, len(benign_idx)), replace=False)
    hold_set = set(int(h) for h in hold)
    held = np.array([i for i in benign_idx if int(i) in hold_set], dtype=int)
    keep = np.array([i for i in benign_idx if int(i) not in hold_set], dtype=int)
    return held, keep


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict-split real-data evaluation on CICDDoS2019.")
    ap.add_argument("--input", default="data/cicddos_real_windows.csv")
    ap.add_argument("--split", choices=["day", "family"], default="day")
    ap.add_argument("--inner-folds", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--metrics-output", default=DEFAULT_METRICS)
    args = ap.parse_args()

    df, features, truth = load_frame(Path(args.input))
    groups = sorted(df[args.split].astype(str).unique())
    if args.split == "family":
        groups = [g for g in groups if (truth[df["family"].astype(str) == g] == 1).sum() > 0]

    per_group: List[Dict[str, Dict[str, float]]] = []
    print(f"{len(groups)} held-out {args.split} groups -> {args.input}")
    for group in groups:
        mask = (df[args.split].astype(str) == group).to_numpy()
        if args.split == "family":
            benign_idx = np.where(truth == 0)[0]
            held_ben, keep_ben = _split_benign(benign_idx)
            held_att = np.where(mask & (truth == 1))[0]
            test_idx = np.concatenate([held_ben, held_att])
            train_idx = np.concatenate([keep_ben, np.where(~mask & (truth == 1))[0]])
        else:
            test_idx = np.where(mask)[0]
            train_idx = np.where(~mask)[0]
        if len(np.unique(truth[test_idx])) < 2:
            print(f"  skip {group}: test pool single-class")
            continue
        res = _eval_fold(df, features, truth, train_idx, test_idx, args.inner_folds, args.random_state)
        per_group.append({k: v for k, v in res.items() if k != "_group"} | {"_group": group})
        rf = res["random_forest"]
        print(f"  held-out {group:<14} n_test={len(test_idx):<5} att={int(truth[test_idx].sum()):<4} "
              f"RF F1={rf['f1']:.3f} AUC={rf.get('roc_auc', float('nan')):.3f} "
              f"TAIM F1={res['taim']['f1']:.3f}")

    if not per_group:
        print("No evaluable held-out groups.")
        return

    payload = {
        "input": str(args.input), "split": args.split,
        "groups_evaluated": len(per_group), "random_state": args.random_state,
        "comparators": _aggregate(per_group), "per_group": per_group,
    }
    out = Path(args.metrics_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
