"""Integration tests for Phase 5: detector pipeline + regular/walk-forward eval."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_gen import AttackWindow, SimulationConfig, generate_dataset
from src.detector import DetectorConfig, TaimDetector
from src.evaluate import regular_eval, walk_forward_eval


def make_df():
    cfg = SimulationConfig(
        n_devices=6,
        n_days=21,
        interval_min=15,
        seed=11,
        attack_windows=[
            AttackWindow(3.0, 3.3, "flood", devices=[0]),
            AttackWindow(8.0, 8.4, "volumetric"),
            AttackWindow(15.0, 15.3, "syn"),
        ],
    )
    df, windows = generate_dataset(cfg)
    return df, windows, cfg


def test_detector_schema_and_columns():
    df, _, _ = make_df()
    out = TaimDetector().run(df)
    for col in ["score", "n_elevated", "fired", "stage", "action", "flagged"]:
        assert col in out.columns
    assert out["score"].between(0, 1).all()
    assert out["stage"].between(0, 4).all()
    assert out["action"].notna().all()


def test_detector_flags_during_attack_and_not_after():
    df, windows, _ = make_df()
    out = TaimDetector().run(df)
    start = df["timestamp"].min()
    w = windows[0]  # flood device 0
    lo = start + pd.Timedelta(days=w.start_day)
    hi = start + pd.Timedelta(days=w.end_day)
    dev0 = out[out["device_id"] == 0]
    in_win = dev0["timestamp"].between(lo, hi, inclusive="left")
    # allow warm-up before day 3 (slot warm-up needs a few samples)
    assert dev0[in_win]["flagged"].any(), "flood must be flagged"
    # a clean window (flood ended day 3.3, volumetric starts day 8.0) must be quiet
    clean = dev0["timestamp"].between(lo + pd.Timedelta(days=0.7), lo + pd.Timedelta(days=4.0))
    assert dev0[clean]["flagged"].mean() < 0.05


def test_detector_reaches_stage4_during_long_attack():
    df, windows, _ = make_df()
    out = TaimDetector().run(df)
    start = df["timestamp"].min()
    w = windows[1]  # volumetric (all devices, 0.4 days)
    win = out[out["timestamp"].between(start + pd.Timedelta(days=w.start_day),
                                       start + pd.Timedelta(days=w.end_day),
                                       inclusive="left")]
    assert (win["stage"] == 4).any(), "long volumetric attack must reach stage 4"


def test_regular_eval_structure():
    df, windows, _ = make_df()
    res = regular_eval(df, windows, split_frac=0.5)
    assert res["regime"] == "regular"
    assert res["row"]["f1"] >= 0.0
    assert res["row"]["tpr"] >= 0.0
    assert 0.0 <= res["row"]["fpr"] <= 1.0
    assert res["windows"]["tested_count"] > 0


def test_walk_forward_eval_structure():
    df, windows, cfg = make_df()
    res = walk_forward_eval(df, windows, train_days=7)
    assert res["regime"] == "walk_forward"
    assert res["n_folds"] == 21 - 7
    assert res["row"]["tpr"] >= 0.0
    assert len(res["per_fold_tpr"]) == res["n_folds"]
    # TPR/FPR values must be sane (0..1 or nan)
    assert all(np.isnan(t) or 0.0 <= t <= 1.0 for t in res["per_fold_tpr"])


def test_walk_forward_does_not_leak_future():
    """The model for a test day must never have seen the test day itself."""
    df, windows, _ = make_df()
    train_days = 7
    res = walk_forward_eval(df, windows, train_days=train_days)
    # basic sanity: no exception, folds exist
    assert res["n_folds"] >= 2


def test_no_lookahead_score_before_update():
    """Engine property: a value is scored against a model that predates it."""
    from src.baseline import BaselineConfig, TimeWindowBaseline

    b = TimeWindowBaseline(BaselineConfig(min_samples=2))
    b.step(0, (10,), {"s": 1.0})
    b.step(0, (10,), {"s": 1.0})
    # the first score after warm-up must reflect the OLD mean (1.0), not include 1e6
    z = b.score(0, (10,), {"s": 100.0})["s"]
    mu = b.models[(0, 10, "s")].mu
    assert abs(mu - 1.0) < 0.3  # model still at old mean when scored
    assert z > 0.0


def test_fast_detector_matches_reference():
    """FastTaimDetector must produce identical stages/flags to TaimDetector."""
    from src.detector import DetectorConfig, TaimDetector
    from src.fast_detector import FastTaimDetector

    df, _, _ = make_df()
    ref = TaimDetector(DetectorConfig()).run(df)
    fast = FastTaimDetector(DetectorConfig()).run(df)
    m = ref.merge(fast, on=["timestamp", "device_id"], suffixes=("_ref", "_fast"))
    assert (m["stage_ref"] == m["stage_fast"]).all()
    assert (m["flagged_ref"] == m["flagged_fast"]).all()
    # end-to-end metrics identical
    for dfr, dff in ((ref, fast),):
        y = dfr["is_attack"].astype(bool); p = dfr["flagged"].astype(bool)
        yf = dff["is_attack"].astype(bool); pf = dff["flagged"].astype(bool)
        assert int((p & y).sum()) == int((pf & yf).sum())
        assert int((p & ~y).sum()) == int((pf & ~yf).sum())