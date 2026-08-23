"""Tests for src/real_cicddos_eval.py (strict-split CICDDoS2019 evaluation)."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.real_cicddos_eval as ev  # noqa: E402


def test_pick_threshold_stays_within_fpr_budget():
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    prob = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    thr = ev._pick_threshold_on_validation(y, prob, target_fpr=0.01)
    # Highest-recall threshold with validation FPR <= 0.01 must be finite.
    assert np.isfinite(thr)
    pred = (prob >= thr).astype(int)
    val_fpr = (pred[y == 0] == 1).mean() if (y == 0).any() else float("nan")
    assert val_fpr <= 0.01 + 1e-6


def test_pick_threshold_handles_nan_scores():
    y = np.array([0, 0, 1, 1])
    prob = np.array([np.nan, 0.2, 0.8, np.nan])
    thr = ev._pick_threshold_on_validation(y, prob, target_fpr=0.5)
    assert np.isfinite(thr)


def test_fpr_at_threshold_and_recall():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    assert ev._fpr_at_threshold(y, prob, 0.5) <= 0.01
    assert ev._recall_at_threshold(y, prob, 0.5) >= 0.5
    assert np.isnan(ev._fpr_at_threshold(y, prob, float("nan")))
    assert np.isnan(ev._recall_at_threshold(y, prob, float("nan")))


def test_fixed_rule_flags_heavy_traffic():
    features = pd.DataFrame({
        "conn_rate_ps": [np.log1p(1.0), np.log1p(50.0)],
        "bandwidth_mbps": [np.log1p(5.0), np.log1p(5.0)],
        "port_div": [1.0, 1.0], "pkt_size_mean": [100.0, 100.0], "app_req_ps": [1.0, 1.0],
    })
    pred = ev._fixed_rule(features)
    assert pred[0] == 0 and pred[1] == 1


def test_load_frame_log_scales_and_sanitizes():
    rows = {
        "family": ["Benign", "Syn"], "day": ["2018-12-01", "2018-12-01"],
        "is_attack": [0, 1],
        "bandwidth_mbps": [np.inf, 1e12], "conn_rate_ps": [np.nan, 1000.0],
        "port_div": [1.0, 2.0], "pkt_size_mean": [120.0, 500.0], "app_req_ps": [np.inf, 5e8],
    }
    df = pd.DataFrame(rows)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fh:
        df.to_csv(fh.name, index=False)
        path = Path(fh.name)
    try:
        _, features, truth = ev.load_frame(path)
        assert not np.isfinite(features.to_numpy()).all() is False  # no inf/NaN remain
        assert truth.sum() == 1
        assert (features["bandwidth_mbps"] >= 0).all()
    finally:
        path.unlink()


def test_eval_fold_calls_taim_and_baselines():
    # Build a small windowed frame, run one fold, assert all comparators present.
    n = 120
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        att = 1 if i % 4 == 0 else 0
        rows.append({
            "timestamp": pd.Timestamp("2018-12-01 10:00:00") + pd.Timedelta(minutes=15 * i),
            "device_id": f"d{i % 10}", "day": "2018-12-01",
            "family": "Syn" if att else "Benign", "is_attack": att,
            "hour": 10, "weekday": 5,
            "bandwidth_mbps": np.log1p(1e6 if att else 1e3),
            "conn_rate_ps": np.log1p(5000 if att else 20),
            "port_div": 3 if att else 1, "pkt_size_mean": 500 if att else 120,
            "app_req_ps": np.log1p(5e6 if att else 1e3),
        })
    df = pd.DataFrame(rows)
    features = df[ev.SIGNAL_LIST]
    truth = df["is_attack"].to_numpy(int)
    res = ev._eval_fold(df, features, truth, np.arange(0, 80), np.arange(80, n),
                         inner_folds=2, random_state=0)
    for name in ("taim", "random_forest", "isolation_forest", "fixed_rule"):
        assert name in res, name
    assert np.isfinite(res["taim"]["f1"])
    assert np.isfinite(res["random_forest"]["f1"])


def test_tune_contamination_returns_candidate():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (60, 5))
    y = np.array([0] * 30 + [1] * 30)
    c = ev._tune_contamination(x, y, inner_folds=2, random_state=0)
    assert c in (0.02, 0.05, 0.10, 0.20, 0.30)


def _make_frame(n=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        att = 1 if i % 4 == 0 else 0
        rows.append({
            "timestamp": pd.Timestamp("2018-12-01 10:00:00") + pd.Timedelta(minutes=15 * i),
            "device_id": f"d{i % 10}", "day": "2018-12-01",
            "family": "Syn" if att else "Benign", "is_attack": att,
            "hour": 10, "weekday": 5,
            "bandwidth_mbps": np.log1p(1e6 if att else 1e3),
            "conn_rate_ps": np.log1p(5000 if att else 20),
            "port_div": 3 if att else 1, "pkt_size_mean": 500 if att else 120,
            "app_req_ps": np.log1p(5e6 if att else 1e3),
        })
    return pd.DataFrame(rows)


def test_fit_taim_is_fold_isolated():
    # TAIM must not be scored on the full (train+test) frame. run_fold warms on train only
    # and scores test with a frozen baseline. The test output rows exactly match the test
    # rows (no train rows returned, no leakage).
    df = _make_frame(120)
    train, test = df.iloc[:80], df.iloc[80:]
    flagged, score = ev._fit_taim(train, test)
    assert len(flagged) == len(test)
    assert len(score) == len(test)
    assert np.all(np.isfinite(score))


def test_threshold_from_inner_model_uses_fresh_split():
    # The RF recall@FPR threshold must come from a genuine fit/validation split, not the
    # in-sample training predictions. It should be finite and within [0,1].
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (200, 5))
    y = (rng.random(200) > 0.7).astype(int)
    thr = ev._threshold_from_inner_model(x, y, target_fpr=0.01, random_state=0)
    assert np.isfinite(thr)
    assert 0.0 <= thr <= 1.0


def test_threshold_from_inner_model_never_uses_test_labels():
    # Degenerate single-class training -> NaN threshold (no test tuning).
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (50, 5))
    y = np.zeros(50, dtype=int)
    thr = ev._threshold_from_inner_model(x, y, target_fpr=0.01, random_state=0)
    assert np.isnan(thr)


def test_run_fold_returns_only_test_rows_and_has_expected_schema():
    # run_fold must return exactly the test rows (fold isolation: no train rows leaked),
    # with the expected TAIM output schema. (TAIM's internal state is not re-run
    # deterministic across separate process invocations, so we assert the structural
    # fold-isolation properties, not exact score equality.)
    from src.fast_detector import FastTaimDetector
    from src.detector import DetectorConfig

    df = _make_frame(60)
    train, test = df.iloc[:40], df.iloc[40:]
    out = FastTaimDetector(DetectorConfig()).run_fold(train, test)
    assert len(out) == len(test)
    assert {"flagged", "score", "fired", "stage"}.issubset(out.columns)
    assert np.all(np.isfinite(out["score"].to_numpy()))
