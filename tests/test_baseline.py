"""Unit tests for Phase 2: time-window baseline engine."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baseline import BaselineConfig, TimeWindowBaseline, make_slot


def _run(df, device=0, signal="bandwidth_mbps"):
    """Feed rows through the engine, returning list of z-scores."""
    baseline = TimeWindowBaseline()
    zs = []
    for _, row in df.iterrows():
        slot = make_slot(row["weekday"], row["hour"])
        z = baseline.step(device, slot, {signal: row[signal]})
        zs.append(z[signal])
    return baseline, np.array(zs)


def test_warmup_returns_zero():
    cfg = BaselineConfig(min_samples=5)
    b = TimeWindowBaseline(cfg)
    for i in range(5):  # calls 1..5: n<5 at score time -> zero
        z = b.step(0, (0, 10), {"bandwidth_mbps": 1.0 + i})
        assert z["bandwidth_mbps"] == 0.0
    z = b.step(0, (0, 10), {"bandwidth_mbps": 10.0})  # call 6: n=5 -> warm
    assert b.is_warm(0, (0, 10), "bandwidth_mbps")
    assert z["bandwidth_mbps"] > 0.0


def test_steady_stream_stays_near_zero():
    b = TimeWindowBaseline(BaselineConfig(min_samples=10))
    zs = []
    for i in range(400):
        x = 5.0 + 0.3 * np.sin(i / 20.0)
        z = b.step(0, (0, 14), {"s": x})["s"]
        zs.append(z)
    mature = np.array(zs[100:])
    assert np.abs(mature).mean() < 1.0
    assert np.abs(mature).max() < 4.0


def test_spike_is_scored_but_not_absorbed():
    cfg = BaselineConfig(min_samples=5)
    b = TimeWindowBaseline(cfg)
    # normal phase
    for _ in range(200):
        b.step(0, (0, 14), {"s": 5.0})
    # a single huge spike
    z_spike = b.step(0, (0, 14), {"s": 50.0})["s"]
    assert z_spike > 8.0
    # after the spike, the model should be back near mu=5 quickly (outlier resisted)
    mu_after = b.models[(0, 0, 14, "s")].mu
    assert abs(mu_after - 5.0) < 0.3


def test_sustained_attack_does_not_poison_baseline():
    cfg = BaselineConfig(min_samples=5)
    b = TimeWindowBaseline(cfg)
    for _ in range(200):
        b.step(0, (0, 14), {"s": 5.0})
    for _ in range(50):  # a sustained flood
        z = b.step(0, (0, 14), {"s": 40.0})["s"]
        assert z > 5.0  # stays flagged even after 50 attack steps
    mu_after = b.models[(0, 0, 14, "s")].mu
    assert mu_after < 8.0  # model barely moved despite 50 outliers


def test_slow_drift_is_tracked():
    cfg = BaselineConfig(min_samples=5)
    b = TimeWindowBaseline(cfg)
    zs = []
    for i in range(600):
        x = 1.0 + i * 0.02  # gentle ramp
        z = b.step(0, (0, 14), {"s": x})["s"]
        zs.append(z)
    tail = np.array(zs[400:])
    assert np.abs(tail).mean() < 2.0  # baseline adapts, no false alarm


def test_time_window_specificity():
    """Same numeric value must score far higher in a quiet slot than a busy one."""
    cfg = BaselineConfig(min_samples=5)
    b = TimeWindowBaseline(cfg)
    # train night slot (hour 2) at low values and day slot (hour 14) at high values
    for _ in range(200):
        b.step(0, (0, 2), {"s": 1.0})
        b.step(0, (0, 14), {"s": 8.0})
    z_night = b.score(0, (0, 2), {"s": 6.0})["s"]
    z_day = b.score(0, (0, 14), {"s": 6.0})["s"]
    assert z_night > z_day
    assert z_night > 3.0
    assert z_day < 1.0


def test_slot_isolation():
    """A spike in one slot must not disturb another slot's model."""
    b = TimeWindowBaseline(BaselineConfig(min_samples=5))
    for _ in range(100):
        b.step(0, (0, 2), {"s": 1.0})
        b.step(0, (0, 14), {"s": 8.0})
    b.step(0, (0, 2), {"s": 100.0})  # blow up the night slot
    z_day = b.score(0, (0, 14), {"s": 8.0})["s"]
    assert abs(z_day) < 1.5  # day model unaffected


def test_no_nan_and_clipped_zscores():
    cfg = BaselineConfig(min_samples=3)
    b = TimeWindowBaseline(cfg)
    for _ in range(3):
        b.step(0, (0, 14), {"s": 5.0})
    z = b.step(0, (0, 14), {"s": 1e9})["s"]
    assert z == cfg.z_max  # clipped, not inf/nan


def test_make_slot_weekend():
    assert make_slot(5, 14.0) == (1, 14)
    assert make_slot(2, 3.7) == (0, 3)
