"""Unit tests for Phase 1: synthetic data generator."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_gen import (  # noqa: E402
    SIGNALS,
    AttackWindow,
    SimulationConfig,
    generate_dataset,
)


def make_cfg() -> SimulationConfig:
    return SimulationConfig(
        n_devices=6,
        n_days=7,
        interval_min=15,
        seed=7,
        attack_windows=[
            AttackWindow(1.0, 1.3, "flood", devices=[0]),
            AttackWindow(3.2, 3.5, "syn"),
        ],
    )


def test_shape_and_schema():
    cfg = make_cfg()
    df, _ = generate_dataset(cfg)
    assert len(df) == cfg.n_devices * cfg.n_steps
    assert list(df.columns[:4]) == ["timestamp", "device_id", "hour", "weekday"]
    for s in SIGNALS:
        assert s in df.columns
    assert df["is_attack"].isin([0, 1]).all()


def test_no_nans_in_signals():
    cfg = make_cfg()
    df, _ = generate_dataset(cfg)
    assert not df[SIGNALS].isna().any().any()
    assert df["attack_type"].isna().sum() == (df["is_attack"] == 0).sum()


def test_determinism():
    cfg = make_cfg()
    df1, _ = generate_dataset(cfg)
    df2, _ = generate_dataset(cfg)
    pd.testing.assert_frame_equal(df1, df2)


def test_ground_truth_matches_window():
    cfg = make_cfg()
    df, windows = generate_dataset(cfg)
    # flood window, device 0
    tmin = pd.Timestamp("2026-01-01") + pd.Timedelta(days=1.0)
    tmax = pd.Timestamp("2026-01-01") + pd.Timedelta(days=1.3)
    in_win = df["device_id"].eq(0) & df["timestamp"].between(tmin, tmax, inclusive="left")
    assert in_win.sum() > 0
    assert (df.loc[in_win, "is_attack"] == 1).all()
    assert (df.loc[in_win, "attack_type"] == "flood").all()
    # device 0 outside ALL attack windows must be clean
    attack_ts = set(df.loc[df["device_id"].eq(0) & (df["is_attack"] == 1), "timestamp"])
    clean = df["device_id"].eq(0) & ~df["timestamp"].isin(attack_ts)
    assert (df.loc[clean, "is_attack"] == 0).all()
    assert len(windows) == 2


def test_diurnal_seasonality():
    cfg = make_cfg()
    df, _ = generate_dataset(cfg)
    clean = df[df["is_attack"] == 0]
    day = clean[clean["hour"].between(10, 16)]
    night = clean[clean["hour"].between(0, 4)]
    assert day["bandwidth_mbps"].mean() > 3 * night["bandwidth_mbps"].mean()


def test_attack_types_raise_signals():
    """Attack rows must be elevated vs CLEAN rows at the SAME time-of-day."""
    cfg = make_cfg()
    df, _ = generate_dataset(cfg)
    clean = df[df["is_attack"] == 0]
    for kind in ["flood", "syn"]:
        attack = df[df["attack_type"] == kind]
        hours = attack["hour"].round().unique()
        ref = clean[clean["hour"].round().isin(hours)]
        if kind == "syn":
            assert attack["conn_rate_ps"].mean() > 5 * ref["conn_rate_ps"].mean()
        else:
            assert attack["bandwidth_mbps"].mean() > 4 * ref["bandwidth_mbps"].mean()


def test_invalid_window_rejected():
    import pytest

    with pytest.raises(ValueError):
        AttackWindow(2.0, 1.0, "flood")
    with pytest.raises(ValueError):
        AttackWindow(1.0, 2.0, "nope")


def test_seed_isolation():
    cfg_a = make_cfg()
    cfg_b = make_cfg()
    cfg_b.seed = 99
    df_a, _ = generate_dataset(cfg_a)
    df_b, _ = generate_dataset(cfg_b)
    assert not df_a.equals(df_b)
