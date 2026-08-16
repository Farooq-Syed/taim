"""Unit tests for Phase 3: multi-signal scoring fusion."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scoring import FusionConfig, MultiSignalFusion


def test_single_signal_alone_is_suppressed():
    f = MultiSignalFusion()
    zs = {"bandwidth_mbps": 9.0, "conn_rate_ps": 0.2, "app_req_ps": 0.1}
    r = f.fuse(zs)
    assert r.score == 0.0
    assert r.fired is False
    assert r.n_elevated == 1


def test_two_signals_fire_and_scale():
    f = MultiSignalFusion()
    r1 = f.fuse({"bandwidth_mbps": 5.0, "conn_rate_ps": 5.0, "app_req_ps": 0.1})
    r2 = f.fuse({"bandwidth_mbps": 7.0, "conn_rate_ps": 7.0, "app_req_ps": 0.1})
    assert r1.fired and r1.score > 0.0
    assert r2.score > r1.score
    assert r2.n_elevated == 2


def test_down_signal_counts_when_low():
    """SYN flood: packet size drops while conn rate rises."""
    f = MultiSignalFusion()
    zs = {"conn_rate_ps": 9.0, "pkt_size_mean": -7.0, "bandwidth_mbps": 0.3}
    r = f.fuse(zs)
    assert r.fired
    assert r.n_elevated == 2
    # a positive packet-size z (packets GREW) must NOT count as anomalous
    zs2 = {"conn_rate_ps": 9.0, "pkt_size_mean": 7.0, "bandwidth_mbps": 0.3}
    assert f.fuse(zs2).n_elevated == 1


def test_saturation_caps_score():
    f = MultiSignalFusion(FusionConfig(z_elevation=2.0, z_saturate=8.0))
    r = f.fuse({"bandwidth_mbps": 1e6, "conn_rate_ps": 1e6, "app_req_ps": 0.1})
    assert r.score <= 1.0
    assert r.elevations["bandwidth_mbps"] == 1.0


def test_weights_change_score():
    cfg = FusionConfig(signal_weights={"bandwidth_mbps": 3.0})
    f1 = MultiSignalFusion()
    f2 = MultiSignalFusion(cfg)
    zs = {"bandwidth_mbps": 7.0, "conn_rate_ps": 3.0}
    r1 = f1.fuse(zs)
    r2 = f2.fuse(zs)
    assert r2.score > r1.score


def test_below_elevation_threshold_ignored():
    f = MultiSignalFusion(FusionConfig(z_elevation=2.0))
    zs = {"bandwidth_mbps": 1.8, "conn_rate_ps": 1.5, "app_req_ps": 1.9}
    assert f.fuse(zs).fired is False


def test_min_signals_config_respected():
    cfg = FusionConfig(min_signals=3)
    f = MultiSignalFusion(cfg)
    zs = {"bandwidth_mbps": 5.0, "conn_rate_ps": 5.0, "app_req_ps": 0.2}
    assert f.fuse(zs).fired is False
    zs["app_req_ps"] = 5.0
    assert f.fuse(zs).fired is True


def test_negative_z_never_fires_down_unless_configured():
    f = MultiSignalFusion()
    # bandwidth going DOWN is not anomalous by default
    r = f.fuse({"bandwidth_mbps": -9.0, "conn_rate_ps": 5.0})
    assert r.n_elevated == 1
