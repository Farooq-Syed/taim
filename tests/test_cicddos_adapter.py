"""Tests for the CICDDoS2019 -> taim windowed-telemetry adapter."""

import tempfile
from pathlib import Path

import pandas as pd

from src.cicddos_adapter import parse_flows


def _write_flows(path: Path) -> None:
    rows = [
        # one window: 3 flows from one src IP, all benign
        {"Timestamp": "2019-01-01 10:00:00", "Src IP": "10.0.0.10", "Dst Port": 443,
         "Flow Bytes/s": 100000, "Flow Packets/s": 100, "Label": "Benign"},
        {"Timestamp": "2019-01-01 10:02:00", "Src IP": "10.0.0.10", "Dst Port": 443,
         "Flow Bytes/s": 200000, "Flow Packets/s": 150, "Label": "Benign"},
        {"Timestamp": "2019-01-01 10:05:00", "Src IP": "10.0.0.10", "Dst Port": 443,
         "Flow Bytes/s": 300000, "Flow Packets/s": 200, "Label": "Benign"},
        # another window: attack flows from a second src IP
        {"Timestamp": "2019-01-01 10:07:00", "Src IP": "10.0.0.20", "Dst Port": 80,
         "Flow Bytes/s": 1000000, "Flow Packets/s": 5000, "Label": "SYN"},
        {"Timestamp": "2019-01-01 10:09:00", "Src IP": "10.0.0.20", "Dst Port": 80,
         "Flow Bytes/s": 2000000, "Flow Packets/s": 6000, "Label": "SYN"},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_parse_flows_schema_and_aggregation():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "flows.csv"
        _write_flows(path)
        frame = parse_flows([path], bucket_min=15)
        assert {"timestamp", "device_id", "bandwidth_mbps", "conn_rate_ps",
                "port_div", "pkt_size_mean", "app_req_ps", "is_attack", "attack_type"}.issubset(frame.columns)
        assert len(frame) == 2  # two (device, bucket) groups
        benign = frame[frame["is_attack"] == 0].iloc[0]
        attack = frame[frame["is_attack"] == 1].iloc[0]
        # attack device has much heavier traffic (real DDoS signal)
        assert attack["bandwidth_mbps"] > benign["bandwidth_mbps"]


def test_parse_flows_official_schema_and_metadata():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "flows.csv"
        # Official CICDDoS2019 schema (Source IP / Destination Port) with uppercase BENIGN.
        rows = [
            {"Timestamp": "2018-12-01 10:00:00", "Source IP": "172.16.0.5",
             "Destination Port": 443, "Flow Bytes/s": 100000, "Flow Packets/s": 100,
             "Label": "BENIGN"},
            {"Timestamp": "2018-12-01 10:02:00", "Source IP": "172.16.0.5",
             "Destination Port": 443, "Flow Bytes/s": 200000, "Flow Packets/s": 150,
             "Label": "BENIGN"},
            {"Timestamp": "2018-12-01 10:07:00", "Source IP": "172.16.0.6",
             "Destination Port": 80, "Flow Bytes/s": 1000000, "Flow Packets/s": 5000,
             "Label": "Syn"},
        ]
        pd.DataFrame(rows).to_csv(path, index=False)
        frame = parse_flows([path], bucket_min=15, include_metadata=True)
        assert len(frame) == 2
        assert {"family", "day"}.issubset(frame.columns)
        benign = frame[frame["is_attack"] == 0].iloc[0]
        attack = frame[frame["is_attack"] == 1].iloc[0]
        assert benign["family"] == "Benign"
        assert attack["family"] == "Syn"
        assert frame["day"].nunique() == 1


def test_parse_flows_preserves_capture_id_for_day():
    # The day tag must come from the source capture identifier, NOT the per-file internal
    # (drifted) timestamp. Pass a '_capture' source column and assert it is preserved as day.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "flows.csv"
        rows = [
            {"Timestamp": "2018-12-01 10:00:00", "Source IP": "172.16.0.5",
             "Destination Port": 443, "Flow Bytes/s": 100000, "Flow Packets/s": 100,
             "Label": "BENIGN"},
            {"Timestamp": "2019-03-11 10:00:00", "Source IP": "172.16.0.6",
             "Destination Port": 80, "Flow Bytes/s": 1000000, "Flow Packets/s": 5000,
             "Label": "Syn"},
        ]
        frame_rows = pd.DataFrame(rows)
        frame_rows["_capture"] = "2018-12-01"
        frame_rows.to_csv(path, index=False)
        frame = parse_flows([path], bucket_min=15, include_metadata=True)
        assert frame["day"].unique()[0] == "2018-12-01"
        assert "capture_id" in frame.columns
