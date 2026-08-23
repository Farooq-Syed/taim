"""CICDDoS2019 (real DDoS flows) -> taim per-device windowed telemetry.

taim consumes a long format of per-device signal windows:
  timestamp, device_id, hour, weekday, bandwidth_mbps, conn_rate_ps, port_div,
  pkt_size_mean, app_req_ps, is_attack, attack_type

This adapter converts CICDDoS2019 flow records (the real, public CC-BY-4.0 dataset) into
that schema by treating each source IP as a "device" and each time bucket as a window:

  bandwidth_mbps  = sum(Flow Bytes/s) * 8 / 1e6 per bucket     (throughput)
  conn_rate_ps    = number of flows per bucket                 (connection rate)
  port_div        = distinct destination ports / flows         (protocol diversity)
  pkt_size_mean   = mean(Flow Bytes/s / Flow Packets/s)        (avg packet size)
  app_req_ps      = sum(Flow Packets/s)                        (packet/request rate)
  is_attack       = (Label != 'Benign'); attack_type = Label

Usage (requires the downloaded CICDDoS2019 CSVs):
  python cicddos_adapter.py --flows CICDDoS/*.csv --bucket-min 15 --output data/cicddos_real_windows.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_flows(paths: list[Path], bucket_min: int = 15) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path, low_memory=False)
        df["@timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["device_id"] = df["Src IP"].astype(str)
        df["flow_bytes"] = pd.to_numeric(df["Flow Bytes/s"], errors="coerce").fillna(0)
        df["flow_packets"] = pd.to_numeric(df["Flow Packets/s"], errors="coerce").fillna(0)
        df["dst_port"] = df["Dst Port"].astype(str)
        df["label_raw"] = df["Label"].astype(str)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["@timestamp"])
    data["bucket"] = data["@timestamp"].dt.floor(f"{bucket_min}min")

    rows = []
    for (bucket, device), group in data.groupby(["bucket", "device_id"], sort=False):
        flows = len(group)
        port_div = group["dst_port"].nunique() / flows if flows else 0.0
        pkt = (group["flow_bytes"] / group["flow_packets"].replace(0, pd.NA)).mean()
        is_attack = bool((group["label_raw"] != "Benign").any())
        attack_type = group["label_raw"].iloc[0] if is_attack else "Benign"
        rows.append({
            "timestamp": bucket,
            "device_id": device,
            "hour": bucket.hour,
            "weekday": bucket.weekday(),
            "bandwidth_mbps": round(float(group["flow_bytes"].sum() * 8 / 1e6), 4),
            "conn_rate_ps": int(flows),
            "port_div": round(float(port_div), 4),
            "pkt_size_mean": round(float(pkt) if pd.notna(pkt) else 0.0, 4),
            "app_req_ps": round(float(group["flow_packets"].sum()), 4),
            "is_attack": int(is_attack),
            "attack_type": attack_type,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", nargs="+", required=True, help="CICDDoS2019 flow CSVs.")
    ap.add_argument("--bucket-min", type=int, default=15)
    ap.add_argument("--output", default="data/cicddos_real_windows.csv")
    args = ap.parse_args()

    frame = parse_flows([Path(p) for p in args.flows], args.bucket_min)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"Wrote {len(frame)} CICDDoS windows (real) -> {out}")
    print(f"  attack windows: {(frame['is_attack'] == 1).sum()}  benign: {(frame['is_attack'] == 0).sum()}")


if __name__ == "__main__":
    main()
