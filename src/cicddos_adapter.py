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


def parse_flows(paths: list[Path], bucket_min: int = 15, include_metadata: bool = False) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        # Official CICDDoS2019 uses 'Source IP'; adapt both the CICDDoS2019 and the
        # generic 'Src IP' naming.
        src_col = "Source IP" if "Source IP" in df.columns else "Src IP"
        dst_port_col = "Destination Port" if "Destination Port" in df.columns else "Dst Port"
        ts_col = "Timestamp" if "Timestamp" in df.columns else "@timestamp"
        df["@timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
        df["device_id"] = df[src_col].astype(str)
        df["flow_bytes"] = pd.to_numeric(df["Flow Bytes/s"], errors="coerce").fillna(0)
        df["flow_packets"] = pd.to_numeric(df["Flow Packets/s"], errors="coerce").fillna(0)
        df["dst_port"] = df[dst_port_col].astype(str)
        df["label_raw"] = df["Label"].astype(str).str.strip()
        df["is_benign"] = df["label_raw"].str.lower().eq("benign")
        df["family"] = _family_from_label(df[["label_raw", "is_benign"]])
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["@timestamp"])
    data["bucket"] = data["@timestamp"].dt.floor(f"{bucket_min}min")

    rows = []
    for (bucket, device), group in data.groupby(["bucket", "device_id"], sort=False):
        flows = len(group)
        port_div = group["dst_port"].nunique() / flows if flows else 0.0
        pkt = (group["flow_bytes"] / group["flow_packets"].replace(0, pd.NA)).mean()
        has_attack = bool((~group["is_benign"]).any())
        attack_type = group.loc[~group["is_benign"], "label_raw"].iloc[0] if has_attack else "Benign"
        row = {
            "timestamp": bucket,
            "device_id": device,
            "hour": bucket.hour,
            "weekday": bucket.weekday(),
            "bandwidth_mbps": round(float(group["flow_bytes"].sum() * 8 / 1e6), 4),
            "conn_rate_ps": int(flows),
            "port_div": round(float(port_div), 4),
            "pkt_size_mean": round(float(pkt) if pd.notna(pkt) else 0.0, 4),
            "app_req_ps": round(float(group["flow_packets"].sum()), 4),
            "is_attack": int(has_attack),
            "attack_type": attack_type,
        }
        if include_metadata:
            row["family"] = "Benign" if not has_attack else attack_type
            row["day"] = bucket.date().isoformat()
        rows.append(row)
    return pd.DataFrame(rows)


def _family_from_label(df: pd.DataFrame) -> str:
    """Map a CICDDoS2019 label frame (label_raw, is_benign) to a family string."""
    if df["is_benign"].all():
        return "Benign"
    fam = df.loc[~df["is_benign"], "label_raw"].iloc[0]
    return "Benign" if not fam else fam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", nargs="+", required=True, help="CICDDoS2019 flow CSVs.")
    ap.add_argument("--bucket-min", type=int, default=15)
    ap.add_argument("--output", default="data/cicddos_real_windows.csv")
    ap.add_argument("--include-metadata", action="store_true",
                    help="Attach 'family' and 'day' columns so strict family/day hold-out "
                         "evaluation is possible. Off by default.")
    args = ap.parse_args()

    frame = parse_flows([Path(p) for p in args.flows], args.bucket_min, args.include_metadata)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"Wrote {len(frame)} CICDDoS windows (real) -> {out}")
    print(f"  attack windows: {(frame['is_attack'] == 1).sum()}  benign: {(frame['is_attack'] == 0).sum()}")
    if args.include_metadata and "family" in frame.columns:
        print(f"  families: {frame['family'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
