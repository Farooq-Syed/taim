"""Build a real CICDDoS2019 windowed dataset for TAIM from the official captures.

The official CICDDoS2019 flow CSVs are large; this script streams a bounded,
deterministic per-family row sample from the two captures and uses the TAIM adapter
to produce per-(device, bucket) signal windows, tagging each with its attack family
and capture day. The output is what the strict-split real-data evaluation consumes.

Inputs (from scripts/download_real_data.py): the two official capture zips under
data/real/csvs/. Columns are the official CICFlowMeter schema (Source IP, Timestamp,
Destination Port, Flow Bytes/s, Flow Packets/s, Label).

Usage:
  python scripts/build_real_windows.py --rows-per-family 800000 --bucket-min 15
      --output data/cicddos_real_windows.csv
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.cicddos_adapter import parse_flows  # noqa: E402

CAPTURES = {
    "01-12": "2018-12-01",
    "03-11": "2019-03-11",
}
DEFAULT_ZIPS = {
    "01-12": "data/real/csvs/CSV-01-12.zip",
    "03-11": "data/real/csvs/CSV-03-11.zip",
}
# Families in each capture to include (skip the 9.3 GB TFTP file and empty dirs).
INCLUDE = {
    "01-12": ["Syn.csv", "DrDoS_UDP.csv", "DrDoS_SSDP.csv", "UDPLag.csv",
              "DrDoS_NTP.csv", "DrDoS_LDAP.csv", "DrDoS_MSSQL.csv", "DrDoS_DNS.csv",
              "DrDoS_NetBIOS.csv", "DrDoS_SNMP.csv"],
    "03-11": ["LDAP.csv", "MSSQL.csv", "NetBIOS.csv", "Portmap.csv", "Syn.csv",
              "UDP.csv", "UDPLag.csv"],
}


def _stream_sample(zip_path: Path, member: str, n_rows: int) -> pd.DataFrame:
    """Read up to n_rows from a CSV inside a zip, stripping header whitespace."""
    parts = []
    got = 0
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            for chunk in pd.read_csv(fh, low_memory=False, chunksize=250_000):
                chunk.columns = [c.strip() for c in chunk.columns]
                parts.append(chunk)
                got += len(chunk)
                if got >= n_rows:
                    break
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a real CICDDoS2019 windowed dataset.")
    ap.add_argument("--rows-per-family", type=int, default=800_000,
                    help="Max rows sampled per family file (keeps memory bounded).")
    ap.add_argument("--bucket-min", type=int, default=1,
                    help="Window bucket in minutes. 1 min is the default because each "
                         "CICDDoS2019 family attack burst lasts only a few minutes; coarser "
                         "buckets collapse a family's attacks into 1-2 windows and give no "
                         "per-family test support.")
    ap.add_argument("--output", default="data/cicddos_real_windows.csv")
    ap.add_argument("--zips", nargs="+", default=[DEFAULT_ZIPS[c] for c in CAPTURES],
                    help="Paths to the two capture zips (01-12, 03-11).")
    args = ap.parse_args()

    zip_paths = {cap: Path(p) for cap, p in zip(CAPTURES, args.zips)}
    all_parts: list[str] = []
    temp_dir = Path("data/real/_build")
    temp_dir.mkdir(parents=True, exist_ok=True)
    for cap, date in CAPTURES.items():
        if not zip_paths[cap].exists():
            print(f"  skip capture {cap}: zip not present at {zip_paths[cap]}")
            continue
        # Write each family sample as a temp CSV so the adapter consumes it uniformly.
        for member in INCLUDE[cap]:
            full = f"{cap}/{member}"
            try:
                sample = _stream_sample(zip_paths[cap], full, args.rows_per_family)
            except (KeyError, FileNotFoundError):
                print(f"  (skip missing member {full})")
                continue
            if len(sample) == 0:
                continue
            tmp = temp_dir / f"{cap}_{Path(member).stem}.csv"
            sample["_capture"] = date
            sample.to_csv(tmp, index=False)
            all_parts.append(str(tmp))
            print(f"  captured {cap}/{member}: {len(sample):,} rows")

    if not all_parts:
        print("No data captured; check the capture zips exist under data/real/csvs/.")
        return

    frame = parse_flows([Path(p) for p in all_parts], args.bucket_min, include_metadata=True)
    # The capture-day identifier is preserved directly from the source capture (in the
    # adapter), not re-derived from each file's internal (drifted) timestamps.
    frame["day"] = frame["capture_id"].astype(str)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"Wrote {len(frame):,} real CICDDoS windows -> {out}")
    print(f"  attack: {int(frame['is_attack'].sum()):,}  benign: {int((frame['is_attack'] == 0).sum()):,}")
    print(f"  families: {frame['family'].value_counts().to_dict()}")
    print(f"  days: {frame['day'].nunique()} -> {sorted(frame['day'].unique())}")


if __name__ == "__main__":
    main()
