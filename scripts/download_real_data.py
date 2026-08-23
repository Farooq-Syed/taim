"""Download the official CICDDoS2019 flow captures from the Hugging Face mirror.

The official CICDDoS2019 site requires a manual registration form; this script pulls the
same full schema from the public `bencorn/CICDDoS2019` HF mirror (CC-BY-4.0). Two captures:

  - CSV-01-12.zip  (2.3 GB uncompressed inside; capture day 2018-12-01)
  - CSV-03-11.zip  (0.9 GB; capture day 2019-03-11)

Usage:
  python scripts/download_real_data.py
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = "bencorn/CICDDoS2019"
FILES = ["csvs/CSV-01-12.zip", "csvs/CSV-03-11.zip"]
DEST = Path("data/real/csvs")


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import hf_hub_download

    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        local = DEST / Path(name).name
        if local.exists() and local.stat().st_size > 1_000_000:
            print(f"  cached {local.name} ({local.stat().st_size/1e6:.0f} MB)")
            continue
        print(f"  downloading {name} ...")
        p = hf_hub_download(repo_id=REPO, filename=name, repo_type="dataset",
                            local_dir=str(DEST))
        print(f"  -> {Path(p).name}")
    print("Done.")


if __name__ == "__main__":
    main()
