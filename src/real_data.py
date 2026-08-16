"""Real-data validation on NSL-KDD (real KDD99 tcpdump flows, labeled).

NSL-KDD is class-balanced (46.5% attack) and has no per-source IPs, so it does
not directly match anomaly-detection assumptions. We construct a realistic
timeline instead: real NORMAL flows form the background network, and bursts of
real ATTACK flows are inserted as attack windows (attack buckets ~4-6% of the
timeline). Signals are derived from the real flow features:
  bandwidth   = sum(src_bytes+dst_bytes) per interval
  conn_rate   = flows per second
  port_div    = distinct services
  pkt_size    = mean flow size (proxy; KDD has no per-packet counts)
  app_req     = flows per second (proxy)

This is the closest real-world test the format allows. Caveats are reported
honestly alongside the numbers.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\Farooq Syed\taim")
from src.data_gen import AttackWindow, SimulationConfig, generate_dataset  # noqa: F401
from src.detector import DetectorConfig
from src.fast_detector import FastTaimDetector
from src.ml_experiment import eval_env, run_with_scorer
from src.temporal import PCAWindowScorer, WindowedMeanScorer

KDD = r"C:\Users\Farooq Syed\taim\data\real\KDDTrain+.txt"
COLS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate", "label", "difficulty"]


def build_timeline(n_bursts: int = 5, bucket_s: int = 60,
                   burst_flows: int = 400, seed: int = 0) -> pd.DataFrame:
    df = pd.read_csv(KDD, header=None, names=COLS)
    normal = df[df["label"] == "normal"].reset_index(drop=True)
    attacks = df[df["label"] != "normal"].reset_index(drop=True)
    print(f"normal flows={len(normal)}  attack flows={len(attacks)}")

    rng = np.random.default_rng(seed)
    # chunk the normal stream, then splice attack bursts in between
    total = len(normal) + n_bursts * burst_flows
    n_chunks = n_bursts + 1
    per = len(normal) // n_chunks
    seq = []
    for i in range(n_chunks):
        lo, hi = i * per, min((i + 1) * per, len(normal))
        seq.append(normal.iloc[lo:hi])
        if i < n_bursts:
            start = int(rng.integers(0, max(len(attacks) - burst_flows, 1)))
            seq.append(attacks.iloc[start:start + burst_flows])
    t = pd.concat(seq, ignore_index=True)

    # assign a timestamp per flow (1 flow/second) and bucket
    start = pd.Timestamp("2026-01-01 00:00")
    t["ts"] = start + pd.to_timedelta(np.arange(len(t)), unit="s")
    t = t.sort_values("ts").reset_index(drop=True)
    t["bucket"] = (t["ts"] - start).dt.total_seconds() // bucket_s
    t["is_attack"] = (t["label"] != "normal").astype(int)

    g = t.groupby("bucket")
    out = pd.DataFrame({
        "timestamp": (g["ts"].first() + pd.Timedelta(seconds=bucket_s)).to_numpy(),
        "bandwidth_mbps": g.apply(lambda x: (x["src_bytes"] + x["dst_bytes"]).sum()) / bucket_s * 8 / 1e6,
        "conn_rate_ps": g["src_bytes"].count() / bucket_s,
        "port_div": g["service"].nunique(),
        "pkt_size_mean": g.apply(lambda x: (x["src_bytes"] + x["dst_bytes"]).mean()),
        "app_req_ps": g["src_bytes"].count() / bucket_s,
        "is_attack": g["is_attack"].max(),
        "attack_frac": g["is_attack"].mean(),
    }).reset_index()
    out["hour"] = (out["timestamp"].dt.hour + out["timestamp"].dt.minute / 60)
    out["weekday"] = (out["timestamp"].dt.dayofweek)
    out["attack_type"] = None
    print(f"buckets={len(out)}  attack buckets={int(out['is_attack'].sum())} "
          f"({out['is_attack'].mean():.1%})")
    return out


def run_real(seed: int = 0) -> None:
    df = build_timeline(seed=seed)
    # single "device" (no per-source IPs available) -> device_id 0
    df["device_id"] = 0
    systems = {
        "A_current": lambda: FastTaimDetector(DetectorConfig()).run(df),
        "B_windowmean": lambda: run_with_scorer(df, WindowedMeanScorer),
        "C_pca_ae": lambda: run_with_scorer(df, PCAWindowScorer),
    }
    print("\n================ REAL DATA (NSL-KDD) ================")
    for name, fn in systems.items():
        out = fn()
        y = out["is_attack"].astype(bool)
        p = out["flagged"].astype(bool)
        tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum())
        tn = int((~p & ~y).sum())
        tpr = tp / (tp + fn) if tp + fn else np.nan
        fpr = fp / (fp + tn) if fp + tn else np.nan
        prec = tp / (tp + fp) if tp + fp else np.nan
        f1 = 2 * prec * tpr / (prec + tpr) if prec + tpr else 0.0
        print(f"{name}: TPR={tpr:.3f} FPR={fpr:.4f} precision={prec:.3f} F1={f1:.3f} "
              f"(tp={tp} fp={fp} fn={fn} tn={tn})")


if __name__ == "__main__":
    run_real()