"""Shared plotting helpers for TAIM reports."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_aggregate_signal(
    df: pd.DataFrame,
    signal: str,
    windows: list,
    title: str,
    out_path: str,
    attack_windows_by_kind: dict | None = None,
) -> None:
    """Plot a signal aggregated across devices, shading attack windows."""
    agg = (
        df.groupby("timestamp")[signal]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(agg["timestamp"], agg[signal], lw=0.8, color="#1f77b4")
    ax.set_title(title)
    ax.set_ylabel(signal)
    ax.set_xlabel("date")

    if attack_windows_by_kind is None:
        attack_windows_by_kind = {}
    start = pd.Timestamp("2026-01-01")
    colors = {
        "volumetric": "#d62728",
        "flood": "#ff7f0e",
        "syn": "#9467bd",
        "lowslow": "#8c564b",
    }
    for w in windows:
        kind = w.kind
        lo = start + pd.Timedelta(days=w.start_day)
        hi = start + pd.Timedelta(days=w.end_day)
        ax.axvspan(lo, hi, alpha=0.25, color=colors.get(kind, "gray"), label=kind)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        seen = {}
        for h, l in zip(handles, labels):
            seen[l] = h
        ax.legend(list(seen.values()), list(seen.keys()), fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_signal_matrix(
    df: pd.DataFrame, signals: list[str], out_path: str, sample_hours: int = 96
) -> None:
    """Per-signal aggregate plots for a preview dashboard."""
    fig, axes = plt.subplots(len(signals), 1, figsize=(14, 2.2 * len(signals)), sharex=True)
    for ax, s in zip(axes, signals):
        agg = df.groupby("timestamp")[s].mean()
        ax.plot(agg.index, agg.values, lw=0.7, color="#2ca02c")
        ax.set_ylabel(s, fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_fold_metrics(folds: list[dict], out_path: str) -> None:
    """Per-fold TPR/FPR from a walk-forward run."""
    days = [f["test_day"] for f in folds]
    tpr = [f["tpr"] for f in folds]
    fpr = [f["fpr"] for f in folds]
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(days, tpr, "o-", color="#1f77b4", label="fold TPR")
    ax1.set_ylabel("TPR", color="#1f77b4")
    ax1.set_xlabel("test day")
    ax2 = ax1.twinx()
    ax2.plot(days, fpr, "s-", color="#d62728", label="fold FPR")
    ax2.set_ylabel("FPR", color="#d62728")
    ax1.set_ylim(0, 1.05)
    ax2.set_ylim(0, 0.4)
    fig.suptitle("Walk-forward per-fold performance (TPR vs FPR)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_device_stages(
    out: pd.DataFrame, device: int, windows: list, out_path: str, start: pd.Timestamp
) -> None:
    """Ladder stage over time for one device, with attack windows shaded."""
    dev = out[out["device_id"] == device]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.step(dev["timestamp"], dev["stage"], where="post", lw=1.0, color="#1f77b4")
    ax.set_ylim(-0.1, 4.1)
    ax.set_yticks(range(5))
    ax.set_ylabel("ladder stage")
    ax.set_xlabel("date")
    colors = {"volumetric": "#d62728", "flood": "#ff7f0e", "syn": "#9467bd", "lowslow": "#8c564b"}
    for w in windows:
        lo = start + pd.Timedelta(days=w.start_day)
        hi = start + pd.Timedelta(days=w.end_day)
        ax.axvspan(lo, hi, alpha=0.2, color=colors.get(w.kind, "gray"))
    ax.set_title(f"Device {device}: response-ladder stage over time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
