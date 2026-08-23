"""Phase 1: Synthetic network-traffic generator.

Simulates a small LAN observed at fixed intervals (default 15 min).
Each device follows a normal behaviour pattern (hour-of-day x day-of-week
seasonality + noise) and optional attack windows are injected. Ground truth
(is_attack, attack_type) is labelled per device and per timestep so that
later phases can be scored objectively.

Signals per device per interval:
  bandwidth_mbps  - throughput (proxy: bytes/s)
  conn_rate_ps    - new connection rate (proxy: SYN/s)
  port_div        - distinct destination ports (protocol diversity)
  pkt_size_mean   - mean packet size (bytes)
  app_req_ps      - application-layer request rate

Attack types:
  volumetric - many devices, big bandwidth jump (aggregate DDoS)
  flood      - one device, huge bandwidth + connection spike
  syn        - connection rate explosion, small packets
  lowslow    - many devices, mild per-device rise (aggregate crosses)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

SIGNALS = ["bandwidth_mbps", "conn_rate_ps", "port_div", "pkt_size_mean", "app_req_ps"]


@dataclass
class AttackWindow:
    """An attack episode injected into the simulation.

    start/end are day-fractions (0.0 .. n_days). devices is a list of
    device ids participating, or None meaning "all devices".
    """

    start_day: float
    end_day: float
    kind: str  # one of ATTACK_TYPES
    devices: Optional[list] = None
    intensity: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in ATTACK_TYPES:
            raise ValueError(f"unknown attack type: {self.kind}")
        if self.end_day <= self.start_day:
            raise ValueError("end_day must be > start_day")


ATTACK_TYPES = ("volumetric", "flood", "syn", "lowslow")


@dataclass
class SimulationConfig:
    n_devices: int = 10
    n_days: int = 42  # 6 weeks: gives plenty of history for walk-forward
    interval_min: int = 15  # steps per day = 96
    seed: int = 42
    noise_sigma: float = 0.35       # multiplicative noise level of the fleet
    diurnal_peak: float = 14.0      # hour of peak activity
    weekend_factor: float = 0.62    # activity multiplier on weekends
    flash_crowds: list = field(default_factory=list)  # (start_day, end_day, boost)
    attack_windows: list = field(default_factory=list)

    @property
    def steps_per_day(self) -> int:
        return 24 * 60 // self.interval_min

    @property
    def n_steps(self) -> int:
        return self.n_days * self.steps_per_day


def _diurnal(hour: float, weekend: bool, peak_hour: float = 14.0,
             weekend_factor: float = 0.62) -> float:
    """Normalized activity level: peak at peak_hour, floor at night."""
    peak = 0.10 + 0.90 * float(np.exp(-(((hour - peak_hour) / 5.0) ** 2)))
    if weekend:
        peak *= weekend_factor
    return peak


def _base_device_scales(rng: np.random.Generator, n: int) -> np.ndarray:
    """Per-device activity scale, log-uniform so the fleet has busy+idle nodes."""
    return np.exp(rng.uniform(np.log(0.15), np.log(3.0), size=n))


def _apply_attack(sig: dict[str, np.ndarray], kind: str, intensity: float) -> None:
    """Apply attack modifiers in place to the per-device signal arrays."""
    m = intensity
    if kind == "volumetric":
        sig["bandwidth_mbps"] *= 4.5 * m
        sig["conn_rate_ps"] *= 2.5 * m
        sig["app_req_ps"] *= 3.0 * m
    elif kind == "flood":
        sig["bandwidth_mbps"] *= 9.0 * m
        sig["conn_rate_ps"] *= 7.0 * m
        sig["app_req_ps"] *= 4.0 * m
    elif kind == "syn":
        sig["conn_rate_ps"] *= 22.0 * m
        sig["pkt_size_mean"] *= 0.45
        sig["bandwidth_mbps"] *= 1.6 * m
    elif kind == "lowslow":
        sig["bandwidth_mbps"] *= 3.0 * m
        sig["conn_rate_ps"] *= 2.5 * m
    else:  # pragma: no cover - guarded by dataclass
        raise ValueError(kind)


def generate_dataset(cfg: SimulationConfig) -> tuple[pd.DataFrame, list[AttackWindow]]:
    """Produce the long-form dataset with ground truth."""
    rng = np.random.default_rng(cfg.seed)
    spd = cfg.steps_per_day
    n_steps = cfg.n_steps
    n_dev = cfg.n_devices

    start = pd.Timestamp("2026-01-01 00:00")
    timestamps = pd.date_range(start, periods=n_steps, freq=f"{cfg.interval_min}min")

    hours = np.arange(n_steps) % spd / spd * 24.0
    day_idx = np.arange(n_steps) // spd
    base_date = pd.Timestamp("2026-01-01").dayofweek  # weekday of 2026-01-01
    weekdays = (day_idx + base_date) % 7
    weekend_mask = (weekdays >= 5).astype(float)

    scales = _base_device_scales(rng, n_dev)

    # ---- build normal signals per device ----
    rows: list[pd.DataFrame] = []
    nsig = cfg.noise_sigma
    peak = cfg.diurnal_peak
    wf = cfg.weekend_factor
    for d in range(n_dev):
        seasonal = np.array([
            _diurnal(h, weekend_mask[i] > 0, peak_hour=peak, weekend_factor=wf)
            for i, h in enumerate(hours)
        ])
        noise_band = np.exp(rng.normal(0.0, nsig, size=n_steps))
        noise_conn = np.exp(rng.normal(0.0, nsig, size=n_steps))
        noise_req = np.exp(rng.normal(0.0, nsig, size=n_steps))

        bw = scales[d] * seasonal * 8.0 * noise_band          # Mbps
        cr = scales[d] * seasonal * 40.0 * noise_conn         # conns/s
        ar = scales[d] * seasonal * 12.0 * noise_req          # reqs/s
        # protocol diversity drifts gently, bounded
        port_div = 6.0 + 10.0 * seasonal + rng.normal(0.0, 1.2, size=n_steps)
        pkt = 700.0 + 250.0 * (0.5 - seasonal) + rng.normal(0.0, 60.0, size=n_steps)

        df_d = pd.DataFrame(
            {
                "timestamp": timestamps,
                "device_id": d,
                "hour": hours,
                "weekday": weekdays,
                "bandwidth_mbps": bw,
                "conn_rate_ps": cr,
                "port_div": port_div.clip(min=1.0),
                "pkt_size_mean": pkt.clip(min=64.0),
                "app_req_ps": ar.clip(min=0.0),
                "is_attack": 0,
                "attack_type": None,
            }
        )
        rows.append(df_d)

    df = pd.concat(rows, ignore_index=True)
    tvals = np.arange(n_steps) / spd  # day fraction per step

    # ---- inject legitimate flash crowds (kept as normal, no label) ----
    for (fc_start, fc_end, boost) in cfg.flash_crowds:
        in_window = (tvals >= fc_start) & (tvals < fc_end)
        steps = np.where(in_window)[0]
        if len(steps) == 0:
            continue
        mask = df["timestamp"].isin(timestamps[steps])
        for col in ("bandwidth_mbps", "conn_rate_ps", "app_req_ps"):
            df.loc[mask, col] = df.loc[mask, col] * boost

    # ---- inject attacks ----
    for w in cfg.attack_windows:
        in_window = (tvals >= w.start_day) & (tvals < w.end_day)
        steps = np.where(in_window)[0]
        if len(steps) == 0:
            continue
        devices = list(range(n_dev)) if w.devices is None else list(w.devices)
        mask = df["device_id"].isin(devices) & df["timestamp"].isin(
            timestamps[in_window]
        )
        # apply modifiers on the selected rows
        sig = {
            "bandwidth_mbps": df.loc[mask, "bandwidth_mbps"].to_numpy(copy=True),
            "conn_rate_ps": df.loc[mask, "conn_rate_ps"].to_numpy(copy=True),
            "port_div": df.loc[mask, "port_div"].to_numpy(copy=True),
            "pkt_size_mean": df.loc[mask, "pkt_size_mean"].to_numpy(copy=True),
            "app_req_ps": df.loc[mask, "app_req_ps"].to_numpy(copy=True),
        }
        _apply_attack(sig, w.kind, w.intensity)
        for col, arr in sig.items():
            df.loc[mask, col] = arr
        df.loc[mask, "is_attack"] = 1
        df.loc[mask, "attack_type"] = w.kind

    df["attack_type"] = df["attack_type"].astype("object")
    return df, list(cfg.attack_windows)


def default_config() -> SimulationConfig:
    """A realistic default schedule: normal periods + each attack type."""
    return SimulationConfig(
        n_devices=10,
        n_days=42,
        interval_min=15,
        seed=42,
        attack_windows=[
            AttackWindow(7.1, 7.4, "volumetric"),        # all devices
            AttackWindow(14.3, 14.7, "syn"),             # all devices
            AttackWindow(21.1, 21.4, "flood", devices=[2]),
            AttackWindow(28.5, 29.2, "lowslow"),         # all devices
            AttackWindow(35.0, 35.3, "flood", devices=[5, 7]),
        ],
    )


if __name__ == "__main__":
    import sys

    cfg = default_config()
    df, windows = generate_dataset(cfg)
    out = Path(__file__).resolve().parents[1] / "data" / "dataset_42d.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"rows: {len(df)}  devices: {df.device_id.nunique()}  steps: {cfg.n_steps}")
    print(f"signals: {SIGNALS}")
    print(f"attack steps (device-level): {int(df.is_attack.sum())} "
          f"({100 * df.is_attack.mean():.2f}% of rows)")
    print("attack windows:")
    for w in windows:
        n = int(((df.timestamp.isin(df.loc[df.attack_type == w.kind].timestamp))).sum())
        print(f"  {w.kind:10s} day {w.start_day}-{w.end_day} devices={w.devices}")
    # sanity: daytime vs night average bandwidth
    day = df[df.hour.between(10, 16) & (df.is_attack == 0)]
    night = df[df.hour.between(0, 4) & (df.is_attack == 0)]
    print(f"\nmean bandwidth daytime (no attack): {day.bandwidth_mbps.mean():.2f} Mbps")
    print(f"mean bandwidth night    (no attack): {night.bandwidth_mbps.mean():.2f} Mbps")
    if df[SIGNALS].isna().any().any():
        print("WARNING: NaNs present in signal columns")
        sys.exit(1)
    if df["attack_type"].isna().sum() != (df["is_attack"] == 0).sum():
        print("WARNING: attack_type labels inconsistent")
        sys.exit(1)
    print("Phase 1 validation: OK")
