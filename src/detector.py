"""Phase 5a: End-to-end TAIM detector.

Wires together baseline (Phase 2), fusion (Phase 3) and ladder (Phase 4) and
runs them causally (score-then-update) over a dataset, producing a per-device
per-timestep decision:

    score  - composite suspicion in [0,1] (device-level maxed with aggregate)
    fired  - fusion rule satisfied (>= min_signals elevated)
    stage  - ladder stage 0..4
    action - mitigation action for the stage
    flagged- mitigation engaged (stage >= 2)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.baseline import BaselineConfig, TimeWindowBaseline, hour_slot
from src.ladder import LadderConfig, ResponseLadder
from src.scoring import FusionConfig, MultiSignalFusion

AGG_DEVICE = "agg"

SIGNAL_LIST = ["bandwidth_mbps", "conn_rate_ps", "port_div", "pkt_size_mean", "app_req_ps"]
SUM_SIGNALS = ["bandwidth_mbps", "conn_rate_ps", "app_req_ps"]


class DetectorConfig:
    def __init__(
        self,
        baseline: BaselineConfig | None = None,
        fusion: FusionConfig | None = None,
        ladder: LadderConfig | None = None,
        slot_fn=hour_slot,
        broad_frac: float = 0.5,
        bw_frac_z: float = 1.0,
        bw_frac_threshold: float = 0.5,
        broad_bw_steps: int = 3,
    ) -> None:
        self.baseline = baseline or BaselineConfig(
            min_samples=3,
            sigma_floor_rel=0.5,
            alpha=0.05,
            signal_floor_abs={"pkt_size_mean": 120.0, "port_div": 2.0},
            signal_floor_rel={"pkt_size_mean": 0.0, "port_div": 0.0},
        )
        self.fusion = fusion or FusionConfig(z_elevation=2.0, z_saturate=6.0)
        self.ladder = ladder or LadderConfig(
            score_high=0.30, score_low=0.15, sustain_z=1.5, sustain_steps=12
        )
        self.slot_fn = slot_fn
        # A broad (volumetric / low-and-slow) elevation implicates the whole
        # LAN, so the aggregate signal is applied to ALL devices when:
        #   * >= broad_frac devices are individually suspicious, OR
        #   * >= bw_frac_threshold devices show elevated bandwidth (bw_frac_z).
        # The bandwidth-activity fraction is the key discriminator: an
        # all-hands lowslow shows ~70% of devices active, a targeted flood of
        # a couple of devices shows ~13% -> so innocent devices stay unflagged.
        # When the bandwidth-activity gate fires it also injects a minimum
        # score (broad_bw_score) so the ladder's fast path escalates promptly.
        self.broad_frac = broad_frac
        self.bw_frac_z = bw_frac_z
        self.bw_frac_threshold = bw_frac_threshold
        self.broad_bw_score = 0.35
        # broad-bw must persist for this many consecutive steps before the
        # score injection applies (a single step of many devices above z is
        # normal; a sustained stretch is a genuine low-and-slow/volumetric).
        self.broad_bw_steps = broad_bw_steps


class TaimDetector:
    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self.baseline = TimeWindowBaseline(self.config.baseline)
        self.fusion = MultiSignalFusion(self.config.fusion)
        self.ladder = ResponseLadder(self.config.ladder)

    def _aggregate(self, g: pd.DataFrame) -> dict[str, float]:
        # Mean across devices (not sum): averaging reduces noise by ~sqrt(n)
        # while keeping the same scale as the per-device baselines, so the
        # sigma floor stays comparable and genuine broad changes register.
        return {
            "bandwidth_mbps": float(g["bandwidth_mbps"].mean()),
            "conn_rate_ps": float(g["conn_rate_ps"].mean()),
            "app_req_ps": float(g["app_req_ps"].mean()),
            "port_div": float(g["port_div"].mean()),
            "pkt_size_mean": float(g["pkt_size_mean"].mean()),
        }

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        slot_fn = self.config.slot_fn
        broad_streak = 0
        for ts, g in df.groupby("timestamp", sort=True):
            slot = slot_fn(int(g["weekday"].iloc[0]), int(g["hour"].iloc[0]))
            # aggregate-level scoring first (volumetric attacks hit this hard)
            za = self.baseline.step(AGG_DEVICE, slot, self._aggregate(g))
            agg_comp = self.fusion.fuse(za)

            # score every device first so we can judge whether the elevation
            # is broad (many devices) before deciding to apply the aggregate.
            dev_scores = []
            for _, row in g.iterrows():
                device = int(row["device_id"])
                values = {s: float(row[s]) for s in SIGNAL_LIST}
                z = self.baseline.step(device, slot, values)
                dev_comp = self.fusion.fuse(z)
                bw_z = z["bandwidth_mbps"]
                max_z_signal = max(z, key=lambda s: abs(z[s]))
                dev_scores.append(
                    (device, dev_comp, abs(z[max_z_signal]), max_z_signal, bw_z)
                )

            n_devices = len(dev_scores)
            n_suspicious = sum(1 for d in dev_scores if d[1].score > 0.0 or d[1].fired)
            n_bw_active = sum(1 for d in dev_scores if d[4] >= self.config.bw_frac_z)
            broad_bw_ok = n_bw_active / n_devices >= self.config.bw_frac_threshold
            broad = (n_suspicious / n_devices >= self.config.broad_frac) or broad_bw_ok
            agg_score = agg_comp.score if broad else 0.0
            broad_streak = broad_streak + 1 if broad_bw_ok else 0
            broad_bw_score = self.config.broad_bw_score if broad_streak >= self.config.broad_bw_steps else 0.0
            if broad:
                agg_max_signal = max(za, key=lambda s: abs(za[s]))
                agg_max_z = abs(za[agg_max_signal])
            else:
                agg_max_z, agg_max_signal = 0.0, None

            for device, dev_comp, max_z, max_z_signal, _bw in dev_scores:
                score = max(dev_comp.score, agg_score, broad_bw_score)
                fired = dev_comp.fired or (broad and agg_comp.fired) or broad_bw_ok
                if agg_max_z > max_z:
                    max_z, max_z_signal = agg_max_z, agg_max_signal
                stage, action = self.ladder.step(
                    device, score, fired, max_z=max_z, max_z_signal=max_z_signal
                )
                rows.append(
                    {
                        "timestamp": ts,
                        "device_id": device,
                        "score": score,
                        "n_elevated": max(dev_comp.n_elevated, agg_comp.n_elevated),
                        "fired": fired,
                        "stage": stage,
                        "action": action,
                        "flagged": stage >= 2,
                    }
                )
        out = pd.DataFrame(rows)
        return df.merge(out, on=["timestamp", "device_id"], how="left")