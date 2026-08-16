"""Phase 3: Multi-signal scoring fusion.

Converts per-signal z-scores from the baseline engine into a composite
suspicion score in [0, 1]. Two mechanisms cut false positives:

  * direction awareness  - some signals are anomalous on the way DOWN
    (e.g. packet size shrinks during a SYN flood)
  * fusion rule          - at least `min_signals` signals must be elevated
    before the composite is non-zero, so a lone spurious spike cannot
    trigger mitigation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass
class CompositeScore:
    score: float          # 0..1 composite suspicion
    n_elevated: int       # how many signals passed their elevation threshold
    elevations: dict[str, float]  # per-signal elevation in [0, 1]
    fired: bool           # n_elevated >= min_signals


@dataclass
class FusionConfig:
    z_elevation: float = 2.0      # |z| needed for a signal to "count"
    z_saturate: float = 8.0       # |z| at which elevation saturates to 1.0
    min_signals: int = 2          # minimum elevated signals to fire
    down_signals: tuple = ("pkt_size_mean",)  # anomalous when LOW
    signal_weights: dict = field(default_factory=dict)  # optional per-signal weights


class MultiSignalFusion:
    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()

    def z_to_elevation(self, z: float, signal: str) -> float:
        cfg = self.config
        signed = -z if signal in cfg.down_signals else z
        if signed <= cfg.z_elevation:
            return 0.0
        return float(
            np.clip(
                (signed - cfg.z_elevation) / (cfg.z_saturate - cfg.z_elevation),
                0.0,
                1.0,
            )
        )

    def fuse(self, zscores: Mapping[str, float]) -> CompositeScore:
        cfg = self.config
        elevations = {
            sig: self.z_to_elevation(z, sig) for sig, z in zscores.items()
        }
        n_elevated = int(sum(1 for e in elevations.values() if e > 0.0))
        if n_elevated < cfg.min_signals:
            return CompositeScore(0.0, n_elevated, elevations, fired=False)

        # weighted average of all elevated signals (weights default to equal)
        elevated = {s: e for s, e in elevations.items() if e > 0.0}
        weights = []
        for s in elevated:
            w = cfg.signal_weights.get(s, 1.0)
            if s in cfg.down_signals:
                w = cfg.signal_weights.get(f"down:{s}", w)
            weights.append(max(w, 1e-9))
        score = float(
            np.sum([e * w for e, w in zip(elevated.values(), weights)])
            / np.sum(weights)
        )
        return CompositeScore(min(score, 1.0), n_elevated, elevations, fired=True)
