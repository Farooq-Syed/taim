"""Phase 2: Time-window baseline engine.

Maintains a per-(device, time-slot, signal) statistical model of "normal"
behaviour using an outlier-resistant EWMA. Scoring a new observation yields
a z-score against its own slot's model, so e.g. 5 Mbps at 3am is treated very
differently from 5 Mbps at 2pm.

Critical design choice (look-ahead avoidance): the caller must follow the
online convention `score THEN update`, i.e. never let the observation being
scored feed the model it is scored against.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass
class BaselineConfig:
    alpha: float = 0.2               # EWMA learning rate
    min_samples: int = 8             # warm-up before z-scores are trusted
    outlier_k: float = 3.5           # |z| beyond which an observation is an
                                     # outlier and is excluded from the model
    sigma_floor_rel: float = 0.5     # default relative floor (rate metrics whose
                                     # noise is ~multiplicative)
    sigma_floor_abs: float = 1e-6    # default absolute floor
    signal_floor_rel: dict = field(default_factory=dict)   # per-signal override
    signal_floor_abs: dict = field(default_factory=dict)   # per-signal override
    z_max: float = 10.0              # clip z-scores to [-z_max, z_max]


@dataclass
class SignalModel:
    mu: float = 0.0
    var: float = 1.0
    n: int = 0

    def sigma_floor(self, config: BaselineConfig, signal: str) -> float:
        rel = config.signal_floor_rel.get(signal, config.sigma_floor_rel)
        abs_ = config.signal_floor_abs.get(signal, config.sigma_floor_abs)
        return max(abs_, rel * abs(self.mu))


def make_slot(weekday: int, hour: float) -> tuple:
    """Coarse time slot: (is_weekend, hour). Refines per-slot baselines without
    over-fragmenting the (thin) history."""
    return (1 if weekday >= 5 else 0, int(hour))


def hour_slot(weekday: int, hour: float) -> tuple:
    """Hour-of-day slot only (weekend variance is absorbed in sigma). Needs far
    fewer warm-up samples per slot than (is_weekend, hour), which matters for
    short walk-forward training windows."""
    return (int(hour),)


class TimeWindowBaseline:
    def __init__(self, config: BaselineConfig | None = None) -> None:
        self.config = config or BaselineConfig()
        self.models: dict[tuple, SignalModel] = {}

    # ---- internal helpers -------------------------------------------------
    def _key(self, device_id, slot, signal):
        if not isinstance(slot, tuple):
            slot = (slot,)
        return (device_id, *slot, signal)

    def _get(self, device_id, slot, signal) -> SignalModel:
        k = self._key(device_id, slot, signal)
        return self.models.setdefault(k, SignalModel())

    def is_warm(self, device_id, slot, signal) -> bool:
        return self._get(device_id, slot, signal).n >= self.config.min_samples

    # ---- scoring ----------------------------------------------------------
    def score(self, device_id, slot, values: Mapping[str, float]) -> dict[str, float]:
        """Return per-signal z-scores (0.0 until the slot is warm)."""
        cfg = self.config
        out = {}
        for signal, x in values.items():
            if not self.is_warm(device_id, slot, signal):
                out[signal] = 0.0
                continue
            m = self._get(device_id, slot, signal)
            sigma = max(float(np.sqrt(m.var)), m.sigma_floor(cfg, signal))
            z = (float(x) - m.mu) / sigma
            out[signal] = float(np.clip(z, -cfg.z_max, cfg.z_max))
        return out

    # ---- update -----------------------------------------------------------
    def update(self, device_id, slot, values: Mapping[str, float]) -> None:
        cfg = self.config
        for signal, x in values.items():
            m = self._get(device_id, slot, signal)
            if m.n == 0:
                m.mu = float(x)
                m.var = max((0.2 * abs(float(x)) + 1e-9) ** 2, cfg.sigma_floor_abs ** 2)
                m.n += 1
                continue
            resid = float(x) - m.mu
            sigma = max(float(np.sqrt(m.var)), m.sigma_floor(cfg, signal))
            z = abs(resid) / sigma
            if z > cfg.outlier_k:
                # outlier: score it, but NEVER let it move the model
                # (otherwise a sustained attack inflates mu/var and the
                # baseline silently absorbs the attack).
                m.n += 1
                continue
            m.mu += cfg.alpha * resid
            m.var += cfg.alpha * (resid ** 2 - m.var)
            m.n += 1

    # ---- online step: score first, then update -----------------------------
    def step(self, device_id, slot, values: Mapping[str, float]) -> dict[str, float]:
        zs = self.score(device_id, slot, values)
        self.update(device_id, slot, values)
        return zs

    # ---- stats --------------------------------------------------------------
    def summary(self) -> dict[str, list]:
        rows = {"device": [], "signal": [], "slot": [], "mu": [], "sigma": [], "n": []}
        cfg = self.config
        for key, m in self.models.items():
            device, slot, signal = key[0], key[1:-1], key[-1]
            rows["device"].append(device)
            rows["signal"].append(signal)
            rows["slot"].append(slot)
            rows["mu"].append(m.mu)
            rows["sigma"].append(max(float(np.sqrt(m.var)), m.sigma_floor(cfg, signal)))
            rows["n"].append(m.n)
        return rows

    def warm_fraction(self, required: int | None = None) -> float:
        if not self.models:
            return 0.0
        req = required if required is not None else self.config.min_samples
        warm = sum(1 for m in self.models.values() if m.n >= req)
        return warm / len(self.models)
