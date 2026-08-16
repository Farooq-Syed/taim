"""Vectorized TAIM detector.

Functional equivalent of TaimDetector (same causal score-then-update semantics,
same fusion / broad-gate / ladder logic) but with the per-observation Python
overhead removed:

  * models live in numpy arrays indexed [slot, signal, device]
  * every timestep is a handful of numpy ops across all devices + signals
  * the ladder is a per-device state machine stored as arrays

Produces byte-identical scores/stages to TaimDetector (verified by tests).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.baseline import hour_slot
from src.detector import DetectorConfig, SIGNAL_LIST

DOWN_INDEX = {s: i for i, s in enumerate(SIGNAL_LIST)}


def _signal_index(name: str) -> int:
    return SIGNAL_LIST.index(name)


class FastTaimDetector:
    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        c = self.config
        self.n_signals = len(SIGNAL_LIST)
        self.n_slots = 24  # hour-of-day slots
        self.bw_idx = _signal_index("bandwidth_mbps")
        self.mu = None
        self.var = None
        self.nobs = None
        self.down_mask = np.zeros(self.n_signals, dtype=bool)
        for s in c.fusion.down_signals:
            self.down_mask[_signal_index(s)] = True
        # per-signal elevation/sigma config
        self.z_elev = c.fusion.z_elevation
        self.z_sat = c.fusion.z_saturate
        self.min_signals = c.fusion.min_signals
        sig_weights = c.fusion.signal_weights
        self.weights = np.ones(self.n_signals)
        for s, w in sig_weights.items():
            self.weights[_signal_index(s)] = w
        self.sfr = {_signal_index(s): v for s, v in c.baseline.signal_floor_rel.items()}
        self.sfa = {_signal_index(s): v for s, v in c.baseline.signal_floor_abs.items()}
        self.rel_floor = c.baseline.sigma_floor_rel
        self.abs_floor = c.baseline.sigma_floor_abs
        self.alpha = c.baseline.alpha
        self.min_samples = c.baseline.min_samples
        self.outlier_k = c.baseline.outlier_k
        self.z_max = c.baseline.z_max
        # ladder
        lc = c.ladder
        self.score_high = lc.score_high
        self.score_low = lc.score_low
        self.esc_steps = lc.escalate_steps
        self.deesc_steps = lc.deescalate_steps
        self.max_stage = lc.max_stage
        self.sustain_z = lc.sustain_z
        self.sustain_steps = lc.sustain_steps
        self.sustain_same_signal = lc.sustain_same_signal

    def _reset_models(self, n_devices: int) -> None:
        self.mu = np.zeros((self.n_slots, self.n_signals, n_devices))
        self.var = np.zeros((self.n_slots, self.n_signals, n_devices))
        self.nobs = np.zeros((self.n_slots, self.n_signals, n_devices), dtype=np.int64)
        self.agg_mu = np.zeros((self.n_slots, self.n_signals))
        self.agg_var = np.zeros((self.n_slots, self.n_signals))
        self.agg_n = np.zeros((self.n_slots, self.n_signals), dtype=np.int64)

    def _sigma(self, slot: int) -> np.ndarray:
        mu_s = self.mu[slot]
        var_s = self.var[slot]
        sigma = np.sqrt(np.maximum(var_s, 0.0))
        for si in range(self.n_signals):
            rel = self.sfr.get(si, self.rel_floor)
            abs_ = self.sfa.get(si, self.abs_floor)
            floor = np.maximum(abs_, rel * np.abs(mu_s[si]))
            sigma[si] = np.maximum(sigma[si], floor)
        return sigma

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        devs = np.sort(df["device_id"].unique())
        n_dev = len(devs)
        dev_lookup = {d: i for i, d in enumerate(devs)}
        self._reset_models(n_dev)

        # pre-allocate per-timestep arrays
        X = np.empty((self.n_signals, n_dev))
        stage = np.zeros(n_dev, dtype=np.int64)
        high = np.zeros(n_dev, dtype=np.int64)
        low = np.zeros(n_dev, dtype=np.int64)
        sust_streak = np.zeros(n_dev, dtype=np.int64)
        sust_sig = np.full(n_dev, -1, dtype=np.int64)
        flagged = np.zeros(n_dev, dtype=bool)

        records = []
        broad_streak = 0
        dt = np.dtype([("ts", object), ("dev", np.int64), ("score", np.float64),
                       ("n_elev", np.int64), ("fired", bool), ("stage", np.int64),
                       ("action", object), ("flag", bool)])

        for ts, g in df.groupby("timestamp", sort=True):
            slot = hour_slot(int(g["weekday"].iloc[0]), int(g["hour"].iloc[0]))[0]
            # build X (signal x device)
            idx = g["device_id"].map(dev_lookup).to_numpy()
            for si, sig in enumerate(SIGNAL_LIST):
                X[si, idx] = g[sig].to_numpy()

            # ---- aggregate baseline ----
            agg_vals = np.empty(self.n_signals)
            for si, sig in enumerate(SIGNAL_LIST):
                agg_vals[si] = X[si, :].mean()
            agg_score, agg_fired, agg_max_z, agg_max_sig = self._score_update_agg(
                slot, agg_vals
            )

            # ---- per-device baseline score-then-update (vectorized) ----
            mu_s = self.mu[slot]
            var_s = self.var[slot]
            n_s = self.nobs[slot]
            sigma = self._sigma(slot)

            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.where(
                    n_s >= self.min_samples,
                    (X - mu_s) / np.maximum(sigma, 1e-12),
                    0.0,
                )
            z = np.clip(z, -self.z_max, self.z_max)

            # ---- fusion (vectorized across devices) ----
            signed = np.where(self.down_mask[:, None], -z, z)
            elev = np.clip((signed - self.z_elev) / (self.z_sat - self.z_elev), 0.0, 1.0)
            elev = np.where(signed <= self.z_elev, 0.0, elev)
            n_elev = (elev > 0.0).sum(axis=0)
            fired = n_elev >= self.min_signals
            denom = (elev > 0.0) * self.weights[:, None]
            numer = (elev * self.weights[:, None]).sum(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                raw = np.where(denom.sum(axis=0) > 0,
                               numer / np.maximum(denom.sum(axis=0), 1e-12), 0.0)
            dev_score = np.clip(raw, 0.0, 1.0)
            dev_score = np.where(fired, dev_score, 0.0)  # suppressed when not fired

            # ---- broad gate ----
            n_devices = n_dev
            n_susp = int(np.sum((dev_score > 0.0) | fired))
            n_bw = int(np.sum(z[self.bw_idx, :] >= c.bw_frac_z))
            broad_bw = n_bw / n_devices >= c.bw_frac_threshold
            broad = (n_susp / n_devices >= c.broad_frac) or broad_bw
            agg_eff = agg_score if broad else 0.0
            broad_streak = broad_streak + 1 if broad_bw else 0
            inj = c.broad_bw_score if broad_streak >= c.broad_bw_steps else 0.0
            if broad:
                a_z, a_sig = agg_max_z, agg_max_sig
            else:
                a_z, a_sig = 0.0, -1
            fired_eff = fired | (broad and agg_fired) | broad_bw

            # ---- final score per device ----
            score = np.maximum(np.maximum(dev_score, agg_eff), inj)
            max_z = np.abs(z).max(axis=0)
            max_z_sig = np.argmax(np.abs(z), axis=0)
            if a_z > 0.0:
                over = a_z > max_z
                max_z = np.where(over, a_z, max_z)
                max_z_sig = np.where(over, a_sig, max_z_sig)

            # ---- ladder (vectorized state machine) ----
            inc_high = score > self.score_high
            dec_low = score < self.score_low
            high = np.where(inc_high, high + 1, 0)
            low = np.where(dec_low, low + 1, 0)
            esc = (high >= self.esc_steps) & (stage < self.max_stage)
            stage = np.where(esc, stage + 1, stage)
            high = np.where(esc, 0, high)
            deesc = (low >= self.deesc_steps) & (stage > 0)
            stage = np.where(deesc, stage - 1, stage)
            low = np.where(deesc, 0, low)

            # sustain path
            sust_on = max_z >= self.sustain_z
            if self.sustain_same_signal:
                same = max_z_sig == sust_sig
                sust_streak = np.where(
                    sust_on, np.where(same, sust_streak + 1, 1), 0
                )
                sust_sig = np.where(sust_on, max_z_sig, -1)
            else:
                sust_streak = np.where(sust_on, sust_streak + 1, 0)
                sust_sig = np.where(sust_on, max_z_sig, -1)
            sesc = (sust_streak >= self.sustain_steps) & (stage < self.max_stage)
            stage = np.where(sesc, stage + 1, stage)
            sust_streak = np.where(sesc, 0, sust_streak)

            flagged = stage >= 2

            # ---- baseline update (skip outliers) ----
            upd_mask = (np.abs(z) <= self.outlier_k) & (self.nobs[slot] > 0)
            resid = X - mu_s
            new_mu = mu_s + self.alpha * resid
            new_var = var_s + self.alpha * (resid ** 2 - var_s)
            self.mu[slot] = np.where(upd_mask, new_mu, mu_s)
            self.var[slot] = np.where(upd_mask, new_var, var_s)
            # init for nobs == 0
            init_mask = self.nobs[slot] == 0
            self.mu[slot] = np.where(init_mask, X, self.mu[slot])
            self.var[slot] = np.where(
                init_mask,
                np.maximum((0.2 * np.abs(X) + 1e-9) ** 2, self.abs_floor ** 2),
                self.var[slot],
            )
            self.nobs[slot] += 1

            # aggregate baseline update
            self._update_agg(slot, agg_vals)

            actions = np.array(["allow", "watch", "soft_cap_70pct", "hard_cap_30pct",
                                "drop_deauth"])
            for j in range(n_dev):
                records.append((ts, int(devs[j]), float(score[j]), int(n_elev[j]),
                                bool(fired_eff[j]), int(stage[j]), actions[int(stage[j])],
                                bool(flagged[j])))

        out = pd.DataFrame(records, columns=["timestamp", "device_id", "score", "n_elevated",
                                             "fired", "stage", "action", "flagged"])
        return df.merge(out, on=["timestamp", "device_id"], how="left")

    # ---- aggregate model helpers ----
    def _score_update_agg(self, slot: int, agg_vals: np.ndarray):
        mu = self.agg_mu[slot]
        var = self.agg_var[slot]
        n = self.agg_n[slot]
        sigma = np.sqrt(np.maximum(var, 0.0))
        for si in range(self.n_signals):
            rel = self.sfr.get(si, self.rel_floor)
            abs_ = self.sfa.get(si, self.abs_floor)
            floor = np.maximum(abs_, rel * np.abs(mu[si]))
            sigma[si] = np.maximum(sigma[si], floor)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(n >= self.min_samples, (agg_vals - mu) / np.maximum(sigma, 1e-12), 0.0)
        z = np.clip(z, -self.z_max, self.z_max)
        signed = np.where(self.down_mask, -z, z)
        elev = np.clip((signed - self.z_elev) / (self.z_sat - self.z_elev), 0.0, 1.0)
        elev = np.where(signed <= self.z_elev, 0.0, elev)
        n_elev = int((elev > 0.0).sum())
        fired = n_elev >= self.min_signals
        denom = np.sum((elev > 0.0) * self.weights)
        numer = np.sum(elev * self.weights)
        score = min(numer / denom, 1.0) if (denom > 0 and fired) else 0.0
        msi = int(np.argmax(np.abs(z))) if n_elev else -1
        a_z = float(np.abs(z).max()) if n_elev else 0.0
        # update (aggregate absorbs like reference: outlier-excluded)
        upd = (np.abs(z) <= self.outlier_k) & (n > 0)
        resid = agg_vals - mu
        self.agg_mu[slot] = np.where(upd, mu + self.alpha * resid, mu)
        self.agg_var[slot] = np.where(upd, var + self.alpha * (resid ** 2 - var), var)
        init = n == 0
        self.agg_mu[slot] = np.where(init, agg_vals, self.agg_mu[slot])
        self.agg_var[slot] = np.where(
            init, np.maximum((0.2 * np.abs(agg_vals) + 1e-9) ** 2, self.abs_floor ** 2),
            self.agg_var[slot])
        self.agg_n[slot] += 1
        return score, fired, a_z, msi

    def _update_agg(self, slot: int, agg_vals: np.ndarray) -> None:
        # scoring already updated; nothing else needed (update happens inside _score_update_agg)
        pass