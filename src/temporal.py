"""Temporal / ML scoring experiment.

Two optional scorers that plug into FastTaimDetector and score each device
from a *window* of recent z-scores instead of a single step:

  WindowedMeanScorer   (non-ML) - rolling mean of bandwidth z per device,
                                  calibrated adaptively on warm-up data.
  PCAWindowScorer      (ML)     - PCA autoencoder over flattened z-windows;
                                  reconstruction error is the anomaly score.

Both are unsupervised and calibrated on the first `train_days` of each
environment (no look-ahead). The experiment A/B/C in ml_experiment.py decides
whether the ML version beats both the current detector (A) and plain temporal
averaging (B).
"""
from __future__ import annotations

import numpy as np

from sklearn.decomposition import PCA

from src.baseline import hour_slot

BW_IDX = 0  # bandwidth_mbps is index 0 in SIGNAL_LIST


class _BaseWindowScorer:
    def __init__(self, window_steps: int, train_days: int, step_min: int,
                 streak_required: int = 3) -> None:
        self.window_steps = window_steps
        self.train_steps = train_days * (24 * 60 // step_min)
        self.buffer = None        # (n_signals, window_steps, n_devices)
        self.fitted = False
        self.n_train = 0
        self.train_windows = []   # for offline fitting
        self.streak_required = streak_required
        self.streak = None        # per-device consecutive-elevated counter

    def _push(self, z: np.ndarray) -> np.ndarray | None:
        """Slide z (signal x device) into the rolling window; return the
        full window once filled (signal x window x device)."""
        if self.buffer is None:
            self.buffer = np.zeros((z.shape[0], self.window_steps, z.shape[1]))
            self.buffer[:] = np.nan
            self.streak = np.zeros(z.shape[1], dtype=np.int64)
        self.buffer[:, :-1, :] = self.buffer[:, 1:, :]
        self.buffer[:, -1, :] = z
        if np.isnan(self.buffer).any():
            return None
        return self.buffer

    def _apply_streak(self, raw: np.ndarray) -> np.ndarray:
        """Only report the temporal score after it stays elevated for
        `streak_required` consecutive steps (single-step noise is ignored)."""
        elev = raw > 0.0
        self.streak = np.where(elev, self.streak + 1, 0)
        return np.where(self.streak >= self.streak_required, raw, 0.0)


class WindowedMeanScorer(_BaseWindowScorer):
    """B: rolling mean of each device's bandwidth z over the window.
    Calibrated so that 'normal' window means score ~0."""

    def __init__(self, window_steps: int, train_days: int, step_min: int) -> None:
        super().__init__(window_steps, train_days, step_min, streak_required=4)
        self.calib = 0.0
        self.span = 1.0

    def step(self, z: np.ndarray, slot: int) -> np.ndarray | None:
        win = self._push(z)
        if win is None:
            return None
        # device window-mean of bandwidth z
        m = np.nanmean(win[BW_IDX], axis=0)  # (n_devices,)
        if not self.fitted:
            self.n_train += 1
            if self.n_train <= self.train_steps:
                self.train_windows.append(m)
                return None
            # fit calibration from warm-up window means (very conservative)
            arr = np.concatenate(self.train_windows)
            self.calib = float(np.percentile(arr, 99.0))
            self.span = max(float(np.percentile(arr, 99.9) - self.calib), 0.3)
            self.fitted = True
        raw = np.clip((m - self.calib) / self.span, 0.0, 1.0)
        return self._apply_streak(raw)


class PCAWindowScorer(_BaseWindowScorer):
    """C: PCA autoencoder over flattened windows of all 5 z-signals.
    Reconstruction MSE is mapped to [0,1] using warm-up percentiles."""

    def __init__(self, window_steps: int, train_days: int, step_min: int,
                 n_components: int = 12) -> None:
        super().__init__(window_steps, train_days, step_min, streak_required=4)
        self.pca = None
        self.p90 = 0.0
        self.p99 = 0.0
        self.n_components = n_components

    def step(self, z: np.ndarray, slot: int) -> np.ndarray | None:
        win = self._push(z)
        if win is None:
            return None
        # flatten window -> (n_devices, signals*window)
        feat = win.transpose(2, 0, 1).reshape(z.shape[1], -1)
        if not self.fitted:
            self.n_train += 1
            if self.n_train <= self.train_steps:
                self.train_windows.append(feat)
                return None
            X = np.vstack(self.train_windows)
            self.pca = PCA(n_components=self.n_components, whiten=True)
            self.pca.fit(X)
            train_rec = self._rec_err(X)
            self.p90 = float(np.percentile(train_rec, 99.0))
            self.p99 = max(float(np.percentile(train_rec, 99.9)), self.p90 + 1e-6)
            self.fitted = True
            self.train_windows = None
        err = self._rec_err(feat)
        raw = np.clip((err - self.p90) / (self.p99 - self.p90), 0.0, 1.0)
        return self._apply_streak(raw)

    def _rec_err(self, X: np.ndarray) -> np.ndarray:
        rec = self.pca.inverse_transform(self.pca.transform(X))
        return ((X - rec) ** 2).mean(axis=1)
