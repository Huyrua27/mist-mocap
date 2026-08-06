# -*- coding: utf-8 -*-
"""Time-varying (drift) desynchronization and sliding-window affine recovery.

Real multi-device rigs do not share a clock: camera B's frame ``t`` captures the
scene at continuous reference time ``alpha*t + beta`` (alpha != 1 => the offset
drifts across the take). A single-lag method (cross-correlation) cannot represent
this; the offset it applies is right at one instant and wrong everywhere else.

Recovering drift requires estimating a *local* offset in short sliding windows and
fitting a line through them. Short windows are exactly where classical CC becomes
unreliable and a learned estimator wins -- so drift is the regime where the learned
model has a structural advantage.  Task #24 / B3 (for 3DV).
"""
from __future__ import annotations

import numpy as np

from ..core.types import KeypointSequence
from .interpolation import cubic_sample


def drift_offset(index: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """True extra time of B relative to A at each index: (alpha-1)*t + beta."""
    return (alpha - 1.0) * np.asarray(index, dtype=np.float64) + beta


def warp_stream(clean_xy: np.ndarray, alpha: float, beta: float,
                length: int, start: int = 0) -> np.ndarray:
    """Camera-B observation stream under drift: obs[t] = clean(alpha*(start+t)+beta).

    ``clean_xy`` is the full clean trajectory; sampling is cubic with edge
    extrapolation so the caller only needs a modest margin.
    """
    idx = alpha * (start + np.arange(length, dtype=np.float64)) + beta
    return cubic_sample(clean_xy, idx, extrapolate=True)


def theilsen_line(centers: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Robust slope/intercept (median of pairwise slopes) -- rejects the
    catastrophic single-window failures that CC produces on short windows."""
    centers = np.asarray(centers, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(centers) & np.isfinite(values)
    centers, values = centers[ok], values[ok]
    if len(centers) < 2:
        return 0.0, float(values.mean()) if len(values) else 0.0
    slopes = []
    for i in range(len(centers)):
        dx = centers[i + 1:] - centers[i]
        dy = values[i + 1:] - values[i]
        m = dy[dx != 0] / dx[dx != 0]
        slopes.append(m)
    slope = float(np.median(np.concatenate(slopes)))
    intercept = float(np.median(values - slope * centers))
    return slope, intercept


def sliding_offsets(predict, a_stream: np.ndarray, b_stream: np.ndarray,
                    fps: float, window: int, stride: int):
    """Local offset per window via ``predict(a_seq, b_seq) -> dt_frames``.

    Returns (centers, offsets) over windows where prediction succeeds.
    """
    n = min(len(a_stream), len(b_stream))
    centers, offsets = [], []
    for s in range(0, n - window + 1, stride):
        a = KeypointSequence(a_stream[s:s + window].astype(np.float64), fps, name="a")
        b = KeypointSequence(b_stream[s:s + window].astype(np.float64), fps, name="b")
        try:
            dt = float(predict(a, b))
        except Exception:
            continue
        centers.append(s + window / 2.0)
        offsets.append(dt)
    return np.asarray(centers), np.asarray(offsets)


def _hampel(values: np.ndarray, k: int = 2, n_sigma: float = 2.5) -> np.ndarray:
    """Replace outliers by the local median (rejects CC's catastrophic windows)."""
    v = np.asarray(values, dtype=np.float64).copy()
    out = v.copy()
    for i in range(len(v)):
        lo, hi = max(0, i - k), min(len(v), i + k + 1)
        window = v[lo:hi]
        med = np.median(window)
        mad = 1.4826 * np.median(np.abs(window - med)) + 1e-9
        if abs(v[i] - med) > n_sigma * mad:
            out[i] = med
    return out


def fit_drift(predict, a_stream, b_stream, fps, window, stride, mode="line"):
    """Sliding-window local offsets -> per-index offset estimate.

    mode="line": robust affine fit (Theil-Sen) -- optimal when drift is linear.
    mode="curve": outlier-reject then interpolate the local offsets -- tracks
    *non-linear* drift that a global line cannot represent. Only reliable when the
    local estimates are trustworthy on short windows (i.e. with the learned model).
    """
    centers, offsets = sliding_offsets(predict, a_stream, b_stream, fps, window, stride)
    n = min(len(a_stream), len(b_stream))
    idx = np.arange(n, dtype=np.float64)
    if len(centers) < 2:
        return np.full(n, float(np.median(offsets)) if len(offsets) else 0.0), centers, offsets
    if mode == "curve":
        cleaned = _hampel(offsets)
        return np.interp(idx, centers, cleaned), centers, offsets
    slope, intercept = theilsen_line(centers, offsets)
    return slope * idx + intercept, centers, offsets


def warp_offsets(clean_xy: np.ndarray, delta: np.ndarray, base: float) -> np.ndarray:
    """General (possibly non-linear) drift: obs[t] = clean(base + t + delta[t])."""
    idx = base + np.arange(len(delta), dtype=np.float64) + np.asarray(delta, dtype=np.float64)
    return cubic_sample(clean_xy, idx, extrapolate=True)


def sine_offset(index: np.ndarray, amp: float, period: float, beta: float) -> np.ndarray:
    """Non-linear (oscillating) offset: a global line cannot fit this."""
    return amp * np.sin(2.0 * np.pi * np.asarray(index, dtype=np.float64) / period) + beta
