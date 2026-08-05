"""Cross-correlation baseline with parabolic sub-frame peak refinement."""
from __future__ import annotations

import numpy as np

from ...core.interfaces import SyncMethod
from ...core.types import KeypointSequence, SyncResult


def motion_features(seq: KeypointSequence) -> np.ndarray:
    """View-tolerant per-joint speed features with missing-value interpolation."""
    xy = np.asarray(seq.xy, dtype=np.float64).copy()
    time = np.arange(seq.T)
    for joint in range(seq.J):
        for coordinate in range(2):
            values = xy[:, joint, coordinate]
            valid = np.isfinite(values)
            if valid.sum() < 2:
                values[:] = 0.0
            elif not valid.all():
                values[~valid] = np.interp(time[~valid], time[valid], values[valid])
    velocity = np.gradient(xy, axis=0)
    features = np.linalg.norm(velocity, axis=-1)
    features -= features.mean(axis=0, keepdims=True)
    scale = features.std(axis=0, keepdims=True)
    return features / np.where(scale > 1e-8, scale, 1.0)


def lag_scores(
    features_a: np.ndarray,
    features_b: np.ndarray,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(features_a), len(features_b))
    if n < 4:
        raise ValueError("cross-correlation requires at least four frames")
    a, b = features_a[:n], features_b[:n]
    if float(np.std(a)) < 1e-10 or float(np.std(b)) < 1e-10:
        raise ValueError("insufficient motion for cross-correlation")
    limit = min(int(max_lag), n - 3)
    lags = np.arange(-limit, limit + 1, dtype=np.float64)
    scores = np.empty(len(lags), dtype=np.float64)
    for index, lag_value in enumerate(lags.astype(int)):
        if lag_value >= 0:
            left, right = a[lag_value:], b[: n - lag_value]
        else:
            left, right = a[: n + lag_value], b[-lag_value:]
        scores[index] = float(np.mean(left * right))
    return lags, scores


def refined_peak(lags: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    index = int(np.argmax(scores))
    delta = 0.0
    if 0 < index < len(scores) - 1:
        y0, y1, y2 = scores[index - 1 : index + 2]
        denominator = y0 - 2 * y1 + y2
        if abs(denominator) > 1e-12:
            delta = float(np.clip(0.5 * (y0 - y2) / denominator, -1.0, 1.0))
    lag = float(lags[index] + delta)
    baseline = float(np.median(scores))
    spread = float(np.std(scores))
    confidence = 0.0 if spread < 1e-12 else (float(scores[index]) - baseline) / spread
    return lag, float(np.clip(confidence / 6.0, 0.0, 1.0))


class CrossCorrelation(SyncMethod):
    name = "CC+parabol"

    def __init__(self, max_lag: int = 20):
        self.max_lag = int(max_lag)

    def predict(self, a: KeypointSequence, b: KeypointSequence) -> SyncResult:
        lags, scores = lag_scores(motion_features(a), motion_features(b), self.max_lag)
        lag, confidence = refined_peak(lags, scores)
        return SyncResult(dt_frames=lag, confidence=confidence)
