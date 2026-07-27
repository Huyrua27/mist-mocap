# -*- coding: utf-8 -*-
"""Baseline 1: Cross-Correlation + nội suy parabol (sub-frame).

Trượt tín hiệu keypoint đa kênh tìm điểm tương quan cực đại, rồi nội suy parabol
quanh đỉnh để đạt độ phân giải dưới khung. Giới hạn: giả định tín hiệu đẳng cấu
giữa 2 view (sai khi hình chiếu khác pha) — chính là điểm model cần vượt.

Owner: P1 (WS1).
"""
from __future__ import annotations
import numpy as np
from ...core.interfaces import SyncMethod
from ...core.types import KeypointSequence, SyncResult


class CrossCorrelation(SyncMethod):
    name = "CC+parabol"

    def __init__(self, max_lag: int = 20):
        self.max_lag = max_lag

    @staticmethod
    def _sig(seq: KeypointSequence) -> np.ndarray:
        x = seq.xy.reshape(seq.T, -1)                    # (T, 2J)
        x = x - x.mean(0, keepdims=True)
        return x / (x.std(0, keepdims=True) + 1e-8)

    def predict(self, a: KeypointSequence, b: KeypointSequence) -> SyncResult:
        sa, sb = self._sig(a), self._sig(b)
        n = min(len(sa), len(sb))
        sa, sb = sa[:n], sb[:n]
        maxlag = int(min(self.max_lag, n // 3))
        lags = np.arange(-maxlag, maxlag + 1)
        c = np.empty(len(lags))
        for i, L in enumerate(lags):
            if L >= 0:
                X, Y = sa[L:], sb[:n - L]               # pair sa[k+L] với sb[k]
            else:
                X, Y = sa[:n + L], sb[-L:]
            c[i] = np.sum(X * Y) / len(X)
        k = int(np.argmax(c))
        delta = 0.0
        if 0 < k < len(c) - 1:                          # nội suy parabol -> sub-frame
            y0, y1, y2 = c[k - 1], c[k], c[k + 1]
            den = y0 - 2 * y1 + y2
            if abs(den) > 1e-12:
                delta = 0.5 * (y0 - y2) / den
        dt = float(lags[k] + delta)                     # đỉnh khi sa[k+dt]=sb[k] -> dt
        conf = float((c[k] - np.median(c)) / (np.std(c) + 1e-9))
        return SyncResult(dt_frames=dt, confidence=max(0.0, min(1.0, conf / 10)))
