# -*- coding: utf-8 -*-
"""Baseline sàn: ZeroOffset (đoán ∆t=0). Dùng làm mốc dưới — mọi method phải hơn cái này."""
from __future__ import annotations
from ...core.interfaces import SyncMethod
from ...core.types import KeypointSequence, SyncResult


class ZeroOffset(SyncMethod):
    name = "Zero(sàn)"

    def predict(self, a: KeypointSequence, b: KeypointSequence) -> SyncResult:
        return SyncResult(dt_frames=0.0, confidence=0.0)
