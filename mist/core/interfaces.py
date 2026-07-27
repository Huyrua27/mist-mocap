# -*- coding: utf-8 -*-
"""Interface CHUNG cho mọi phương pháp sync.

Baseline (P1) và model ContinuSyncFormer (P2) đều kế thừa SyncMethod và cài `predict`.
Nhờ vậy benchmark harness chạy MỌI method bằng đúng một vòng lặp.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from .types import KeypointSequence, SyncResult


class SyncMethod(ABC):
    """Ước lượng độ lệch thời gian giữa 2 chuỗi keypoint cùng một cảnh, 2 góc nhìn."""
    name: str = "base"

    @abstractmethod
    def predict(self, a: KeypointSequence, b: KeypointSequence) -> SyncResult:
        """Trả SyncResult với dt_frames = offset của `b` so với `a` (đơn vị frame).

        Quy ước dấu: dt_frames > 0 nghĩa là `b` bị TRỄ so với `a` dt_frames khung.
        Đây đúng nghĩa nhãn mà benchmark.desync.inject_offset() tạo ra, nên method
        đúng sẽ khôi phục lại chính con số đã tiêm.
        """
        raise NotImplementedError
