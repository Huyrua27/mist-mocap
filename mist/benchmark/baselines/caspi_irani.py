# -*- coding: utf-8 -*-
"""Baseline 3: Caspi–Irani — căn chỉnh không-thời gian bằng affine t'=αt+β. [STUB — P1]

Kế hoạch:
  - Thiết lập ràng buộc trên "quỹ đạo đặc trưng" xuyên thời gian giữa 2 view.
  - Ước lượng đồng thời α (tỉ lệ frame-rate/clock drift) và β (offset) bằng bình phương
    tối thiểu trên các điểm tương ứng; β đánh giá trong miền SỐ THỰC (sub-frame).
  - Giới hạn: phụ thuộc chất lượng đặc trưng, giả định homography 2D, nhạy clock drift.

TODO(P1): cài predict() trả SyncResult(dt_frames=β).  (α báo trong meta nếu cần.)
"""
from __future__ import annotations
from ...core.interfaces import SyncMethod
from ...core.types import KeypointSequence, SyncResult


class CaspiIrani(SyncMethod):
    name = "Caspi-Irani"

    def predict(self, a: KeypointSequence, b: KeypointSequence) -> SyncResult:
        raise NotImplementedError("TODO(P1): cài ước lượng affine t'=αt+β")
