# -*- coding: utf-8 -*-
"""Baseline 2: Dynamic Time Warping (+ oversampling cho sub-frame).  [STUB — P1 cài]

Kế hoạch:
  - Xây ma trận chi phí giữa 2 chuỗi tín hiệu keypoint, quy hoạch động tìm đường warp.
  - DTW thô chỉ ánh xạ chỉ số NGUYÊN → oversample (nội suy ×k) trước khi warp để đạt
    độ phân giải phân số; ước lượng ∆t = độ trễ trung vị của đường warp.
  - Giới hạn cần nêu trong paper: O(N^2), nhiễu nội suy, sập khi rớt khung hàng loạt.

TODO(P1): cài predict(). Có thể dùng librosa.sequence.dtw hoặc tự viết DP.
Tạm thời raise NotImplementedError → harness tự bỏ qua (điền 'nan').
"""
from __future__ import annotations
from ...core.interfaces import SyncMethod
from ...core.types import KeypointSequence, SyncResult


class DTW(SyncMethod):
    name = "DTW"

    def predict(self, a: KeypointSequence, b: KeypointSequence) -> SyncResult:
        raise NotImplementedError("TODO(P1): cài DTW + oversample cho sub-frame")
