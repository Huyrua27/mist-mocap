# -*- coding: utf-8 -*-
"""ContinuSyncFormer — model đề xuất.  [STUB kiến trúc — Owner: P2 (WS2)]

Cải tiến so với InSynFormer (xem paper, mục "Đề Xuất Cải Tiến Kiến Trúc"):
  1. Continuous Temporal Encoding (CTE): RoPE cho trục thời gian liên tục   [task #12]
  2. Cross-View Attention + soft-alignment                                   [task #13]
  3. Hierarchical Regression Head: ước lượng trực tiếp ∆t ∈ R (không phân bin) [task #14]
  4. (tuỳ chọn) Occlusion-Aware attention mask                               [task #18]

Input:  chuỗi 2D keypoints 2 camera (dùng KeypointSequence trong core.types).
Output: ∆t (frame, thực).

torch là dependency RIÊNG của phần model — xem requirements-model.txt.
Phần benchmark (P1) KHÔNG cần torch để chạy.
"""
from __future__ import annotations
from ..core.interfaces import SyncMethod
from ..core.types import KeypointSequence, SyncResult

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
    nn = object  # để định nghĩa class không lỗi khi thiếu torch


class ContinuSyncFormer(nn.Module if _HAS_TORCH else object):
    """Encoder (temporal self-attn + RoPE mỗi cam) → Decoder (cross-view attn)
    → Hierarchical regression head (∆t ∈ R)."""

    def __init__(self, n_joints=17, d_model=256, n_heads=8, n_layers=4):
        if not _HAS_TORCH:
            raise ImportError("Cần torch cho model: pip install -r requirements-model.txt")
        super().__init__()
        self.d_model = d_model
        # TODO(P2 #11): input embedding (2*J -> d_model)
        # TODO(P2 #12): RoPE / Continuous Temporal Encoding
        # TODO(P2 #11): temporal self-attention encoder (n_layers)
        # TODO(P2 #13): cross-view attention decoder + soft-alignment
        # TODO(P2 #14): hierarchical regression head -> scalar ∆t
        raise NotImplementedError("TODO(P2): dựng các khối theo comment trên")

    def forward(self, kp_a, kp_b):
        """kp_a, kp_b: tensor (B, T, J, 2) -> ∆t (B,)."""
        raise NotImplementedError


class ContinuSyncMethod(SyncMethod):
    """Bọc model đã train thành SyncMethod để CẮM VÀO benchmark.eval (không sửa harness)."""
    name = "ContinuSyncFormer"

    def __init__(self, checkpoint: str | None = None):
        self.model = None
        self.checkpoint = checkpoint
        # TODO(P2 #16): load checkpoint, set eval()

    def predict(self, a: KeypointSequence, b: KeypointSequence) -> SyncResult:
        # TODO(P2): tiền xử lý -> forward -> SyncResult(dt_frames=...)
        raise NotImplementedError("TODO(P2): inference wrapper cho ContinuSyncFormer")
