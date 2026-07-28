# -*- coding: utf-8 -*-
"""Hierarchical loss: coarse cross-entropy (integer bin) + fine residual + direct regression.

Balances "which frame" (coarse) against "sub-frame phase" (fine) so the model does not
collapse onto the integer part. Task #15 (WS2).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def hierarchical_loss(logits, dt_soft, dt_gt, centers, w_coarse=1.0, w_reg=1.0):
    """Coarse cross-entropy on the integer lag + direct regression on the soft estimate.

    The coarse term keeps the correlation profile peaked at the right lag; the regression
    term sharpens it so the analytic parabolic refinement (in the head) reads an accurate
    sub-frame phase. Task #15 (WS2)."""
    K = (centers.numel() - 1) // 2
    gt_int = torch.round(dt_gt).clamp(-K, K)
    target_bin = (gt_int + K).long()                                  # (B,)
    ce = F.cross_entropy(logits, target_bin)
    reg = F.smooth_l1_loss(dt_soft, dt_gt)
    total = w_coarse * ce + w_reg * reg
    return total, {"ce": ce.item(), "reg": reg.item()}
