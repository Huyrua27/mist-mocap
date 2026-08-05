# -*- coding: utf-8 -*-
"""Kiểu dữ liệu dùng CHUNG cho cả nhóm — mọi module nói cùng một ngôn ngữ.

Đây là "hợp đồng" (contract). Đừng đổi chữ ký các dataclass này mà không báo cả nhóm,
vì benchmark harness (P1), model (P2) và pipeline data thật (P3) đều import từ đây.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class KeypointSequence:
    """Quỹ đạo 2D keypoints của MỘT camera theo thời gian.

    xy         : ndarray (T, J, 2) — T frame, J khớp, toạ độ (x, y) pixel.
    fps        : khung hình/giây.
    timestamps : (T,) giây — tuỳ chọn; webcam dùng host-timestamp thật ở đây.
    name       : nhãn để debug (vd 'iphone1', 'cam0').
    """
    xy: np.ndarray
    fps: float
    timestamps: Optional[np.ndarray] = None
    name: str = ""

    def __post_init__(self):
        self.xy = np.asarray(self.xy, dtype=np.float64)
        assert self.xy.ndim == 3 and self.xy.shape[2] == 2, \
            f"xy phải là (T,J,2), nhận {self.xy.shape}"

    @property
    def T(self) -> int: return self.xy.shape[0]
    @property
    def J(self) -> int: return self.xy.shape[1]


@dataclass
class SyncResult:
    """Kết quả 1 method sync trả về cho 1 cặp camera."""
    dt_frames: float            # offset của b so với a, đơn vị FRAME (thập phân, có dấu)
    confidence: float = 1.0     # [0,1], tuỳ method

    def dt_seconds(self, fps: float) -> float:
        return self.dt_frames / fps


@dataclass
class SyncSample:
    """Một mẫu benchmark: cặp (a, b) + nhãn ground-truth."""
    a: KeypointSequence
    b: KeypointSequence
    dt_gt_frames: float         # offset THẬT đã tiêm (nhãn)
    velocity: float = 0.0       # vận tốc đặc trưng (để chia bucket)
    meta: dict = field(default_factory=dict)
