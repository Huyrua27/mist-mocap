# -*- coding: utf-8 -*-
"""Bộ tiêm lệch thời gian trong MIỀN KEYPOINT (đóng góp thay cho IFID không có).

Ý tưởng: Panoptic đã đồng bộ sẵn. Muốn có cặp lệch với GT chính xác, ta LẤY MẪU LẠI
quỹ đạo 2D tại các thời điểm dịch đi ∆t bằng nội suy. Vì keypoint là tín hiệu trơn,
nội suy sub-frame gần như hoàn hảo → GT ∆t tuyệt đối, KHÔNG artifact như tiêm pixel.

Owner: P1 (WS1). Dùng cho cả train lẫn test của B1.
"""
from __future__ import annotations
import numpy as np
from ..core.types import KeypointSequence


def _resample(xy: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    """Nội suy (T,J,2) từ mốc src_t sang dst_t. Ưu tiên cubic spline (scipy),
    fallback linear nếu thiếu scipy."""
    try:
        from scipy.interpolate import CubicSpline
        return CubicSpline(src_t, xy, axis=0, extrapolate=True)(dst_t)
    except Exception:
        T, J, C = xy.shape
        out = np.empty((len(dst_t), J, C))
        for j in range(J):
            for c in range(C):
                out[:, j, c] = np.interp(dst_t, src_t, xy[:, j, c])
        return out


def inject_offset(seq: KeypointSequence, dt_frames: float) -> KeypointSequence:
    """Trả bản sao của `seq` bị TRỄ dt_frames khung (dt>0 = trễ).

    dt có thể là số thực (sub-frame). Đây chính là nhãn ground-truth cho benchmark.
    """
    src_t = np.arange(seq.T, dtype=float)          # trục thời gian gốc (đơn vị frame)
    dst_t = src_t + dt_frames                      # lấy mẫu tại thời điểm dịch
    xy2 = _resample(seq.xy, src_t, dst_t)
    ts = None if seq.timestamps is None else seq.timestamps + dt_frames / seq.fps
    return KeypointSequence(xy2, seq.fps, timestamps=ts,
                            name=f"{seq.name}[{dt_frames:+.3f}f]")


def make_pair(seq: KeypointSequence, dt_frames: float):
    """Tiện ích: trả (a=gốc, b=lệch, dt_gt) cho harness."""
    return seq, inject_offset(seq, dt_frames), float(dt_frames)
