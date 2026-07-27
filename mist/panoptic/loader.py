# -*- coding: utf-8 -*-
"""Nạp CMU Panoptic → KeypointSequence mỗi camera.  [STUB — Owner: P1 (WS1), task #1,#2]

Panoptic có 3D keypoints + calibration đã ĐỒNG BỘ sẵn → là "sự thật gốc" để tạo B1.
Quy trình:
  1. Tải bằng panoptic-toolbox (getData.sh) — chọn vài sequence.
  2. Đọc 3D skeletons (hdPose3d_stage1_coco19/*.json) + calibration (calibration_*.json).
  3. project_to_2d(): chiếu 3D → 2D mỗi camera bằng ma trận K,R,t.
  4. Trả dict {cam_name: KeypointSequence} — TẤT CẢ đang đồng bộ; desync.inject_offset
     sẽ tạo lệch có nhãn.

TODO(P1): cài đọc JSON + phép chiếu. Tham chiếu:
  https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox
"""
from __future__ import annotations
import numpy as np
from ..core.types import KeypointSequence


def project_to_2d(pts3d: np.ndarray, K, R, t) -> np.ndarray:
    """(T,J,3) thế giới → (T,J,2) ảnh, qua P = K[R|t]. TODO(P1): kiểm định méo ống kính."""
    T, J, _ = pts3d.shape
    Xc = pts3d @ np.asarray(R).T + np.asarray(t).reshape(1, 1, 3)   # world->cam
    x = Xc @ np.asarray(K).T
    return x[..., :2] / x[..., 2:3]


def load_sequence(seq_dir: str) -> dict:
    """Trả {cam_name: KeypointSequence} cho 1 sequence Panoptic. TODO(P1)."""
    raise NotImplementedError(
        "TODO(P1 #2): đọc hdPose3d + calibration Panoptic rồi project_to_2d")
