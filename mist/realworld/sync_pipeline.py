# -*- coding: utf-8 -*-
"""Pha 1: sync 5 cam data thật.  [STUB — Owner: P3 (WS3), task #20,#21]

Đã kiểm chứng trên data: điện thoại chạy MIST Sync Flash nằm trên bàn giữa, màn hình
lóe TRẮNG; dò bằng ROI SÁT màn hình cho SNR ~27 (toàn khung thất bại — phòng sáng).

Quy trình:
  1. auto_flash_roi(video): grid-search vùng transient NHỎ + sáng>=1 lần (đã POC ở
     scratchpad) HOẶC tools/sync_groundtruth.py pick-roi để click tay. 1 ROI/cam
     (điện thoại cố định trên bàn → ROI dùng chung mọi take của cam đó).
  2. detect flash (tools/sync_groundtruth.py detect_flashes) → đỉnh sub-frame mỗi cam.
  3. offset từng cặp cam so với 1 cam tham chiếu → bảng offset 5 cam.
  4. gộp thêm: webcam *_timestamps.csv (host-time) + iPhone timecode (cần ffprobe) để
     chéo kiểm + đo drift α trên take dài.

TODO(P3): nối các bước, tận dụng tools/sync_groundtruth.py (đã có detect_flashes,
          roi_brightness_series, decode_binary_led...).
"""
from __future__ import annotations


def auto_flash_roi(video_path: str):
    """Tự dò ROI màn hình điện thoại (đốm nhỏ, sáng transient). TODO(P3 #20)."""
    raise NotImplementedError("TODO(P3 #20): port POC auto-ROI từ scratchpad")


def solve_offsets(session_dir: str, ref_cam: str = "iphone1") -> dict:
    """Trả {cam: dt_frames so với ref} cho 1 session. TODO(P3 #20,#21)."""
    raise NotImplementedError("TODO(P3): detect flash mỗi cam -> offset 5 cam")
