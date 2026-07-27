# -*- coding: utf-8 -*-
"""Pha 0: giải nén 5 zip → MIST_data/<session>/<cam> + manifest + bỏ file trùng.
[STUB — Owner: P3 (WS3), task #19]

Cấu trúc đích:
  MIST_data/
    S0_sync_anchor/  iphone1.mov iphone2.mov iphone3.mov webcam0.mp4 webcam1.mp4
    S1_calibration/  ...
    ...
    manifest.csv     # cam, session, take, path, res, fps, frames, dur, có_timestamps

TODO(P3): giải nén (zipfile), gom theo tên session, dùng cv2 đọc fps/res/frames,
          phát hiện & bỏ file *_dup* (vd iphone3 S1 trùng). Xem tools/ để tham khảo.
"""
from __future__ import annotations


def organize(zip_paths: list[str], out_dir: str = "MIST_data") -> str:
    raise NotImplementedError("TODO(P3 #19): giải nén + gom session + manifest + dedup")
