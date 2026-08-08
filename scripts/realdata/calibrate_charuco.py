# -*- coding: utf-8 -*-
"""ChArUco calibration for the real multi-camera rig (optional, for 3D).

Intrinsics come from the waving-board segment (many views per camera); extrinsics
come from a moment where all cameras see the SAME STATIC board (the board is the
shared world origin, so camera desync does not matter while it is still).
Outputs one projection matrix P = K[R|t] per camera.  Task #24 (real data).

Requires opencv-contrib-python (cv2.aruco). Set your board's real geometry below.

    python scripts/realdata/calibrate_charuco.py \
        --videos cam0.mp4 cam1.mp4 cam2.mp4 \
        --squares-x 5 --squares-y 7 --square-len 0.035 --marker-len 0.026 \
        --static-frame 300 --out data/realdata/session1/calib.json
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import cv2


def make_board(a):
    if not hasattr(cv2, "aruco"):
        raise SystemExit("need opencv-contrib-python for cv2.aruco")
    ad = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    try:                                          # OpenCV >= 4.7 API
        board = cv2.aruco.CharucoBoard((a.squares_x, a.squares_y),
                                       a.square_len, a.marker_len, ad)
        detector = cv2.aruco.CharucoDetector(board)
        return board, ad, detector
    except Exception:                             # older API
        board = cv2.aruco.CharucoBoard_create(a.squares_x, a.squares_y,
                                               a.square_len, a.marker_len, ad)
        return board, ad, None


def detect(gray, board, ad, detector):
    if detector is not None:
        corners, ids, _, _ = detector.detectBoard(gray)
        return corners, ids
    mc, mids, _ = cv2.aruco.detectMarkers(gray, ad)
    if mids is None or len(mids) == 0:
        return None, None
    _, corners, ids = cv2.aruco.interpolateCornersCharuco(mc, mids, gray, board)
    return corners, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--squares-x", type=int, required=True)
    ap.add_argument("--squares-y", type=int, required=True)
    ap.add_argument("--square-len", type=float, required=True, help="metres")
    ap.add_argument("--marker-len", type=float, required=True, help="metres")
    ap.add_argument("--static-frame", type=int, required=True,
                    help="frame index where all cameras see the same static board")
    ap.add_argument("--calib-stride", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    board, ad, detector = make_board(a)

    out = {"cameras": {}}
    for v in a.videos:
        name = os.path.splitext(os.path.basename(v))[0]
        cap = cv2.VideoCapture(v)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        allc, allids, static = [], [], None
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if i % a.calib_stride == 0 or i == a.static_frame:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                c, ids = detect(gray, board, ad, detector)
                if ids is not None and len(ids) >= 6:
                    if i % a.calib_stride == 0:
                        allc.append(c); allids.append(ids)
                    if i == a.static_frame:
                        static = (c, ids)
            i += 1
        cap.release()
        if len(allc) < 5 or static is None:
            print(f"{name}: insufficient board views ({len(allc)}) or no static frame -> skip")
            continue
        ok, K, dist, *_ = cv2.aruco.calibrateCameraCharuco(allc, allids, board, (W, H), None, None)
        c, ids = static
        ok2, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(c, ids, board, K, dist, None, None)
        R, _ = cv2.Rodrigues(rvec)
        out["cameras"][name] = {"K": K.tolist(), "dist": dist.ravel().tolist(),
                                "R": R.tolist(), "t": tvec.ravel().tolist(),
                                "reproj_rms": float(ok), "width": W, "height": H}
        print(f"{name}: intrinsics rms {ok:.3f}px, extrinsics from static frame OK")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=2)
    print(f"wrote calibration ({len(out['cameras'])} cams) -> {a.out}")


if __name__ == "__main__":
    main()
