# -*- coding: utf-8 -*-
"""Detect periodic flashes in each camera and build the ground-truth drift curve.

Reads one ROI per camera (from `sync_groundtruth.py pick-roi`, saved as
<video>.roi.json, or passed with --roi), detects the flash peaks (sub-frame), and
writes a GT drift file: per non-reference camera, the true offset delta_c(t) that
the sync methods will be scored against.  Task #24 (real data).

    python scripts/realdata/detect_flash_gt.py \
        --videos cam0.mp4 cam1.mp4 cam2.mp4 --ref cam0.mp4 --n-flashes 18 \
        --out data/realdata/session1/gt_drift.json
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import sync_groundtruth as sg
from mist.realworld.flash_gt import build_gt_drift, drift_rate


def load_roi(video, roi_arg):
    if roi_arg:
        return tuple(int(v) for v in roi_arg.split(","))
    side = f"{video}.roi.json"
    if os.path.exists(side):
        d = json.load(open(side, encoding="utf-8"))
        return tuple(int(v) for v in (d["roi"] if "roi" in d else d))
    raise SystemExit(f"no ROI for {video}: pass --roi x,y,w,h or run pick-roi first")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--ref", required=True, help="reference video (offset 0 baseline)")
    ap.add_argument("--n-flashes", type=int, required=True,
                    help="total flash pulses per camera over the take")
    ap.add_argument("--roi", default=None, help="shared ROI x,y,w,h (else per-video .roi.json)")
    ap.add_argument("--min-gap", type=int, default=8, help="min frames between pulses")
    ap.add_argument("--fps", type=float, default=30.0, help="common nominal fps")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    def key(v):
        return os.path.splitext(os.path.basename(v))[0]

    flashes, n_frames = {}, 0
    for v in args.videos:
        roi = load_roi(v, args.roi)
        idxs, bright = sg.roi_brightness_series(v, roi)
        n_frames = max(n_frames, len(idxs))
        peaks = sg.detect_flashes(idxs, bright, n_flashes=args.n_flashes,
                                  min_gap=args.min_gap)
        sub = sorted(p["frame_subpix"] for p in peaks)
        lo = min(p["snr"] for p in peaks)
        print(f"{key(v):>14}: {len(sub)} pulses, min SNR {lo:.1f}"
              f"{'  <-- LOW, check ROI/brightness' if lo < 4 else ''}")
        flashes[key(v)] = sub

    ref = key(args.ref)
    gt = build_gt_drift(flashes, ref, n_frames)
    payload = {"reference": ref, "fps": args.fps, "n_frames": n_frames, "cameras": {}}
    print(f"\n{'camera':>14} {'drift(fr/take)':>14} {'offset span':>14}")
    for cam, g in gt.items():
        off = np.asarray(g["offset"])
        rate = drift_rate(g) * n_frames
        payload["cameras"][cam] = {"flash_ref": g["flash_ref"], "offset": g["offset"],
                                   "curve": g["curve"].tolist()}
        print(f"{cam:>14} {rate:>14.3f} {off.max()-off.min():>14.3f}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(payload, open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote GT drift -> {args.out}")


if __name__ == "__main__":
    main()
