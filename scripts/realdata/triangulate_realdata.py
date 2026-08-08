# -*- coding: utf-8 -*-
"""Real-data 3D downstream (optional): does sync correction lower reprojection error?

There is no 3D ground truth on real footage, so we use multi-view *reprojection
error* as the quality signal: desynced 2D observations cannot be explained by a
single 3D point, so triangulating them and reprojecting yields a large residual;
correct the drift and the residual drops. We compare no-correction vs CC-slide vs
CSF-slide, each camera resampled to the reference timeline by its estimated drift.
Task #24 (real data).

    python scripts/realdata/triangulate_realdata.py \
        --pose-dir data/realdata/session1/pose --calib data/realdata/session1/calib.json \
        --checkpoint checkpoints/csf_b1_t20.pt --window 20
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

import torch

from mist.benchmark.baselines import CrossCorrelation
from mist.benchmark.drift import fit_drift
from mist.benchmark.interpolation import cubic_sample
from mist.core.types import KeypointSequence
from mist.model.panoptic_dataset import _fill_gaps
from mist.panoptic.triangulation import projection_matrix, triangulate_masked


def load_pose(pose_dir, name):
    d = np.load(os.path.join(pose_dir, f"{name}.npz"))
    xy, _ = _fill_gaps(d["xy"].astype(np.float64))
    return xy


def reproj_error(X, P, obs):
    """Mean pixel reprojection error of 3D points X onto one view."""
    Xh = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    proj = (Xh @ P.T)
    proj = proj[:, :2] / proj[:, 2:3]
    ok = np.isfinite(proj).all(-1) & np.isfinite(obs).all(-1)
    return float(np.linalg.norm(proj[ok] - obs[ok], axis=-1).mean()) if ok.any() else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-dir", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--checkpoint", default="checkpoints/csf_b1_t20.pt")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    calib = json.load(open(args.calib, encoding="utf-8"))["cameras"]
    cams = list(calib.keys())
    if len(cams) < 2:
        raise SystemExit("need >= 2 calibrated cameras")
    mats = {c: projection_matrix(calib[c]["K"], calib[c]["R"], calib[c]["t"]) for c in cams}
    pose = {c: load_pose(args.pose_dir, c) for c in cams}
    n = min(len(pose[c]) for c in cams)
    ref = cams[0]

    cc = CrossCorrelation()
    cc_predict = lambda a, b: cc.predict(a, b).dt_frames
    csf_predict = None
    if os.path.exists(args.checkpoint):
        from mist.model.continusyncformer import ContinuSyncFormer
        model = ContinuSyncFormer(n_joints=19, motion_input=True).eval()
        st = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(st["model"] if "model" in st else st)

        def csf_predict(a, b):
            ka = torch.tensor(a.xy, dtype=torch.float32)[None]
            kb = torch.tensor(b.xy, dtype=torch.float32)[None]
            with torch.no_grad():
                return float(model(ka, kb)["dt_hard"][0])

    def drift_for(predict):
        curves = {ref: np.zeros(n)}
        for c in cams[1:]:
            curves[c], _, _ = fit_drift(predict, pose[ref][:n], pose[c][:n],
                                        args.fps, args.window, args.stride)
        return curves

    methods = {"naive": None, "CC-slide": cc_predict}
    if csf_predict:
        methods["CSF-slide"] = csf_predict

    idx = np.arange(n, dtype=np.float64)
    print(f"{len(cams)} cameras, {n} frames, ref={ref}")
    print(f"{'method':>12} {'reproj err (px)':>16}")
    for name, predict in methods.items():
        curves = {c: np.zeros(n) for c in cams} if predict is None else drift_for(predict)
        # Resample each camera to the reference timeline by its drift estimate.
        aligned = {}
        for c in cams:
            aligned[c] = cubic_sample(pose[c][:n], idx - curves[c], extrapolate=True) \
                if predict is not None or c != ref else pose[c][:n]
        pts = np.stack([aligned[c] for c in cams], axis=2)          # (n,19,V,2)
        valid = np.isfinite(pts).all(-1)
        P = pts.reshape(-1, len(cams), 2)
        X = triangulate_masked(P, [mats[c] for c in cams], valid.reshape(-1, len(cams)))
        errs = [reproj_error(X, mats[c], P[:, k, :]) for k, c in enumerate(cams)]
        print(f"{name:>12} {np.nanmean(errs):>16.2f}")
    print("\n(lower reprojection error = more consistent multi-view geometry after sync.)")


if __name__ == "__main__":
    main()
