# -*- coding: utf-8 -*-
"""Downstream 3D study: does sub-frame desync corrupt triangulation, and does
sync correction recover it -- with and without keypoint occlusion?  Task #24 / B3.

For each clip we project Panoptic 3D ground truth into several HD cameras
(pinhole), inject a sub-frame temporal offset into every non-reference camera,
optionally occlude keypoints per view, then triangulate 3D and report MPJPE
against the true 3D:

  naive   : triangulate the desynced observations directly (no correction)
  CC      : estimate each camera's offset by cross-correlation, resample, triangulate
  CSF     : same, using the trained ContinuSyncFormer (occlusion-aware if flagged)
  oracle  : resample by the TRUE offset (upper bound of perfect sync)

Under occlusion, an occluded joint is missing in that view (needs >= 2 visible
views to triangulate) and the sync methods must estimate the offset from degraded
streams -- the regime where a learned, occlusion-aware model can beat classical CC.

At offset 0 with no occlusion the naive MPJPE is ~0, validating the pipeline.
Panoptic world units are centimetres; results are printed in mm (x10).

    python scripts/triangulation_experiment.py --checkpoint checkpoints/csf_b1.pt
    python scripts/triangulation_experiment.py --occlusion-p 0.3 \
        --checkpoint checkpoints/csf_b1_occ.pt --occlusion-aware
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if hasattr(sys.stdout, "reconfigure"):  # non-UTF-8 consoles vs unicode glyphs
    sys.stdout.reconfigure(errors="replace")

import torch

from mist.benchmark.baselines import CrossCorrelation
from mist.benchmark.interpolation import cubic_sample
from mist.core.types import KeypointSequence
from mist.panoptic import load_calibration, load_pose3d, project_to_2d
from mist.panoptic.triangulation import (mpjpe, projection_matrix,
                                         triangulate_masked)

FPS = 29.97
DEFAULT_SEQUENCES = ("160422_haggling1", "160226_haggling1")
cams_cache: dict = {}


def pick_cameras(cams, nodes):
    keys = [(0, n) for n in nodes if (0, n) in cams]
    if len(keys) < 2:
        keys = sorted(k for k in cams if k[0] == 0)[:3]
    return keys


def clean_projections(seq_dir, keys, max_frames):
    xyz, _, _, _ = load_pose3d(seq_dir, max_frames=max_frames)
    mats, proj2d = {}, {}
    for key in keys:
        cam = cams_cache[seq_dir][key]
        mats[key] = projection_matrix(cam["K"], cam["R"], cam["t"])
        proj2d[key] = project_to_2d(xyz, cam["K"], cam["R"], cam["t"])  # pinhole
    return xyz, mats, proj2d


def build_clips(xyz, proj2d, keys, L, margin, stride, min_speed):
    W = L + 2 * margin
    finite = np.isfinite(xyz).all(axis=(1, 2))
    for key in keys:
        finite &= np.isfinite(proj2d[key]).all(axis=(1, 2))
    step = np.linalg.norm(np.diff(proj2d[keys[0]], axis=0), axis=-1)
    speed = np.concatenate([[0], np.median(step, axis=1)]) * FPS
    cum = np.cumsum(np.concatenate([[0], finite.astype(int)]))
    clips = []
    for s in range(0, len(xyz) - W + 1, stride):
        if cum[s + W] - cum[s] != W:
            continue
        if float(np.median(speed[s + margin:s + margin + L])) < min_speed:
            continue
        clips.append(s)
    return clips


def fill_time(xy, vis):
    """Linearly fill occluded (vis==0) joints over time so features stay finite."""
    out = xy.copy()
    t = np.arange(len(xy))
    for j in range(xy.shape[1]):
        for c in range(2):
            ok = vis[:, j] & np.isfinite(xy[:, j, c])
            if 1 < ok.sum() < len(xy):
                out[~ok, j, c] = np.interp(t[~ok], t[ok], xy[ok, j, c])
    return out


def predict_csf(model, a_xy, b_xy, va=None, vb=None):
    ka = torch.tensor(a_xy, dtype=torch.float32)[None]
    kb = torch.tensor(b_xy, dtype=torch.float32)[None]
    va_t = torch.tensor(va, dtype=torch.float32)[None] if va is not None else None
    vb_t = torch.tensor(vb, dtype=torch.float32)[None] if vb is not None else None
    with torch.no_grad():
        out = model(ka, kb, va_t, vb_t)
    return float(out["dt_hard"][0])


def run(args):
    rng = np.random.default_rng(args.seed)
    cc = CrossCorrelation()
    model = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        from mist.model.continusyncformer import ContinuSyncFormer
        model = ContinuSyncFormer(n_joints=19, motion_input=True,
                                  occlusion_aware=args.occlusion_aware).eval()
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"] if "model" in state else state)

    offsets = [float(x) for x in args.offsets.split(",")]
    methods = ["naive", "CC", "oracle"] + (["CSF"] if model else [])
    results = {o: {m: [] for m in methods} for o in offsets}
    p = args.occlusion_p
    n_clips_total = 0

    for seq_id in args.sequences:
        seq_dir = os.path.join(args.root, seq_id)
        cams_cache[seq_dir] = load_calibration(seq_dir)
        keys = pick_cameras(cams_cache[seq_dir], args.cameras)
        xyz, mats, proj2d = clean_projections(seq_dir, keys, args.max_frames)
        margin = int(np.ceil(max(abs(o) for o in offsets))) + 2
        clips = build_clips(xyz, proj2d, keys, args.clip_len, margin,
                            args.stride, args.min_speed)[:args.max_clips]
        n_clips_total += len(clips)
        others = keys[1:]
        mat_list = [mats[k] for k in keys]
        L = args.clip_len

        for start in clips:
            w0 = start - margin
            win = slice(w0, w0 + L + 2 * margin)
            lu = np.arange(margin, margin + L)
            truth = xyz[start:start + L]
            ref_obs = proj2d[keys[0]][win][lu]
            # Per-view visibility over the scored frames (independent occlusion).
            vis = {k: (rng.random((L, 19)) > p) if p > 0 else np.ones((L, 19), bool)
                   for k in keys}

            for d in offsets:
                deltas = {k: d * (1 if i % 2 == 0 else -1)
                          for i, k in enumerate(others)}
                obs_grid = {k: cubic_sample(proj2d[k][win],
                                            np.arange(L + 2 * margin) + deltas[k],
                                            extrapolate=True) for k in others}

                def triangulate_with(shifts):
                    views = [ref_obs]
                    for k in others:
                        views.append(cubic_sample(obs_grid[k], lu - shifts[k],
                                                  extrapolate=True))
                    pts = np.stack(views, axis=2)                # (L,19,V,2)
                    vmask = np.stack([vis[k] for k in keys], axis=2)  # (L,19,V)
                    finite = np.isfinite(pts).all(-1)
                    valid = vmask & finite
                    P = pts.reshape(-1, len(keys), 2)
                    est = triangulate_masked(P, mat_list, valid.reshape(-1, len(keys)))
                    return est.reshape(L, 19, 3)

                results[d]["naive"].append(
                    mpjpe(triangulate_with({k: 0.0 for k in others}), truth))
                results[d]["oracle"].append(
                    mpjpe(triangulate_with(deltas), truth))

                # Offset estimation from (occluded) streams, ref vs each camera.
                ref_fill = fill_time(ref_obs, vis[keys[0]])
                for name in [m for m in ("CC", "CSF") if m in methods]:
                    est_shifts = {}
                    for k in others:
                        cam_obs = obs_grid[k][lu]
                        cam_fill = fill_time(cam_obs, vis[k])
                        try:
                            if name == "CC":
                                a = KeypointSequence(ref_fill, FPS, name="a")
                                b = KeypointSequence(cam_fill, FPS, name="b")
                                est_shifts[k] = float(cc.predict(a, b).dt_frames)
                            else:
                                va = vis[keys[0]].astype(np.float32) if p > 0 else None
                                vb = vis[k].astype(np.float32) if p > 0 else None
                                est_shifts[k] = predict_csf(model, ref_fill, cam_fill, va, vb)
                        except Exception:
                            est_shifts[k] = 0.0
                    results[d][name].append(mpjpe(triangulate_with(est_shifts), truth))

    tag = f"occlusion_p={p}" + (" occ-aware" if args.occlusion_aware else "")
    print(f"\nTriangulation MPJPE (mm) -- {n_clips_total} clips, "
          f"{len(args.sequences)} sequences, {len(args.cameras)} cameras, {tag}")
    print(f"{'offset(fr)':>10} " + "".join(f"{m:>10}" for m in methods))
    for d in offsets:
        row = f"{d:>10.2f} "
        for m in methods:
            vals = [v for v in results[d][m] if np.isfinite(v)]
            row += f"{10.0 * np.mean(vals):>10.2f}" if vals else f"{'-':>10}"
        print(row)
    print("\n(lower MPJPE is better; oracle = perfect-sync bound.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/panoptic_raw")
    ap.add_argument("--checkpoint", default="checkpoints/csf_b1.pt")
    ap.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    ap.add_argument("--cameras", type=int, nargs="+", default=[0, 10, 20])
    ap.add_argument("--offsets", default="0,0.5,1.0,2.0,3.0")
    ap.add_argument("--clip-len", type=int, default=60)
    ap.add_argument("--stride", type=int, default=120)
    ap.add_argument("--max-clips", type=int, default=6)
    ap.add_argument("--max-frames", type=int, default=4000)
    ap.add_argument("--min-speed", type=float, default=40.0)
    ap.add_argument("--occlusion-p", type=float, default=0.0)
    ap.add_argument("--occlusion-aware", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
