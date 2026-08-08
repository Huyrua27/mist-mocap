# -*- coding: utf-8 -*-
"""Real-data drift experiment: recover the true (flash-measured) drift from real
2D keypoints, and compare CC-const / CC-slide / CSF-slide against ground truth.

This is the real-world counterpart of scripts/drift_experiment.py: same methods and
metric, but the offset is the genuine clock drift between independent devices and the
ground truth comes from on-set flashes rather than a simulator.  Task #24 (real data).

    python scripts/realdata/run_realdata_sync.py \
        --pose-dir data/realdata/session1/pose \
        --gt data/realdata/session1/gt_drift.json \
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
from mist.core.types import KeypointSequence
from mist.model.panoptic_dataset import _fill_gaps


def load_pose(pose_dir, name):
    d = np.load(os.path.join(pose_dir, f"{name}.npz"))
    xy, _ = _fill_gaps(d["xy"].astype(np.float64))
    return xy, float(d["fps"])


def csf_predictor(checkpoint):
    from mist.model.continusyncformer import ContinuSyncFormer
    model = ContinuSyncFormer(n_joints=19, motion_input=True).eval()
    st = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(st["model"] if "model" in st else st)

    def predict(a, b):
        ka = torch.tensor(a.xy, dtype=torch.float32)[None]
        kb = torch.tensor(b.xy, dtype=torch.float32)[None]
        with torch.no_grad():
            return float(model(ka, kb)["dt_hard"][0])
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-dir", required=True)
    ap.add_argument("--gt", required=True, help="gt_drift.json from detect_flash_gt.py")
    ap.add_argument("--checkpoint", default="checkpoints/csf_b1_t20.pt")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    gt = json.load(open(args.gt, encoding="utf-8"))
    ref = gt["reference"]
    ref_xy, _ = load_pose(args.pose_dir, ref)
    cc = CrossCorrelation()
    cc_predict = lambda a, b: cc.predict(a, b).dt_frames
    csf_predict = csf_predictor(args.checkpoint) if os.path.exists(args.checkpoint) else None
    methods = ["CC-const", "CC-slide"] + (["CSF-slide"] if csf_predict else [])

    print(f"reference: {ref}   window={args.window}")
    print(f"{'camera':>12} {'true drift':>11} " + "".join(f"{m:>11}" for m in methods))
    rows = {m: [] for m in methods}
    for cam, g in gt["cameras"].items():
        if cam == ref:
            continue
        cam_xy, _ = load_pose(args.pose_dir, cam)
        n = min(len(ref_xy), len(cam_xy), len(g["curve"]))
        a_stream, b_stream = ref_xy[:n], cam_xy[:n]
        delta_true = np.asarray(g["curve"][:n], dtype=np.float64)
        idx = np.arange(n, dtype=np.float64)
        true_span = float(delta_true.max() - delta_true.min())

        est = {}
        est["CC-const"] = np.full(n, cc.predict(
            KeypointSequence(a_stream, args.fps), KeypointSequence(b_stream, args.fps)).dt_frames)
        est["CC-slide"], _, _ = fit_drift(cc_predict, a_stream, b_stream, args.fps,
                                          args.window, args.stride)
        if csf_predict:
            est["CSF-slide"], _, _ = fit_drift(csf_predict, a_stream, b_stream, args.fps,
                                               args.window, args.stride)
        errs = {m: float(np.mean(np.abs(est[m] - delta_true))) for m in methods}
        for m in methods:
            rows[m].append(errs[m])
        print(f"{cam:>12} {true_span:>11.3f} " +
              "".join(f"{errs[m]:>11.4f}" for m in methods))

    print("\n" + f"{'MEAN':>12} {'':>11} " +
          "".join(f"{np.mean(rows[m]):>11.4f}" for m in methods))
    print(f"{'STD':>12} {'':>11} " +
          "".join(f"{np.std(rows[m]):>11.4f}" for m in methods))
    print("\n(error = mean |estimated delta(t) - flash-GT delta(t)| in frames, lower better.)")


if __name__ == "__main__":
    main()
