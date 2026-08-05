# -*- coding: utf-8 -*-
"""Evaluate a trained ContinuSyncFormer against classical baselines on B1 pairs.

    python scripts/eval_b1.py --checkpoint checkpoints/csf_b1.pt --split validation
    python scripts/eval_b1.py --checkpoint checkpoints/csf_b1.pt --split test

All methods see the *identical* (kp_a, kp_b, Δt) pairs, so the comparison is
apples-to-apples. Task #16 (WS2).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from mist.benchmark import metrics
from mist.benchmark.baselines import CrossCorrelation, DTW
from mist.core.types import KeypointSequence
from mist.model.continusyncformer import ContinuSyncFormer
from mist.model.panoptic_dataset import FPS, PanopticPairDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/csf_b1.pt")
    ap.add_argument("--split", choices=("validation", "test"), default="validation")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--root", default="data/panoptic_raw")
    args = ap.parse_args()

    ds = PanopticPairDataset(root=args.root, split=args.split, n=args.n,
                             seed=args.seed)
    print(f"split={args.split}  clips={len(ds.clips)}  samples={args.n}")
    kas, kbs, gts = [], [], []
    for i in range(args.n):
        ka, kb, dt, _, _ = ds[i]
        kas.append(ka); kbs.append(kb); gts.append(float(dt))
    gts = np.asarray(gts)

    model = ContinuSyncFormer(n_joints=19, motion_input=True).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"] if "model" in state else state)

    preds = {"ContinuSyncFormer": [], "CC+parabolic": [], "DTW": []}
    with torch.no_grad():
        ka_t, kb_t = torch.stack(kas), torch.stack(kbs)
        for s in range(0, args.n, 256):
            out = model(ka_t[s:s + 256], kb_t[s:s + 256])
            preds["ContinuSyncFormer"] += out["dt_hard"].tolist()
    cc, dtw = CrossCorrelation(), DTW()
    for ka, kb in zip(kas, kbs):
        a = KeypointSequence(ka.numpy().astype(np.float64), FPS, name="a")
        b = KeypointSequence(kb.numpy().astype(np.float64), FPS, name="b")
        for name, method in (("CC+parabolic", cc), ("DTW", dtw)):
            try:
                preds[name].append(method.predict(a, b).dt_frames)
            except Exception:
                preds[name].append(0.0)

    print(f"\n{'Method':>18} {'Frm.err':>9} {'Accin@0.1':>10} "
          f"{'Accin@0.25':>11} {'MAE_ms':>8}")
    for name, p in preds.items():
        p = np.asarray(p)
        print(f"{name:>18} {metrics.frm_err(p, gts):>9.4f} "
              f"{metrics.accin(p, gts, 0.1):>10.3f} "
              f"{metrics.accin(p, gts, 0.25):>11.3f} "
              f"{metrics.mae_ms(p, gts, FPS):>8.2f}")


if __name__ == "__main__":
    main()
