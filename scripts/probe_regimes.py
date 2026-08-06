# -*- coding: utf-8 -*-
"""Probe: where is classical CC weak? Sweep clip length (and occlusion) and
measure CC vs CSF sub-frame accuracy on B1 validation. If CC degrades sharply on
short windows, the drift-via-short-window learned advantage is real.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
import torch
from mist.benchmark import metrics
from mist.benchmark.baselines import CrossCorrelation
from mist.core.types import KeypointSequence
from mist.model.panoptic_dataset import FPS, PanopticPairDataset


def eval_len(T, n, occ, checkpoint):
    ds = PanopticPairDataset(split="validation", n=n, seed=999, T=T, occlusion_p=occ)
    kas, kbs, gts = [], [], []
    for i in range(n):
        ka, kb, dt, _, _ = ds[i]
        kas.append(ka.numpy().astype(np.float64)); kbs.append(kb.numpy().astype(np.float64))
        gts.append(float(dt))
    gts = np.asarray(gts)
    out = {}
    cc = CrossCorrelation()
    p = []
    for xa, xb in zip(kas, kbs):
        try:
            p.append(cc.predict(KeypointSequence(xa, FPS), KeypointSequence(xb, FPS)).dt_frames)
        except Exception:
            p.append(0.0)
    p = np.asarray(p)
    out["CC"] = (metrics.frm_err(p, gts), metrics.accin(p, gts, 0.1))
    if checkpoint and os.path.exists(checkpoint):
        from mist.model.continusyncformer import ContinuSyncFormer
        m = ContinuSyncFormer(n_joints=19, motion_input=True).eval()
        st = torch.load(checkpoint, map_location="cpu", weights_only=True)
        m.load_state_dict(st["model"] if "model" in st else st)
        with torch.no_grad():
            preds = []
            ka_t, kb_t = torch.tensor(np.array(kas), dtype=torch.float32), torch.tensor(np.array(kbs), dtype=torch.float32)
            for s in range(0, n, 256):
                preds += m(ka_t[s:s+256], kb_t[s:s+256])["dt_hard"].tolist()
        preds = np.asarray(preds)
        out["CSF"] = (metrics.frm_err(preds, gts), metrics.accin(preds, gts, 0.1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/csf_b1.pt")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--lengths", default="16,24,32,48,72")
    ap.add_argument("--occ", type=float, default=0.0)
    a = ap.parse_args()
    print(f"B1 validation, n={a.n}, occlusion={a.occ}  (Frm.err / Acc@0.1)")
    print(f"{'clip_len':>8}  {'CC':>18}  {'CSF (OOD)':>18}")
    for T in [int(x) for x in a.lengths.split(",")]:
        r = eval_len(T, a.n, a.occ, a.checkpoint)
        cc = f"{r['CC'][0]:.3f} / {r['CC'][1]:.3f}"
        csf = f"{r['CSF'][0]:.3f} / {r['CSF'][1]:.3f}" if "CSF" in r else "-"
        print(f"{T:>8}  {cc:>18}  {csf:>18}")


if __name__ == "__main__":
    main()
