# -*- coding: utf-8 -*-
"""Ablation study: toggle each component and report Accin@0.1. Task #17 (WS2).

    python scripts/ablation.py --epochs 6                    # synthetic
    python scripts/ablation.py --data panoptic --lr 1e-4     # B1 (paper table)

Ablated axes:
  - pos_encoding : rope | sincos | none      (Continuous Temporal Encoding vs discrete PE)
  - attention    : cross-view | self-only    (cross-view attention on/off)
  - input        : motion | pose | raw       (per-joint speed vs normalized/raw coords)
Occlusion robustness is evaluated separately (train with --occlusion in train_model.py).
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mist.model.train import train


def build_configs(motion_baseline: bool):
    """Baseline = the strongest config for the data regime; each row flips one axis."""
    base = dict(pos_encoding="rope", cross_view=True, normalize=True,
                motion_input=motion_baseline)
    tag = "motion" if motion_baseline else "pose"
    rows = [
        (f"full (rope+cross+{tag})", dict(base)),
        ("PE=sincos",                dict(base, pos_encoding="sincos")),
        ("PE=none",                  dict(base, pos_encoding="none")),
        ("self-only (no cross)",     dict(base, cross_view=False)),
    ]
    if motion_baseline:
        rows.append(("input=pose (no motion)", dict(base, motion_input=False)))
        rows.append(("input=raw (no motion/norm)",
                     dict(base, motion_input=False, normalize=False)))
    else:
        rows.append(("raw coords (no norm)", dict(base, normalize=False)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--data", choices=("synthetic", "panoptic"), default="synthetic")
    ap.add_argument("--root", default="data/panoptic_raw")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=1000)
    a = ap.parse_args()

    train_ds = val_ds = None
    extra = {}
    if a.data == "panoptic":
        from mist.model.panoptic_dataset import PanopticPairDataset
        extra["n_joints"] = 19
        train_ds = PanopticPairDataset(root=a.root, split="train", n=a.n_train,
                                       seed=1, stride=6)
        val_ds = PanopticPairDataset(root=a.root, split="validation", n=a.n_val,
                                     seed=999)

    rows = []
    for name, kw in build_configs(motion_baseline=a.data == "panoptic"):
        print(f"\n=== {name} ===", flush=True)
        best = train(epochs=a.epochs, lr=a.lr, out=None, model_kwargs={**kw, **extra},
                     n_train=a.n_train, n_val=a.n_val,
                     train_ds=train_ds, val_ds=val_ds, log=True)
        rows.append((name, best))
    print(f"\n============ ABLATION [{a.data}] (best val Accin@0.1) ============")
    for name, best in rows:
        print(f"  {name:28} {best:.3f}")


if __name__ == "__main__":
    main()
