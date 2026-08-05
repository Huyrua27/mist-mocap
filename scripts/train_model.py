# -*- coding: utf-8 -*-
"""Train ContinuSyncFormer on the synthetic benchmark or Panoptic B1.

    python scripts/train_model.py --epochs 10
    python scripts/train_model.py --epochs 10 --occlusion 0.2   # occlusion-robust variant
    python scripts/train_model.py --data panoptic --epochs 10   # B1, task #16
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mist.model.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--occlusion", type=float, default=0.0, help="p occlude keypoint")
    ap.add_argument("--out", default="checkpoints/csf.pt")
    ap.add_argument("--data", choices=("synthetic", "panoptic"), default="synthetic",
                    help="panoptic = B1: real projected trajectories (task #16)")
    ap.add_argument("--root", default="data/panoptic_raw", help="Panoptic raw data root")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--motion", action="store_true",
                    help="motion (per-joint speed) input — recommended for --data panoptic")
    a = ap.parse_args()
    kw = {"occlusion_aware": a.occlusion > 0, "motion_input": a.motion}
    train_ds = val_ds = None
    if a.data == "panoptic":
        from mist.model.panoptic_dataset import PanopticPairDataset
        kw["n_joints"] = 19
        train_ds = PanopticPairDataset(root=a.root, split="train", n=a.n_train,
                                       seed=1, occlusion_p=a.occlusion, stride=6)
        val_ds = PanopticPairDataset(root=a.root, split="validation", n=a.n_val,
                                     seed=999, occlusion_p=a.occlusion)
    best = train(epochs=a.epochs, batch=a.batch, lr=a.lr, out=a.out,
                 model_kwargs=kw, occlusion_p=a.occlusion,
                 n_train=a.n_train, n_val=a.n_val, train_ds=train_ds, val_ds=val_ds)
    print(f"\nBest Accin@0.1 = {best:.3f}")


if __name__ == "__main__":
    main()
