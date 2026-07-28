# -*- coding: utf-8 -*-
"""Train ContinuSyncFormer on the (synthetic by default) keypoint benchmark.

    python scripts/train_model.py --epochs 10
    python scripts/train_model.py --epochs 10 --occlusion 0.2   # occlusion-robust variant

Swap the dataset for Panoptic-projected trajectories to train on B1 (task #16).
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
    a = ap.parse_args()
    kw = {"occlusion_aware": a.occlusion > 0}
    best = train(epochs=a.epochs, batch=a.batch, lr=a.lr, out=a.out,
                 model_kwargs=kw, occlusion_p=a.occlusion)
    print(f"\nBest Accin@0.1 = {best:.3f}")


if __name__ == "__main__":
    main()
