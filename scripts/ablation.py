# -*- coding: utf-8 -*-
"""Ablation study: toggle each component and report Accin@0.1. Task #17 (WS2).

    python scripts/ablation.py --epochs 6

Ablated axes:
  - pos_encoding : rope | sincos | none      (Continuous Temporal Encoding vs discrete PE)
  - attention    : cross-view | self-only    (cross-view attention on/off)
  - input        : pose (normalized) | raw    (normalized keypoints vs raw coords)
Occlusion robustness is evaluated separately (train with --occlusion in train_model.py).
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mist.model.train import train


CONFIGS = [
    ("full (rope+cross+pose)", dict(pos_encoding="rope",   cross_view=True,  normalize=True)),
    ("PE=sincos",              dict(pos_encoding="sincos", cross_view=True,  normalize=True)),
    ("PE=none",                dict(pos_encoding="none",   cross_view=True,  normalize=True)),
    ("self-only (no cross)",   dict(pos_encoding="rope",   cross_view=False, normalize=True)),
    ("raw coords (no norm)",   dict(pos_encoding="rope",   cross_view=True,  normalize=False)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    a = ap.parse_args()
    rows = []
    for name, kw in CONFIGS:
        print(f"\n=== {name} ===")
        best = train(epochs=a.epochs, out=None, model_kwargs=kw, log=True)
        rows.append((name, best))
    print("\n================ ABLATION (Accin@0.1) ================")
    for name, best in rows:
        print(f"  {name:28} {best:.3f}")


if __name__ == "__main__":
    main()
