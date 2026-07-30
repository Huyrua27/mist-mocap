"""Minimal real-data WS1 integration check on the Panoptic sample sequence."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mist.benchmark import eval as bench_eval
from mist.benchmark.baselines import CaspiIrani, CrossCorrelation, DTW
from mist.benchmark.interpolation import cubic_sample
from mist.benchmark.metrics import sequence_velocity_px_s
from mist.core.types import KeypointSequence, SyncSample
from mist.panoptic import load_sequence


def cross_view_sample(
    a: KeypointSequence, b: KeypointSequence, offset: float
) -> SyncSample:
    start = max(0, int(np.ceil(-offset)))
    stop = min(a.T, int(np.floor(b.T - 1 - offset)) + 1)
    if stop - start < 16:
        raise ValueError("not enough common frames after applying the offset")
    reference_time = np.arange(start, stop, dtype=np.float64)
    a_crop = KeypointSequence(
        a.xy[start:stop], a.fps, name=f"{a.name}/crop-{start}-{stop}"
    )
    b_shifted = KeypointSequence(
        cubic_sample(b.xy, reference_time + offset),
        b.fps,
        name=f"{b.name}/offset-{offset:+.3f}",
    )
    return SyncSample(
        a=a_crop,
        b=b_shifted,
        dt_gt_frames=float(offset),
        velocity=sequence_velocity_px_s(a_crop),
        meta={"source_sequence": a.name.split("/", 1)[0]},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=Path("data/panoptic_raw/171204_pose1_sample"),
    )
    parser.add_argument("--max-frames", type=int, default=101)
    args = parser.parse_args()

    views = load_sequence(
        str(args.sequence_dir),
        camera_keys=[(0, 0), (0, 5)],
        max_frames=args.max_frames,
    )
    camera_names = list(views)
    offsets = (-4.5, -2.25, -0.4, 0.6, 1.75, 3.3)
    samples = [
        cross_view_sample(views[camera_names[0]], views[camera_names[1]], offset)
        for offset in offsets
    ]
    methods = [
        CrossCorrelation(max_lag=8),
        DTW(oversample_factor=4, max_warp_frames=8, max_frames=101),
        CaspiIrani(alpha_range=(0.99, 1.01), alpha_steps=5, max_lag=8),
    ]
    print(
        f"sequence={args.sequence_dir.name}; cameras={camera_names}; "
        f"frames={views[camera_names[0]].T}; offsets={offsets}"
    )
    print(bench_eval.table(bench_eval.run(samples, methods)))


if __name__ == "__main__":
    main()
