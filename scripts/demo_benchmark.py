"""Run the WS1 harness on a small deterministic synthetic dataset."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mist.benchmark import eval as bench_eval
from mist.benchmark import synthetic
from mist.benchmark.baselines import CaspiIrani, CrossCorrelation, DTW, ZeroOffset


def main() -> None:
    print("== MIST benchmark demo (synthetic) ==")
    samples = synthetic.make_dataset(
        n=24, T=60, fps=30.0, seed=1, max_offset=2.0
    )
    print(
        f"Generated {len(samples)} samples "
        "(30 fps, offsets in [-2, 2] frames).\n"
    )
    results = bench_eval.run(
        samples, [ZeroOffset(), CrossCorrelation(), DTW(), CaspiIrani()]
    )
    print(bench_eval.table(results))
    print()

    # In production these edges must be fitted on training data and frozen.
    velocity = np.array([sample.velocity for sample in samples])
    edges = tuple(
        round(float(value), 0) for value in np.quantile(velocity, [0.25, 0.5, 0.75])
    )
    print(
        f"(synthetic speed {velocity.min():.0f}-{velocity.max():.0f} px/s; "
        f"edges={edges})"
    )
    print(bench_eval.bucket_table(results, edges=edges))


if __name__ == "__main__":
    main()
