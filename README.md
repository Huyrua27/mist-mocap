# MIST — Millisecond-accurate Intra-frame Synchronization

Official code for **sub-frame temporal synchronization of multi-view video** for motion
capture. MIST estimates the fractional inter-camera offset `Δt ∈ ℝ` (in units of frames)
directly from 2D keypoint trajectories, targeting the sub-frame regime where a residual of
even a fraction of a frame corrupts 3D triangulation of fast motion.

Because the **IFID benchmark is not publicly accessible**, this repo ships a fully
reproducible **controlled-desync benchmark**: temporally-aligned sequences (CMU Panoptic /
synthetic) are re-sampled at a known fractional shift in the *keypoint domain*, yielding
exact, artifact-free ground-truth offsets in unlimited quantity.

---

## Highlights

- **Runs out of the box** — a synthetic benchmark with real numbers, no dataset download, no GPU.
- **Keypoint-domain desync generator** — exact sub-frame ground truth without pixel-interpolation artifacts.
- **Unified evaluation harness** — every method implements one interface; one call produces the metric table.
- **Standard metrics** — Frm.err, Accin@τ, Accex@i, MAE/RMSE (ms), reported per velocity bucket.
- **Baselines included** — Cross-Correlation (+parabolic sub-frame); DTW and Caspi–Irani interfaces.
- **On-set capture tools** — a fade-flash sync signal, a webcam recorder with per-frame host timestamps, and a ground-truth flash detector.

---

## Installation

```bash
git clone <repo-url> && cd mist-sync
python -m venv .venv && . .venv/Scripts/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # numpy, scipy, opencv
pip install -r requirements-model.txt                 # torch — only for the learned model
```

## Quick start

```bash
python scripts/demo_benchmark.py
python tests/test_core.py
```

`demo_benchmark.py` generates synthetic keypoints, injects labeled offsets, runs the
baselines, and prints:

```
    Method           n     Frm.err   Accin@0.1  Accin@0.25      MAE_ms     RMSE_ms
------------------------------------------------------------------------------
 Zero(floor)       240      0.9796      0.0583      0.1333      32.654      37.705
CC+parabolic       240      0.0936       0.725      0.8792       3.118       5.729
```

---

## The benchmark

Ground truth is generated in the **keypoint domain** rather than by interpolating pixels:

1. Load a temporally-synchronized multi-view sequence (CMU Panoptic, or synthetic).
2. For a target offset `Δt`, resample a view's 2D keypoint trajectory at times shifted by
   `Δt` via cubic-spline interpolation. Because keypoint trajectories are smooth, sub-frame
   resampling is near-exact — the injected `Δt` *is* the ground-truth label.
3. Split by **sequence** (never by frame) to avoid train/test leakage.

Evaluation is velocity-bucketed, exposing where methods degrade as object speed rises — the
core failure mode of frame-level synchronization.

---

## Repository structure

```
mist-sync/
├── mist/
│   ├── core/            # Shared contract: types + method interface
│   │   ├── types.py         KeypointSequence, SyncResult, SyncSample
│   │   └── interfaces.py     SyncMethod
│   ├── benchmark/       # Desync generator, metrics, synthetic data, eval harness
│   │   └── baselines/       CrossCorrelation, DTW, CaspiIrani, ZeroOffset
│   ├── model/           # ContinuSyncFormer (RoPE + cross-view attention + regression head)
│   ├── panoptic/        # CMU Panoptic loader / 2D projection
│   └── realworld/       # In-the-wild multi-camera capture pipeline
├── tools/               # sync_flash.html · record_webcams.py · sync_groundtruth.py
├── scripts/             # demo_benchmark.py · run_benchmark.py
├── tests/               # test_core.py
└── docs/                # shoot sheet · formula sheet · benchmark design
```

## Using the API

Every method — classical or learned — implements a single interface, so the harness runs
them identically:

```python
from mist.core import SyncMethod, SyncResult
from mist.benchmark import synthetic, eval

class MyMethod(SyncMethod):
    name = "My"
    def predict(self, a, b):
        return SyncResult(dt_frames=..., confidence=...)   # b relative to a, in frames

samples = synthetic.make_dataset(n=240)
results = eval.run(samples, [MyMethod()])
print(eval.table(results))
```

## Status

| Component | State |
|-----------|-------|
| Desync generator, metrics, harness, synthetic data | ✅ implemented |
| Cross-Correlation baseline | ✅ implemented |
| DTW, Caspi–Irani baselines | ✅ implemented |
| ContinuSyncFormer (model) | ⬜ architecture skeleton |
| CMU Panoptic loader | ✅ calibration + COCO-19 body poses |
| Real-world capture pipeline | ⬜ stub |
| Capture & ground-truth tools | ✅ used on real captures |

## Contributing

Feature branches → pull request against `main`. `python tests/test_core.py` must pass before
merge. Heavy data (videos, archives, extracted frames) is excluded via `.gitignore` and must
not be committed.

## Citation

```bibtex
@misc{mist,
  title  = {MIST: Millisecond-accurate Intra-frame Synchronization for Multi-view Motion Capture},
  author = {TODO},
  year   = {TODO}
}
```

## License

To be added (see `LICENSE`).
