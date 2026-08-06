# MIST — Millisecond-accurate Intra-frame Synchronization

Official code for **sub-frame temporal synchronization of multi-view video** for motion
capture. MIST estimates the fractional inter-camera offset `Δt ∈ ℝ` (in units of frames)
directly from 2D keypoint trajectories, targeting the sub-frame regime where a residual of
even a fraction of a frame corrupts 3D triangulation of fast motion.

Because the **IFID benchmark is not publicly accessible**, this repo ships a fully
reproducible **controlled-desync benchmark**: temporally-aligned sequences (CMU Panoptic /
synthetic) are re-sampled at a known fractional shift in the *keypoint domain*, yielding
exact, artifact-free ground-truth offsets in unlimited quantity.

Beyond a static offset, MIST targets the realistic case where **independent devices drift**:
camera B's clock runs at a slightly different rate, so the inter-camera offset *changes across
a take*. A single lag cannot represent this. **ContinuSyncFormer** estimates a local offset in
short sliding windows — where classical cross-correlation becomes unreliable but a learned
model stays accurate — and tracks the drift, keeping multi-view 3D reconstruction steady.

---

## Highlights

- **Runs out of the box** — a synthetic benchmark with real numbers, no dataset download, no GPU.
- **Keypoint-domain desync generator** — exact sub-frame ground truth without pixel-interpolation artifacts.
- **Unified evaluation harness** — every method implements one interface; one call produces the metric table.
- **ContinuSyncFormer** — a learned sub-frame sync model (RoPE + cross-view attention + hierarchical head) on motion features; competitive with classical baselines and **wins under camera drift**.
- **Downstream 3D study** — inject desync, triangulate, measure MPJPE: sub-frame desync corrupts 3D, and sliding-window learned correction recovers it.
- **Standard metrics** — Frm.err, Accin@τ, Accex@i, MAE/RMSE (ms), reported per velocity bucket.
- **Baselines included** — Cross-Correlation (+parabolic sub-frame); DTW and Caspi–Irani.
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
python scripts/demo_benchmark.py            # synthetic benchmark, classical baselines
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

Learned model and the drift / downstream-3D studies (need `requirements-model.txt` and
the Panoptic sample — see `docs/ws1_results/`):

```bash
python scripts/train_model.py --data panoptic --motion --clip-len 20 --select frmerr \
    --epochs 15 --lr 1e-4 --out checkpoints/csf_b1_t20.pt   # train the sync model (B1)
python scripts/eval_b1.py --checkpoint checkpoints/csf_b1_t20.pt --split test  # vs baselines
python scripts/drift_experiment.py --checkpoint checkpoints/csf_b1_t20.pt --window 20
python scripts/drift_triangulation.py --checkpoint checkpoints/csf_b1_t20.pt --window 20
```

Result tables and a written summary live in `docs/paper/`.

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

## Drift: sub-frame sync matters for 3D, and rigs drift

Real multi-device rigs do not share a clock, so the inter-camera offset drifts across a take.
The downstream study makes the consequence concrete: project Panoptic 3D into several cameras,
inject a (possibly drifting) offset, triangulate, and measure MPJPE against the true 3D.

- **Desync corrupts 3D** — MPJPE grows monotonically with the offset (0 → ~11 mm at 3 frames).
- **A single-lag correction cannot track drift** — its 3D error grows across the take.
- **ContinuSyncFormer, applied in short sliding windows + a robust fit, tracks the drift** and
  holds MPJPE nearest the perfect-sync oracle. It is reliable on short windows where classical
  cross-correlation is not — the mechanism behind the win.

See `docs/paper/results_summary.md` for the full tables and an honest account of the limits.

---

## Repository structure

```
mist-sync/
├── mist/
│   ├── core/            # Shared contract: types + method interface
│   │   ├── types.py         KeypointSequence, SyncResult, SyncSample
│   │   └── interfaces.py     SyncMethod
│   ├── benchmark/       # Desync generator, metrics, synthetic data, eval harness
│   │   ├── baselines/       CrossCorrelation, DTW, CaspiIrani, ZeroOffset
│   │   └── drift.py         Drift generator + sliding-window line/curve recovery
│   ├── model/           # ContinuSyncFormer (RoPE + cross-view attn + hierarchical head)
│   ├── panoptic/        # CMU Panoptic loader / 2D projection / DLT triangulation
│   └── realworld/       # In-the-wild multi-camera capture pipeline (stub)
├── tools/               # sync_flash.html · record_webcams.py · sync_groundtruth.py
├── scripts/             # demo_benchmark · train_model · eval_b1 · finalize_ws1
│                        #  · triangulation_experiment · drift_experiment
│                        #  · drift_triangulation · drift_nonlinear · ablation · probe_regimes
├── tests/               # test_core · test_ws1 · test_ws1_finalization
└── docs/                # ws1_results/ · paper/ (result tables + summary) · shoot/formula sheets
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
| Cross-Correlation, DTW, Caspi–Irani baselines | ✅ implemented |
| CMU Panoptic loader + calibrated projection | ✅ calibration + COCO-19 body poses |
| WS1 Panoptic benchmark (Protocols A/B) | ✅ finalized (`docs/ws1_results/`) |
| ContinuSyncFormer (model) + training on B1 | ✅ implemented, trained |
| Drift benchmark + sliding-window recovery | ✅ implemented |
| Downstream 3D triangulation / MPJPE study | ✅ implemented |
| Capture & ground-truth tools | ✅ used on real captures |
| Real-world capture pipeline | ⬜ stub |

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
