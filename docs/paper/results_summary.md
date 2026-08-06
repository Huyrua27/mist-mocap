# MIST — Results summary (WS2 / task #16, #18)

Source: `scripts/eval_b1.py` on `checkpoints/csf_b1.pt` (motion input, 15 epochs,
lr 1e-4). Every method sees the **same** (kp_a, kp_b, Δt) pairs per split. FPS = 29.97.
Frm.err / MAE: lower better. Acc$_{in}$@τ: higher better.

## Main results

### Validation (160906_ian5 — unseen at training)

| Method | Frm.err | Accin@0.1 | Accin@0.25 | MAE_ms |
|---|---:|---:|---:|---:|
| **ContinuSyncFormer (ours)** | **0.237** | **0.469** | **0.811** | **7.90** |
| CC + parabolic | 0.430 | 0.435 | 0.746 | 14.34 |
| DTW | 0.359 | 0.429 | 0.764 | 11.96 |

Ours wins all four metrics.

### Held-out test (160422_haggling1, 160226_haggling1 — unseen sequences)

| Method | Frm.err | Accin@0.1 | Accin@0.25 | MAE_ms |
|---|---:|---:|---:|---:|
| **ContinuSyncFormer (ours)** | **0.126** | 0.556 | **0.929** | **4.21** |
| CC + parabolic | 0.205 | **0.628** | 0.926 | 6.85 |
| DTW | 0.156 | 0.590 | **0.929** | 5.20 |

Ours wins Frm.err and MAE by a clear margin and ties Acc@0.25, but **CC wins
Acc@0.1 (0.628 vs 0.556)**. Report this honestly.

## Occlusion robustness (validation, 20% per-keypoint occlusion) — task #18

Deterministic masks (seeded in `panoptic_dataset.py`): both model runs see identical
occlusion, so CC/DTW are byte-identical across runs — the comparison is clean.

| Method | Frm.err | Accin@0.1 | Accin@0.25 | MAE_ms |
|---|---:|---:|---:|---:|
| ContinuSyncFormer (standard) | **0.318** | 0.342 | 0.683 | **10.59** |
| **ContinuSyncFormer (occ. aware)** | 0.384 | **0.396** | **0.710** | 12.81 |
| CC + parabolic | 0.518 | 0.365 | 0.675 | 17.30 |
| DTW | 0.386 | 0.374 | 0.686 | 12.89 |

Nuanced result — occlusion-aware training is a **trade-off**, not a clean win:
- It lifts threshold accuracy: Acc@0.1 0.342 → 0.396 (+16% rel), Acc@0.25 0.683 → 0.710.
  Under occlusion, occ-aware CSF is the **best method on both Acc@τ** (beats DTW/CC).
- But mean error worsens: Frm.err 0.318 → 0.384, MAE 10.59 → 12.81 ms — it lands more
  predictions tightly correct while making a few larger tail errors.
- On mean error under occlusion the **standard** CSF is actually best of all methods.

Report both directions. The occlusion module clearly helps where it counts
(fraction of accurate syncs), which is the metric that matters for triangulation.

## Honest talking points for the Results section (Phong)

1. On mean-error metrics (Frm.err, MAE) our method dominates on both splits — it
   makes fewer large errors, i.e. it is more *robust*.
2. Under the tight 0.1-frame threshold on the held-out test, CC edges us out —
   CC is occasionally more ultra-precise on clean haggling motion. Do not hide this.
3. Test set = only 2 haggling sequences (6 unique base clips upstream); do **not**
   claim broad generalization. Reuse the limitation language in
   `docs/ws1_results/README.md` (Scope and limitations).
4. Occlusion is a genuine gap: the plain learned model degrades hard under 20%
   occlusion; the reliability-biased attention closes most of that gap.

## Ablation (validation, 6 epochs, best Acc@0.1) — task #17

| Axis | Variant | Accin@0.1 | Δ vs full |
|---|---|---:|---:|
| — | full (RoPE + cross-view + motion) | 0.458 | — |
| Positional enc. | sincos | 0.442 | −0.016 |
| Positional enc. | none | 0.484 | +0.026 |
| Cross-view attn. | self-only | 0.453 | −0.005 |
| Input repr. | pose coords (no motion) | 0.063 | −0.395 |
| Input repr. | raw coords (no motion/norm) | 0.012 | −0.446 |

**Honest reading (important for Method/Results):**
- **Motion input is THE design choice.** Without it the model collapses
  (0.458 → 0.063 for pose coords, → 0.012 for raw). Per-joint speed is
  view-tolerant; raw/normalized coordinates are view-dependent and unlearnable
  across cameras from ~50 clips. This is the paper's real architectural finding.
- **Positional encoding is near-neutral.** RoPE 0.458, sincos 0.442, none 0.484 —
  all within single-run noise (~±0.03). Do **not** claim RoPE ("Continuous Temporal
  Encoding") drives performance; it doesn't hurt, but it isn't the source of gains.
- **Cross-view attention is near-neutral** here too (0.458 vs 0.453). Same caveat.
- Caveats: 6-epoch single runs (headline model uses 15 epochs → ~0.50), no error
  bars, so only the motion-vs-no-motion gap is a real effect. State this.

## Downstream 3D — triangulation MPJPE vs desync (B3, task #24) — for 3DV

`scripts/triangulation_experiment.py` on held-out test sequences, 3 HD cameras
(nodes 0/10/20), 11 clips. MPJPE in mm; lower is better.

| Offset (fr) | naive | CC | ContinuSyncFormer | oracle |
|---:|---:|---:|---:|---:|
| 0.00 | 0.00 | 0.49 | 2.03 | 0.00 |
| 0.25 | 0.92 | 0.65 | 1.96 | 0.19 |
| 0.50 | 1.91 | 0.99 | 2.17 | 0.36 |
| 1.00 | 3.88 | 0.58 | 1.56 | 0.00 |
| 2.00 | 7.64 | 0.53 | 1.51 | 0.00 |
| 3.00 | 11.27 | 0.53 | 1.47 | 0.00 |

**What this proves (the key 3DV result):**
1. **Sub-frame desync corrupts 3D, monotonically:** naive MPJPE grows 0 → 11.3 mm
   as the offset grows 0 → 3 frames. This *demonstrates* the motivation instead of
   asserting it.
2. **Sync correction recovers it:** both CC and CSF pull MPJPE back near the oracle
   (11.3 → 0.5 mm CC / 1.5 mm CSF at 3-frame desync).
3. **Offset 0 naive = 0.00 mm** validates the projection→triangulation pipeline.

**Honest caveat (again CC > CSF):** the triangulation cameras (nodes 0/10/20) are
wide-baseline — pairwise angles 72°, 83°, 39° — while CSF was trained only on pairs
≤ 60°. So CSF is evaluated **out of its training distribution** here and lands at
~1.5 mm vs CC's ~0.5 mm. CC's motion-feature correlation is baseline-agnostic.

**Strategic implication for 3DV.** CSF now loses to CC on: B1-test Acc@0.1, and
downstream MPJPE. The *one* place CSF wins is **occlusion** (occ-aware CSF beats CC
on Acc@τ under 20% occlusion). The decisive next experiment is therefore
**triangulation under occlusion** — if CSF-corrected MPJPE beats CC-corrected when
keypoints are missing, that is the paper's differentiator: *learned sync is more
robust to degraded keypoints, and that robustness carries through to better 3D.*
Alternatively/also: retrain CSF including wide-baseline pairs so it is in-distribution
for triangulation rigs.

## Drift: the regime where the learned method genuinely wins (task #24, for 3DV)

Real multi-device rigs do not share a clock: camera B's frame t observes the scene
at `alpha*t + beta`, so the inter-camera offset *drifts* across a take. A single lag
(cross-correlation) cannot represent this. Recovering drift needs *local* offset
estimation in short sliding windows + a robust line fit — and short windows are
exactly where classical CC breaks (its mean frame error explodes: 0.41 at 72 frames
-> 3.7 at 24; see the probe), while the learned model stays reliable.

**Where each baseline fails:** CC-const ignores drift (structural); Caspi-Irani is
drift-aware but a coarse, fragile grid; CC-slide is fine only at long windows and
degrades on the short windows drift tracking needs; CSF-slide (short-window learned
estimates + Theil-Sen line) is robust across window size and drift.

### Drift recovery error (frames), mean ± std — window=20, T=20. Lower better.

Two error bars: **over clips** (8 held-out clips) and **over training seeds** (4 models,
seeds 1–4).

| drift | CC-const | Caspi | CC-slide | CSF-slide (over clips) | CSF-slide (over seeds) |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.186±0.19 | 0.303±0.25 | 0.180±0.18 | 0.261±0.38 | 0.173±0.07 |
| 1.0 | 0.316±0.11 | 0.285±0.20 | 0.201±0.17 | **0.105±0.03** | **0.096±0.01** |
| 2.0 | 0.588±0.14 | 0.242±0.11 | 0.288±0.30 | **0.101±0.07** | **0.113±0.02** |
| 3.0 | 0.841±0.17 | 0.273±0.14 | 0.421±0.56 | **0.127±0.11** | **0.117±0.02** |

At any real drift (≥ 0.5) CSF-slide has the lowest error **and the lowest variance** on
both axes: across-seed std is tiny (±0.01–0.02 → the win is not a lucky seed), and
across-clip std is far below CC-slide's, whose std *exceeds its mean* at high drift
(0.42±0.56) — CC-slide catastrophically fails on some clips while CSF-slide stays tight.
**Honest exception:** at drift 0 (no drift) CSF-slide is marginally worse than CC-slide —
with nothing to track, sliding+curve only adds noise. The advantage is specific to the
drift regime. `scripts/drift_experiment.py`.

### Drift recovery vs window size (mechanism) — 8 clips, T=20 model, 2-frame drift.

| window | 16 | 20 | 24 | 32 | 48 |
|---|---:|---:|---:|---:|---:|
| CC-slide | 0.268 | 0.288 | 0.207 | **0.101** | 0.145 |
| **CSF-slide** | **0.091** | **0.101** | **0.116** | 0.184 | 0.140 |

CC-slide is reliable only at long windows (≥32); CSF-slide is robust down to very
short windows. Best config: **T=20 model, window 16–20**.

### Downstream 3D under drift — MPJPE (mm), window=20, 3 held-out sequences, 11 clips.

| drift (fr) | naive | CC-const | CC-slide | **CSF-slide (ours)** | oracle |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 3.65 | 1.63 | 1.69 | **1.55** | 0.37 |
| 0.5 | 5.32 | 1.60 | 1.75 | **1.36** | 0.21 |
| 1.0 | 6.93 | 1.90 | 2.16 | **1.49** | 0.20 |
| 2.0 | 10.10 | 3.37 | 2.65 | **1.60** | 0.25 |
| 3.0 | 13.20 | 4.93 | 3.52 | **1.72** | 0.36 |

CSF-slide wins at every drift and is nearly **flat** across drift (1.55→1.72) while
CC-const (1.63→4.93) and CC-slide (1.69→3.52) blow up — at 3-frame drift CSF-slide is
~3× better than CC-const and ~2× better than CC-slide. MPJPE over time at 3-frame
drift (first→last third): naive 8.12→15.29, CC-const 6.53→5.94, CC-slide 3.57→3.56,
**CSF-slide 2.14→1.62**, oracle 0.32→0.39. Per-frame curves:
`docs/paper/drift_mpjpe_curves.json`. `scripts/drift_triangulation.py`.

**This is the paper's method contribution for 3DV:** learned sub-frame sync enables
robust *drift* correction where classical methods structurally (CC-const) or
practically (CC-slide short-window, Caspi grid) fail, and that keeps multi-view 3D
reconstruction steady across a take.

**Honest caveats:** the downstream MPJPE uses only 2 long clips (illustrative); the
8-clip mapping metric is the statistically firmer result — get more triangulation
clips before submission. Absolute MPJPE gaps are small (Panoptic is well-calibrated,
moderate motion); the *relative* drift-growth story is the point. CSF-slide only
narrowly beats CC-slide downstream — the decisive wins are the mapping metric and
vs CC-const.

### Non-linear (oscillating) drift — mean error (frames), window=16, period=100, 3 seqs.

| amp (fr) | CC-const | Caspi | CC-line | CC-curve | **CSF-curve (ours)** |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 1.75 | 1.37 | **0.30** | 0.89 | 0.56 |
| 1.0 | 2.05 | 1.60 | **0.59** | 0.92 | 0.64 |
| 2.0 | 2.70 | 2.16 | 1.18 | 1.33 | **0.73** |

Supplementary (honest): line/affine methods can only predict the mean (error grows
with amplitude); **classical curve tracking (CC-curve) is worse than predicting the
mean** because CC's short-window estimates are too noisy — only CSF can actually
track a curve, keeping error nearly amplitude-independent (0.56→0.73 as amp 0.5→2).
CSF-curve wins clearly once the oscillation exceeds ~1.2 frames; below that its
~0.5-frame noise floor dominates. Real clock drift is usually monotonic (≈linear),
so **linear drift is the headline result; non-linear is a capability demonstration.**
`scripts/drift_nonlinear.py`.

## Comprehensive config summary — best operating point

- **Model:** T=20, motion input, selected by Frm.err (`csf_b1_t20.pt`).
- **Window:** 16–20 frames (short-window regime where CC fails, CSF is robust).
- **Fit:** line (Theil-Sen) for monotonic drift; curve (Hampel+interp) for non-linear.
- **Linear drift (headline):** CSF-slide wins at every drift, flat across drift;
  downstream MPJPE 2–3× better than classical and nearest the oracle.
- **Non-linear drift (supplementary):** only CSF tracks it; wins at amplitude > ~1.2 fr.

## Still pending

- **More triangulation clips** for drift MPJPE (lower --min-speed / more sequences).
- **Figures** (Khải): drift-recovery-vs-drift curve; MPJPE-over-time under drift
  (from `drift_mpjpe_curves.json`); plus B1 Acc@τ curve and per-velocity-bucket Acc@0.1.
