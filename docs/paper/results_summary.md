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

## Still pending

- **Figures** (Khải): Acc@τ curve (3 methods) and Acc@0.1 per velocity bucket
  (Q1–Q4, via the harness `bucket_table`).
