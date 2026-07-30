# WS1 — CMU Panoptic Synchronization Benchmark

This directory contains the compact verification package for **WS1**, the CMU Panoptic keypoint-domain synchronization benchmark implemented in this repository.

It is intentionally separate from the repository-level `README.md`: this file documents one completed WS1 experiment, its evidence, reproduction path, and limitations.

## Status

**Final status: Tasks 1–10 passed.**

The final run provides:

- six validated Panoptic sequence sources;
- immutable sequence-level train/validation/test splits;
- stable person-track handling;
- deterministic clip, camera-pair, offset, and affine-warp generation;
- train-only fitted and frozen velocity buckets;
- complete held-out evaluation for Cross-Correlation, DTW, and Caspi–Irani;
- a calibrated projection overlay on a real HD frame;
- passing core, WS1 regression, and WS1 finalization test suites.

No published number from InSynFormer or SyncTrack4D is treated as directly comparable to this benchmark because their datasets, inputs, protocols, and metric definitions differ.

## Final run

| Field | Value |
|---|---|
| Run ID | `ws1-3926ab3ccf53` |
| Config hash | `3926ab3ccf53d9eebe9e12e849a7f521f3cdc7b5fcf8d5f71e6b2479c4cb370c` |
| Split hash | `6c88daf12ccf0754e18891f2c928f57073ac2d7ecf6d48ee50f4f5f1bb21ec97` |
| Frozen method config hash | `61fff05943c870fb95d4a862e61c94bc30c520e08c1fa9cf203962cbbef197f4` |
| Physical sequence sources | 6 |
| Stable track segments | 45 |
| Deterministic base clips | 16 |
| Frozen camera pairs | 18 |
| Generated variants | 768 |
| Test base clips | 6 |
| Unique held-out test samples | 288 |
| Prediction rows | 540 |
| Explicit failed rows | 0 |

## Dataset split

| Split | Sequences |
|---|---|
| Train | `171204_pose1_sample`, `160906_band4`, `160906_band1` |
| Validation | `160906_ian5` |
| Test | `160422_haggling1`, `160226_haggling1` |

Splitting is performed by source sequence before person-track, clip, camera-pair, offset, or affine-warp variants are generated. All descendants inherit the original split and lineage.

`171204_pose1_sample` is a compact subset of the parent Panoptic capture. It is retained under its current fixture identifier in this run; it must not be interpreted as an additional independent physical capture.

## Benchmark protocols

### Protocol A — Fractional temporal offset

The second trajectory is generated with:

```text
t_B = t_A + beta
alpha = 1
```

The held-out test uses balanced signed, zero, integer, and fractional offsets:

```text
beta ∈ {-7.5, -3.0, -0.5, 0.0, 0.25, 2.0, 7.5} frames
```

Compatible methods:

- Cross-Correlation with guarded parabolic refinement;
- bounded oversampled DTW;
- Caspi–Irani in offset-only mode.

### Protocol B — Affine temporal mapping

The second trajectory is generated with:

```text
t_B = alpha * t_A + beta
```

The held-out test covers:

```text
alpha ∈ {0.96, 1.00, 1.04}
beta  ∈ {-2.0, 0.0, 2.0} frames
```

Caspi–Irani is evaluated using alpha error, beta error, and error over the complete estimated temporal mapping—not beta error alone.

All interpolation is restricted to common valid support. Endpoint repetition and out-of-domain extrapolation are not allowed.

## Velocity buckets

Velocity is defined as the median finite displacement of COCO-19 joints between adjacent 3D world-coordinate frames, multiplied by the stream FPS.

- Unit: `cm/s`
- Body-scale normalization: none
- Fit unit: unique training base clips
- Validation/test fitting: prohibited

Frozen internal edges:

```text
[4.803012, 6.241352, 8.574502]
```

Buckets are reported as `Q1`–`Q4`. The held-out test contains no Q4 motion, so this run makes no empirical performance claim for Q4.

## Held-out results

### Protocol A

| Method | Samples | Success | Mean frame error | Approx. MAE |
|---|---:|---:|---:|---:|
| DTW | 126 | 126 | 0.00893 frame | 0.298 ms |
| Cross-Correlation | 126 | 126 | 0.04692 frame | 1.566 ms |
| Caspi–Irani | 126 | 126 | 0.24256 frame | 8.093 ms |

### Protocol B

| Method | Samples | Success | Mean temporal-mapping MAE |
|---|---:|---:|---:|
| Caspi–Irani | 162 | 162 | 0.03490 frame |

Detailed per-sequence, per-velocity-bucket, per-offset-band, per-alpha-band, runtime, and failure statistics are stored in the generated summary artifacts.

These values are benchmark results for this exact deterministic protocol. They are not claims about all CMU Panoptic motion, pixel-domain video synchronization, or state-of-the-art performance.

## Projection verification

`projection_overlay.png` is generated from:

| Field | Value |
|---|---|
| Sequence | `171204_pose1_sample` |
| Camera | `00_00` |
| Frame | `39` |
| Person ID | `0` |
| Resolution | `1920 × 1080` |
| In-frame joints | `17 / 19` |
| Projection | calibrated, distortion-aware |
| Manual pixel correction | none |

The pose frame index and sequentially decoded HD video-frame index both equal 39. The frame is selected deterministically by maximum in-frame joint count, confidence sum, and then lowest frame index.

## Verification artifacts

Expected files in this directory:

| File | Purpose |
|---|---|
| `README.md` | This overview |
| `acceptance_checklist.md` | Final status of Tasks 1–10 |
| `data_validation_report.md` | Sequence, frame, calibration, and track validation |
| `summary_metrics.md` | Human-readable benchmark report |
| `summary_metrics.json` | Machine-readable grouped metrics |
| `resolved_config.yaml` | Fully resolved final experiment configuration |
| `projection_overlay.png` | Real-HD-frame projection evidence |
| `projection_overlay_metadata.json` | Projection provenance and coordinates |
| `test_results.txt` | Recorded test commands and outputs |
| `reference_numbers.md` | Published context numbers and citations |

Large or reproducible intermediate outputs should normally remain outside Git, including raw Panoptic data, HD video, downloaded archives, caches, `predictions.csv`, generated sample manifests, and full runtime directories.

## Reproduction

From the repository root:

```powershell
python scripts/finalize_ws1.py resolve
python scripts/finalize_ws1.py prepare
python scripts/finalize_ws1.py evaluate
$env:PYTHONPATH=(Resolve-Path 'data\ws1_runtime').Path
python scripts/finalize_ws1.py overlay
python scripts/run_ws1_test_suites.py
python scripts/finalize_ws1.py report
```

The downloader is intentionally minimal: calibration and COCO-19 body poses are downloaded for benchmark sequences, while only one HD video is needed for projection verification.

## Test status

The recorded final run passed:

- Python compilation for `mist`, `scripts`, and `tests`;
- all existing core tests;
- all 9 WS1 regression tests;
- all 21 WS1 finalization tests.

The finalization tests cover deterministic manifests and sample IDs, sequence lineage, train-only velocity fitting, bounded interpolation, downloader resume and repair, archive safety, invalid calibration and pose rejection, CC/DTW edge cases, affine controls, complete aggregation, and failure preservation.

## Scope and limitations

This result is sufficient to close WS1 as an engineering and integration deliverable.

The held-out benchmark contains six unique test base clips derived from two haggling sequences. Camera, offset, and affine variants increase protocol coverage but are not independent motion observations. Therefore:

- do not interpret 288 generated samples as 288 independent physical motions;
- do not claim generalization over all Panoptic activities;
- do not claim performance in Q4, which is absent from the held-out set;
- do not claim that DTW is universally superior on CMU Panoptic;
- do not compare the numbers directly with published video-domain benchmarks.

A stronger internal benchmark would use roughly 30–50 unique held-out base clips over 4–6 sequences and at least three activity types, with coverage in all four frozen velocity buckets. Paper-grade claims would require a larger, predeclared protocol, more held-out activities, and uncertainty or repeated-split analysis.

## Acceptance summary

WS1 is accepted when the checked-in evidence confirms:

1. minimal validated Panoptic data coverage;
2. real calibrated HD projection;
3. bounded fractional desynchronization;
4. leakage-safe sequence splitting;
5. complete mixed-FPS evaluation;
6. train-only frozen velocity buckets;
7. complete held-out Cross-Correlation evaluation;
8. complete held-out DTW evaluation;
9. affine recovery below, equal to, and above unit temporal scale;
10. corrected published-reference metadata with a non-comparability disclaimer.

The final artifact set satisfies these requirements.
