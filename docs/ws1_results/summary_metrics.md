# WS1 Final Held-Out Benchmark

## 1. Scope

Final deterministic CMU Panoptic keypoint-domain synchronization benchmark.
This protocol differs from the published InSynFormer and SyncTrack4D protocols;
their reference numbers are context only and are not directly comparable.

## 2. Dataset sequences

| Sequence | Split | Activity | Pose files | Valid pose frames | Invalid pose files | Valid tracks |
|---|---|---|---:|---:|---:|---:|
| 171204_pose1_sample | train | range_of_motion | 101 | 101 | 0 | 1 |
| 160906_band4 | train | band_multi_person | 9840 | 9840 | 0 | 3 |
| 160906_band1 | train | band_multi_person | 7332 | 1772 | 0 | 3 |
| 160906_ian5 | validation | solo_articulation | 2872 | 2871 | 0 | 2 |
| 160422_haggling1 | test | haggling_multi_person | 13579 | 11675 | 1 | 18 |
| 160226_haggling1 | test | haggling_multi_person | 11465 | 10407 | 0 | 18 |

## 3. Train/validation/test split

Assignments are immutable by physical sequence ID. Split hash: `6c88daf12ccf0754e18891f2c928f57073ac2d7ecf6d48ee50f4f5f1bb21ec97`.

## 4. Person-track statistics

45 accepted and 0 rejected stable Panoptic-ID track segments.

## 5. Camera-pair selection

18 frozen pairs selected from calibration-distance quantiles (small/medium/wide), without prediction access.

## 6. Clip/sample generation

16 deterministic base clips and 768 generated variants.

## 7. Velocity definition

Median finite COCO-19 joint displacement between adjacent 3D world-coordinate
frames, multiplied by per-stream FPS; unit cm/s; no body-scale normalization.

## 8. Frozen velocity edges

`[4.803012, 6.241352, 8.574501999999999]` fitted on unique training base clips only.

## 9. Protocol A definition

`t_B = t_A + beta`, alpha fixed to 1; balanced deterministic beta controls.

## 10. Protocol B definition

`t_B = alpha*t_A + beta`, including alpha below/equal/above 1 and signed beta.

## 11. Method configurations

Frozen method config hash: `61fff05943c870fb95d4a862e61c94bc30c520e08c1fa9cf203962cbbef197f4`.

## 12. Overall results

| Method | Protocol | n | Success | Failure | Primary mean error |
|---|---|---:|---:|---:|---:|
| caspi_irani | affine | 162 | 162 | 0 | 0.0348996560634584 |
| caspi_irani | offset | 126 | 126 | 0 | 0.2425559510146729 |
| cross_correlation | offset | 126 | 126 | 0 | 0.046924744754846766 |
| dtw | offset | 126 | 126 | 0 | 0.008928571428571428 |

## 13. Macro-by-sequence results

```json
{
  "caspi_irani/affine": {
    "sequence_count": 2,
    "macro_mapping_mae_frames_mean": 0.03489965606345839
  },
  "caspi_irani/offset": {
    "sequence_count": 2,
    "macro_frame_error_mean": 0.24255595101467292
  },
  "cross_correlation/offset": {
    "sequence_count": 2,
    "macro_frame_error_mean": 0.04692474475484677
  },
  "dtw/offset": {
    "sequence_count": 2,
    "macro_frame_error_mean": 0.008928571428571428
  }
}
```

## 14. Per-velocity-bucket results

Available in `summary_metrics.json` groups with dimension `velocity_bucket`.

## 15. Per-offset-band results

Available in groups with dimension `beta_band`.

## 16. Per-alpha-band results

Available in groups with dimension `alpha_band`.

## 17. Failure analysis

0 explicit failed/insufficient-motion rows; no row was dropped.
The held-out distribution has zero Q4 samples; Q4 is retained explicitly with
`n=0` in `summary_metrics.json` rather than hidden or relabelled.

## 18. Runtime

Mean, median, and p90 runtime per method/protocol are in `summary_metrics.json`.

## 19. Projection verification

The overlay uses one sequentially decoded real HD frame, calibrated distortion-aware
projection, confidence masks, and no manual pixel correction.

## 20. Test outputs

See `test_results.txt`.

## 21. Limitations

This is a six-sequence keypoint-domain benchmark with bounded clip and camera-pair
sampling. It is not a video-pixel synchronization benchmark or a SOTA claim.

## 22. Exact reproduction commands

```powershell
python scripts/finalize_ws1.py resolve
python scripts/finalize_ws1.py prepare
python scripts/finalize_ws1.py evaluate
$env:PYTHONPATH=(Resolve-Path 'data\ws1_runtime').Path
python scripts/finalize_ws1.py overlay
python scripts/run_ws1_test_suites.py
python scripts/finalize_ws1.py report
```
