# Real-data drift pipeline (Task #24)

Validate the drift-correction method on **real, independently-clocked cameras**, with
**flash ground truth**. Core claim (sync error vs flash GT) needs only steps 0–3; the
3D reprojection check (steps 4–5) is optional.

## 0. Install

```bash
pip install -r requirements-realdata.txt   # mediapipe + opencv-contrib (aruco)
```

## Capture recap (what the footage must contain)

- 3–5 **independent** cameras (different phones + webcams) on tripods, arc ~120°,
  ~30° apart, all seeing the full body + the flash + (briefly) the ChArUco board.
- Each device: **lock FPS=30, exposure, focus, WB** (Open Camera on Android). Do **not**
  time-sync the devices — the drift between their clocks is what we measure.
- Flash source (a 6th device / LED torch, **not** a screen) bounced off the ceiling,
  **pulsed every ~20 s** through the take (start + periodic + end).
- Start each take with a **waving ChArUco** segment and one **static-board** moment
  (all cameras see the same still board) for calibration.

## 1. Pick one flash ROI per camera (static cameras → reuse across the take)

```bash
python tools/sync_groundtruth.py pick-roi --src cam0.mp4   # writes cam0.mp4.roi.json
```

## 2. Flash → ground-truth drift curve

```bash
python scripts/realdata/detect_flash_gt.py --videos cam0.mp4 cam1.mp4 cam2.mp4 \
    --ref cam0.mp4 --n-flashes 18 --out data/realdata/s1/gt_drift.json
```

Check every camera reports the same pulse count and SNR ≥ ~4. `drift(fr/take)` shows the
real accumulated clock drift per camera.

## 3. 2D pose + sync comparison (the headline real-data result)

```bash
python scripts/realdata/extract_pose2d.py --videos cam0.mp4 cam1.mp4 cam2.mp4 \
    --out-dir data/realdata/s1/pose
python scripts/realdata/run_realdata_sync.py --pose-dir data/realdata/s1/pose \
    --gt data/realdata/s1/gt_drift.json --checkpoint checkpoints/csf_b1_t20.pt --window 20
```

Reports mean |estimated − flash-GT| drift error (frames) per camera for CC-const /
CC-slide / CSF-slide — the real-data analogue of Table 2.

## 4–5. Optional 3D: calibrate + reprojection check

```bash
python scripts/realdata/calibrate_charuco.py --videos cam0.mp4 cam1.mp4 cam2.mp4 \
    --squares-x 5 --squares-y 7 --square-len 0.035 --marker-len 0.026 \
    --static-frame 300 --out data/realdata/s1/calib.json
python scripts/realdata/triangulate_realdata.py --pose-dir data/realdata/s1/pose \
    --calib data/realdata/s1/calib.json --checkpoint checkpoints/csf_b1_t20.pt --window 20
```

Set `--squares-x/-y`, `--square-len`, `--marker-len` to **your** board; `--static-frame`
to a frame index in the static-board moment. Lower reprojection error after correction =
sync makes the multi-view geometry consistent.

## Notes

- `data/` is git-ignored — raw footage and outputs stay local.
- The model checkpoint is reproducible: retrain with the command in the top-level README.
- Same window/model as the synthetic drift experiment, so numbers are comparable.
