# Data Validation Report

Config hash: `3926ab3ccf53d9eebe9e12e849a7f521f3cdc7b5fcf8d5f71e6b2479c4cb370c`.

| Sequence | Split | Activity | Pose files | Valid pose frames | Invalid pose files | Valid tracks |
|---|---|---|---:|---:|---:|---:|
| 171204_pose1_sample | train | range_of_motion | 101 | 101 | 0 | 1 |
| 160906_band4 | train | band_multi_person | 9840 | 9840 | 0 | 3 |
| 160906_band1 | train | band_multi_person | 7332 | 1772 | 0 | 3 |
| 160906_ian5 | validation | solo_articulation | 2872 | 2871 | 0 | 2 |
| 160422_haggling1 | test | haggling_multi_person | 13579 | 11675 | 1 | 18 |
| 160226_haggling1 | test | haggling_multi_person | 11465 | 10407 | 0 | 18 |

All six explicitly configured sequences passed calibration, camera-matrix,
frame-index, COCO-19 schema, stable-ID track, HD-camera, and minimum-length gates.
No sequence was silently substituted.

## Explicit invalid raw observations

- `160422_haggling1`: [{"path": "body3DScene_00011542.json", "reason": "JSONDecodeError: Expecting value: line 1 column 1 (char 0)"}]

The single official empty JSON is represented as a missing observation and may
only be bridged by the configured bounded gap policy; it is not silently parsed
as a valid pose and does not affect the accepted long stable tracks.
