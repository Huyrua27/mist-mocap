"""WS1 regression tests. Runnable directly without pytest."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mist.benchmark import metrics, synthetic
from mist.benchmark.baselines import CaspiIrani, CrossCorrelation, DTW
from mist.benchmark.desync import make_sample, split_by_sequence
from mist.core.types import KeypointSequence
from mist.panoptic.loader import load_sequence, project_to_2d


def test_projection_reference_formula():
    points = np.array([[[0.2, -0.1, 2.0], [0.5, 0.3, 3.0]]])
    K = np.array([[1000.0, 5.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    R = np.eye(3)
    t = np.array([0.1, -0.2, 0.5])
    distortion = np.array([0.01, -0.001, 0.0002, -0.0003, 0.00001])

    camera = points.reshape(-1, 3) @ R.T + t
    x = camera[:, 0] / camera[:, 2]
    y = camera[:, 1] / camera[:, 2]
    r2 = x * x + y * y
    radial = 1 + distortion[0] * r2 + distortion[1] * r2**2 + distortion[4] * r2**3
    xd = x * radial + 2 * distortion[2] * x * y + distortion[3] * (r2 + 2 * x * x)
    yd = y * radial + distortion[2] * (r2 + 2 * y * y) + 2 * distortion[3] * x * y
    expected = np.column_stack(
        [K[0, 0] * xd + K[0, 1] * yd + K[0, 2], K[1, 1] * yd + K[1, 2]]
    ).reshape(1, 2, 2)
    np.testing.assert_allclose(
        project_to_2d(points, K, R, t, distortion), expected, atol=1e-10
    )


def test_loader_minimal_fixture():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        calibration = {
            "cameras": [
                {
                    "panel": 0,
                    "node": 0,
                    "name": "00_00",
                    "K": [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
                    "R": np.eye(3).tolist(),
                    "t": [0.0, 0.0, 0.0],
                    "distCoef": [0.0] * 5,
                }
            ]
        }
        (root / "calibration_fixture.json").write_text(
            json.dumps(calibration), encoding="utf-8"
        )
        pose_dir = root / "hdPose3d_stage1_coco19"
        pose_dir.mkdir()
        for frame in range(5):
            joints = []
            for joint in range(19):
                joints.extend([frame + joint * 0.1, joint * 0.2, 10.0, 1.0])
            payload = {"bodies": [{"id": 7, "joints19": joints}]}
            (pose_dir / f"body3DScene_{frame:08d}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
        loaded = load_sequence(
            str(root), camera_keys=[(0, 0)], person_id=None, fps=30.0
        )
        assert set(loaded) == {"00_00"}
        assert loaded["00_00"].xy.shape == (5, 19, 2)
        np.testing.assert_allclose(loaded["00_00"].timestamps, np.arange(5) / 30.0)
        np.testing.assert_allclose(loaded["00_00"].xy[0, 0], [50.0, 40.0])


def test_desync_crops_boundaries_and_preserves_lineage():
    time = np.arange(30, dtype=np.float64)
    xy = np.stack([time, time**2], axis=-1)[:, None, :]
    seq = KeypointSequence(xy, 30.0, name="source-A")
    sample = make_sample(seq, 2.5, source_sequence="source-A")
    assert sample.a.T == 27
    assert sample.a.T == sample.b.T
    assert sample.meta["source_sequence"] == "source-A"
    expected = (np.arange(27) + 2.5) ** 2
    np.testing.assert_allclose(sample.b.xy[:, 0, 1], expected, atol=1e-9)
    assert not np.all(sample.b.xy[-2] == sample.b.xy[-1])


def test_true_sequence_split_and_single_sequence_rejection():
    samples = []
    for sequence_index in range(4):
        seq = synthetic.make_trajectory(T=50, seed=sequence_index)
        for offset in (-1.0, 1.0):
            samples.append(
                make_sample(
                    seq,
                    offset,
                    source_sequence=f"sequence-{sequence_index}",
                )
            )
    train, test = split_by_sequence(samples, test_ratio=0.25, seed=3)
    train_sources = {sample.meta["source_sequence"] for sample in train}
    test_sources = {sample.meta["source_sequence"] for sample in test}
    assert train_sources.isdisjoint(test_sources)
    try:
        split_by_sequence(samples[:2])
    except ValueError as error:
        assert "at least two" in str(error)
    else:
        raise AssertionError("single-source split should fail closed")


def test_metric_boundaries_and_per_sample_fps():
    assert metrics.accin([0.1], [0.0], 0.1) == 1.0
    assert metrics.mae_ms([1.0, 1.0], [0.0, 0.0], [25.0, 50.0]) == 30.0
    expected_rmse = np.sqrt((40.0**2 + 20.0**2) / 2)
    assert abs(metrics.rmse_ms([1.0, 1.0], [0.0, 0.0], [25.0, 50.0]) - expected_rmse) < 1e-12


def test_velocity_edges_are_fitted_once():
    edges = metrics.fit_velocity_edges([1.0, 2.0, 3.0, 4.0, 100.0])
    pred = [0.0, 0.0, 0.0, 0.0]
    gt = [0.0, 0.0, 0.0, 0.0]
    buckets = metrics.by_velocity_bucket(pred, gt, [0.5, 2.5, 3.5, 200.0], 30.0, edges)
    assert sum(bucket["n"] for bucket in buckets.values()) == 4


def _rotated_shift_pair(dt=1.7):
    seq = synthetic.make_trajectory(T=140, fps=30.0, seed=11, speed=1.8)
    sample = make_sample(seq, dt, source_sequence="synthetic-rotation")
    angle = 0.8
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    transformed = sample.b.xy @ rotation.T * 1.7 + np.array([120.0, -35.0])
    sample.b = KeypointSequence(transformed, sample.b.fps, name="rotated-view")
    return sample


def test_cross_correlation_cross_view_transform():
    sample = _rotated_shift_pair()
    prediction = CrossCorrelation(max_lag=10).predict(sample.a, sample.b).dt_frames
    assert abs(prediction - sample.dt_gt_frames) < 0.15


def test_dtw_subframe_offset():
    sample = _rotated_shift_pair(dt=-2.25)
    prediction = DTW(
        oversample_factor=4, max_warp_frames=8, max_frames=120
    ).predict(sample.a, sample.b).dt_frames
    assert abs(prediction - sample.dt_gt_frames) < 0.8


def _analytic_pose(times: np.ndarray, joints: int = 9) -> np.ndarray:
    output = np.empty((len(times), joints, 2), dtype=np.float64)
    for joint in range(joints):
        output[:, joint, 0] = np.sin(
            (0.04 + joint * 0.003) * times
            + (0.00012 + joint * 0.00001) * times**2
            + joint
        )
        output[:, joint, 1] = np.cos(
            (0.07 + joint * 0.002) * times
            - (0.00009 + joint * 0.000015) * times**2
            - joint * 0.4
        )
    return output


def test_affine_alpha_beta_recovery():
    time = np.arange(180, dtype=np.float64)
    alpha, beta = 1.015, 2.4
    a = KeypointSequence(_analytic_pose(time), 30.0, name="affine-A")
    b = KeypointSequence(_analytic_pose(alpha * time + beta), 30.0, name="affine-B")
    estimate = CaspiIrani(
        alpha_range=(0.99, 1.04), alpha_steps=21, max_lag=8
    ).estimate(a, b)
    assert abs(estimate.alpha - alpha) <= 0.005
    assert abs(estimate.beta - beta) < 0.5


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"ALL {len(tests)} WS1 TESTS PASS")
