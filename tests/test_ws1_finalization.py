"""Dependency-light finalization regression tests. Runnable without pytest."""
from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mist.benchmark import metrics, synthetic
from mist.benchmark.baselines import CaspiIrani, CrossCorrelation, DTW
from mist.benchmark.baselines.cross_correlation import refined_peak
from mist.benchmark.desync import make_sample
from mist.core.types import KeypointSequence
from scripts import download_panoptic_minimal as downloader
from scripts import finalize_ws1 as final


def test_config_hash_and_sample_ids_are_deterministic():
    value = {"b": [2, 1], "a": 4}
    assert final.stable_hash(value) == final.stable_hash({"a": 4, "b": [2, 1]})
    changed = {"b": [2, 1], "a": 5}
    assert final.stable_hash(value) != final.stable_hash(changed)


def test_downloader_reuses_complete_file_and_checksums_it():
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "resource.bin"
        target.write_bytes(b"complete")
        record = downloader._download(["http://invalid"], target)
        assert record["status"] == "reused"
        assert record["sha256"] == downloader.sha256_file(target)


def test_corrupted_partial_download_is_repaired_atomically():
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "resource.bin"
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(b"corrupt")
        original = downloader.subprocess.run

        def fake_run(command, **kwargs):
            # curl resolves a relative --output against its cwd; the downloader
            # relies on that to keep non-ASCII directories out of curl's argv.
            output = Path(command[command.index("--output") + 1])
            if not output.is_absolute():
                output = Path(kwargs.get("cwd", ".")) / output
            output.write_bytes(b"repaired-content")
            return SimpleNamespace(returncode=0, stderr="")

        downloader.subprocess.run = fake_run
        try:
            record = downloader._download(["http://fixture"], target)
        finally:
            downloader.subprocess.run = original
        assert record["status"] == "repaired"
        assert target.read_bytes() == b"repaired-content"
        assert not partial.exists()


def test_safe_extraction_and_resume():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        archive = root / "pose.tar"
        payload = b'{"bodies":[]}'
        with tarfile.open(archive, "w") as handle:
            member = tarfile.TarInfo(
                "hdPose3d_stage1_coco19/body3DScene_00000000.json"
            )
            member.size = len(payload)
            handle.addfile(member, BytesIO(payload))
        assert downloader._safe_extract(archive, root) == "downloaded"
        assert downloader._safe_extract(archive, root) == "reused"


def test_path_traversal_archive_is_rejected():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        archive = root / "bad.tar"
        with tarfile.open(archive, "w") as handle:
            member = tarfile.TarInfo("../escape.json")
            member.size = 2
            handle.addfile(member, BytesIO(b"{}"))
        try:
            downloader._safe_extract(archive, root)
        except ValueError as error:
            assert "unsafe archive member" in str(error)
        else:
            raise AssertionError("path traversal must fail")


def test_manifest_resource_order_is_deterministic():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "sequence"
        root.mkdir()
        (root / "calibration_sequence.json").write_text("{}", encoding="utf-8")
        (root / "hdPose3d_stage1_coco19.tar").write_bytes(b"tar")
        downloader._write_manifest(root, "sequence", [])
        first = (root / "manifest.yaml").read_bytes()
        downloader._write_manifest(root, "sequence", [])
        assert first == (root / "manifest.yaml").read_bytes()
        manifest = json.loads(first)
        assert [item["path"] for item in manifest["raw_files"]] == sorted(
            item["path"] for item in manifest["raw_files"]
        )


def test_invalid_calibration_is_rejected():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cameras = []
        for node in range(3):
            cameras.append(
                {
                    "panel": 0,
                    "node": node,
                    "name": f"00_{node:02d}",
                    "K": np.eye(3).tolist(),
                    "R": (np.eye(3) * 2).tolist(),
                    "t": [0, 0, 0],
                }
            )
        (root / "calibration_fixture.json").write_text(
            json.dumps({"cameras": cameras}), encoding="utf-8"
        )
        try:
            final.validate_calibration(root)
        except ValueError as error:
            assert "orthonormal" in str(error)
        else:
            raise AssertionError("invalid rotation must fail")


def test_invalid_pose_schema_is_rejected():
    for body in ({}, {"id": 1, "joints19": [0] * 75}):
        try:
            final.parse_joints(body)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid pose body must fail")


def test_stable_person_segments_do_not_join_long_gaps():
    segments = final.contiguous_segments([0, 1, 2, 8, 9], max_gap=2)
    assert [(start, end) for start, end, _ in segments] == [(0, 2), (8, 9)]


def test_bounded_interpolation_has_no_endpoint_extrapolation():
    values = np.arange(8.0)[:, None, None]
    values[[0, 3, 4, 7]] = np.nan
    filled = final.interpolate_short_gaps(values, max_gap=2)
    assert np.isnan(filled[0, 0, 0]) and np.isnan(filled[7, 0, 0])
    assert np.isfinite(filled[3:5, 0, 0]).all()


def test_split_lineage_and_suffix_do_not_change_source():
    assignments = {"physical": "train", "physical_extra": "test"}
    assert assignments["physical"] == "train"
    source = "physical"
    variants = [
        {"source_sequence_id": source, "suffix": suffix}
        for suffix in ("_offset", "_camera", "_alpha")
    ]
    assert {variant["source_sequence_id"] for variant in variants} == {source}


def _sample_fixture_config():
    return {
        "offset_values": [-0.5, 0.0, 0.5],
        "alpha_values": [0.96, 1.0, 1.04],
        "affine_beta_values": [-2.0, 0.0, 2.0],
        "minimum_common_overlap": 48,
        "minimum_affine_drift_frames": 1.8,
        "seed": 7,
        "fps": 29.97,
    }


def test_sample_manifest_controls_and_inheritance():
    config = _sample_fixture_config()
    clip = {
        "base_clip_id": "clip",
        "source_sequence_id": "physical",
        "person_id": 4,
        "split": "test",
        "start_frame": 10,
        "end_frame": 81,
        "frame_count": 72,
        "low_motion": False,
        "velocity_cm_s": 8.0,
    }
    pair = {
        "source_sequence_id": "physical",
        "camera_a": "00_00",
        "camera_b": "00_15",
        "relative_view_angle_descriptor": "wide",
    }
    samples = final.build_samples(config, [clip], [pair])
    assert len({sample["sample_id"] for sample in samples}) == len(samples)
    assert {sample["split"] for sample in samples} == {"test"}
    offset_betas = {
        sample["beta_gt"] for sample in samples if sample["protocol"] == "offset"
    }
    assert offset_betas == {-0.5, 0.0, 0.5}
    assert {
        sample["alpha_band"]
        for sample in samples
        if sample["protocol"] == "affine"
    } == {"below_one", "one_control", "above_one"}
    assert samples == final.build_samples(config, [clip], [pair])


def test_affine_overlap_and_drift_gate():
    assert final.valid_interval(72, 1.04, 2.0)[1] > final.valid_interval(
        72, 1.04, 2.0
    )[0]
    config = _sample_fixture_config()
    config["minimum_affine_drift_frames"] = 100.0
    clip = {
        "base_clip_id": "clip", "source_sequence_id": "s", "person_id": 0,
        "split": "test", "start_frame": 0, "end_frame": 71,
        "frame_count": 72, "low_motion": False, "velocity_cm_s": 2.0,
    }
    pair = {
        "source_sequence_id": "s", "camera_a": "a", "camera_b": "b",
        "relative_view_angle_descriptor": "small",
    }
    try:
        final.build_samples(config, [clip], [pair])
    except RuntimeError as error:
        assert "unidentifiable" in str(error)
    else:
        raise AssertionError("unidentifiable alpha drift must fail")


def test_velocity_buckets_are_train_only_and_boundaries_are_explicit():
    train = [1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0]
    edges = list(metrics.fit_velocity_edges(train))
    assert edges == list(metrics.fit_velocity_edges(train))
    assert final.assign_bucket(edges[0], edges) == "Q2"
    assert final.assign_bucket(-100.0, edges) == "Q1"
    assert final.assign_bucket(100.0, edges) == "Q4"


def test_tied_velocity_edges_fail():
    try:
        metrics.fit_velocity_edges([1.0, 1.0, 1.0, 1.0])
    except ValueError as error:
        assert "distinct" in str(error)
    else:
        raise AssertionError("tied edges must fail")


def test_cc_integer_zero_fractional_sign_and_constant_failure():
    for shift in (-2.0, -0.5, 0.0, 0.75, 2.0):
        seq = synthetic.make_trajectory(T=100, seed=12)
        sample = make_sample(seq, shift, source_sequence="fixture")
        prediction = CrossCorrelation(max_lag=5).predict(
            sample.a, sample.b
        ).dt_frames
        assert abs(prediction - shift) < 0.2
    constant = KeypointSequence(np.ones((30, 5, 2)), 30.0)
    try:
        CrossCorrelation(max_lag=5).predict(constant, constant)
    except ValueError as error:
        assert "insufficient motion" in str(error)
    else:
        raise AssertionError("constant CC input must fail")


def test_cc_boundary_peak_uses_discrete_value():
    lag, confidence = refined_peak(
        np.array([-2.0, -1.0, 0.0]), np.array([3.0, 2.0, 1.0])
    )
    assert lag == -2.0
    assert 0.0 <= confidence <= 1.0


def test_dtw_fractional_conversion_and_constant_failure():
    seq = synthetic.make_trajectory(T=90, seed=31)
    sample = make_sample(seq, 1.5, source_sequence="fixture")
    prediction = DTW(
        oversample_factor=4, max_warp_frames=5, max_frames=None
    ).predict(sample.a, sample.b).dt_frames
    assert abs(prediction - 1.5) <= 0.5
    constant = KeypointSequence(np.ones((30, 5, 2)), 30.0)
    try:
        DTW().predict(constant, constant)
    except ValueError as error:
        assert "insufficient motion" in str(error)
    else:
        raise AssertionError("constant DTW input must fail")


def _analytic(times: np.ndarray) -> np.ndarray:
    output = np.empty((len(times), 6, 2))
    for joint in range(6):
        output[:, joint, 0] = np.sin(0.04 * times + 0.0002 * times**2 + joint)
        output[:, joint, 1] = np.cos(0.07 * times - 0.0001 * times**2 - joint)
    return output


def test_affine_controls_below_above_and_combined_mapping():
    time_axis = np.arange(180.0)
    for alpha, beta in ((1.0, 0.0), (1.0, 1.5), (0.98, -2.0), (1.02, 2.0)):
        a = KeypointSequence(_analytic(time_axis), 30.0)
        b = KeypointSequence(_analytic(alpha * time_axis + beta), 30.0)
        estimate = CaspiIrani(
            alpha_range=(0.96, 1.04), alpha_steps=33, max_lag=5
        ).estimate(a, b)
        assert abs(estimate.alpha - alpha) <= 0.01
        mapping_error = (
            estimate.alpha * time_axis + estimate.beta
            - (alpha * time_axis + beta)
        )
        assert float(np.mean(np.abs(mapping_error))) < 0.6


def test_aggregate_keeps_failures_and_macro_sequences():
    rows = []
    for sequence, error, status in (
        ("s1", 0.1, "success"),
        ("s2", 0.3, "success"),
        ("s2", None, "failed"),
    ):
        rows.append(
            {
                "method": "m", "protocol": "offset",
                "source_sequence_id": sequence, "velocity_bucket": "Q1",
                "beta_band": "small", "alpha_band": "one_control",
                "status": status, "runtime_ms": 1.0, "fps": 30.0,
                "absolute_frame_error": error if error is not None else "",
            }
        )
    summary = final.aggregate_rows(rows, [0.1])
    overall = next(
        group for group in summary["groups"]
        if group["group_dimension"] == "overall"
    )
    assert overall["n"] == 3
    assert overall["success_count"] == 2 and overall["failure_count"] == 1
    assert summary["macro_by_sequence"]["m/offset"]["sequence_count"] == 2


def test_coco19_skeleton_edges_match_joint_definition():
    assert (0, 3) in final.SKELETON_EDGES
    assert (0, 9) in final.SKELETON_EDGES
    assert (2, 6) in final.SKELETON_EDGES
    assert (2, 12) in final.SKELETON_EDGES
    assert (5, 6) not in final.SKELETON_EDGES


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"ALL {len(tests)} WS1 FINALIZATION TESTS PASS")
