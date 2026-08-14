# -*- coding: utf-8 -*-
"""Simulated 2D pose-estimator noise, to bridge clean Panoptic projections toward
real detector output (HRNet / ViTPose).  Task #24 (realism without real capture).

Real 2D keypoint detectors are not exact: every joint carries a few-pixel
localization error, joints occasionally fail grossly (left/right swaps, jumps), and
some are missed entirely under occlusion. We inject the first two here (they keep the
coordinates finite); missed detections are modeled separately as visibility dropout
(the dataset's ``occlusion_p``). Applying this to both views makes the benchmark a far
more honest stand-in for real footage than clean projected keypoints.
"""
from __future__ import annotations

import numpy as np


def apply_detector_noise(xy: np.ndarray, rng, noise_px: float = 0.0,
                         outlier_p: float = 0.0, outlier_px: float = 25.0,
                         corr_frames: float = 5.0) -> np.ndarray:
    """Return a finite noisy copy of ``xy`` (T, J, 2) in pixels.

    noise_px    : std of per-joint localization error (pixels).
    corr_frames : temporal correlation length of that error, in frames. Real 2D
                  detectors mislocalize a joint *consistently* across nearby frames,
                  so the error is temporally smooth; pure per-frame iid noise (0)
                  is an unrealistic worst case that erases the motion signal. The
                  absolute error std stays ``noise_px``; only its frame-to-frame
                  jitter shrinks with ``corr_frames``.
    outlier_p   : per-joint-per-frame probability of a gross detector failure.
    outlier_px  : std of the extra displacement applied to those outlier joints.
    """
    out = np.asarray(xy, dtype=np.float64).copy()
    if noise_px > 0:
        e = rng.normal(0.0, 1.0, size=out.shape)
        w = int(round(corr_frames))
        if w > 1 and out.shape[0] > w:                        # temporal smoothing
            k = np.ones(w) / w
            e = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, e)
            e /= (e.std() + 1e-9)                             # restore unit std
        out += noise_px * e
    if outlier_p > 0:
        hit = rng.random(out.shape[:-1]) < outlier_p          # (T, J)
        n = int(hit.sum())
        if n:
            out[hit] += rng.normal(0.0, outlier_px, size=(n, 2))
    return out
