# -*- coding: utf-8 -*-
"""Test sanity — chạy: python -m pytest tests/  (hoặc python tests/test_core.py)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from mist.benchmark import synthetic, metrics
from mist.benchmark.desync import inject_offset
from mist.benchmark.baselines import CrossCorrelation


def test_metrics_perfect():
    gt = [0.1, -0.3, 0.5, 1.2]
    assert metrics.frm_err(gt, gt) == 0.0
    assert metrics.accin(gt, gt, 0.1) == 1.0


def test_metrics_known():
    # sai số cố định 0.2 frame @30fps -> MAE = 0.2*1000/30 = 6.667 ms
    pred = [0.2, 0.2]; gt = [0.0, 0.0]
    assert abs(metrics.mae_ms(pred, gt, 30.0) - 6.6667) < 1e-3
    assert metrics.accin(pred, gt, 0.1) == 0.0     # 0.2 > 0.1 -> trượt
    assert metrics.accin(pred, gt, 0.25) == 1.0


def test_desync_roundtrip_cc():
    """Tiêm offset rồi CC phải khôi phục lại (đúng dấu, sai số nhỏ)."""
    cc = CrossCorrelation()
    errs = []
    for seed in range(20):
        a = synthetic.make_trajectory(T=120, fps=30, seed=seed, speed=1.5)
        dt = float(np.random.default_rng(seed).uniform(-2, 2))
        b = inject_offset(a, dt)
        pred = cc.predict(a, b).dt_frames
        errs.append(abs(pred - dt))
    mean_err = float(np.mean(errs))
    print(f"CC sai số trung bình = {mean_err:.4f} frame")
    assert mean_err < 0.15, f"CC lệch quá lớn ({mean_err}) — kiểm tra dấu/logic"


if __name__ == "__main__":
    test_metrics_perfect(); test_metrics_known(); test_desync_roundtrip_cc()
    print("ALL TESTS PASS")
