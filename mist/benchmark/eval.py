# -*- coding: utf-8 -*-
"""Harness chạy MỌI method trên MỘT tập mẫu → bảng metric. Dùng chung cả nhóm.

Owner: P1 (WS1). P2 chỉ cần bọc model thành SyncMethod là cắm vào đây được.
"""
from __future__ import annotations
from typing import Iterable
from ..core.interfaces import SyncMethod
from ..core.types import SyncSample
from . import metrics


def run(samples: list[SyncSample], methods: Iterable[SyncMethod]) -> dict:
    """Chạy từng method trên toàn bộ mẫu. Trả dict[name] -> (preds, gts, vels, fps)."""
    results = {}
    for m in methods:
        preds, gts, vels = [], [], []
        for s in samples:
            try:
                r = m.predict(s.a, s.b)
                preds.append(r.dt_frames)
            except NotImplementedError:
                preds.append(float("nan"))
            gts.append(s.dt_gt_frames)
            vels.append(s.velocity)
        results[m.name] = (preds, gts, vels, samples[0].a.fps)
    return results


def table(results: dict) -> str:
    """Bảng metric tổng thể (text, in ra terminal)."""
    cols = ["Method", "n", "Frm.err", "Accin@0.1", "Accin@0.25", "MAE_ms", "RMSE_ms"]
    lines = ["  ".join(f"{c:>10}" for c in cols), "-" * 78]
    for name, (p, g, v, fps) in results.items():
        s = metrics.summary(p, g, fps)
        row = [name[:10], s["n"], s["Frm.err"], s["Accin@0.1"],
               s["Accin@0.25"], s["MAE_ms"], s["RMSE_ms"]]
        lines.append("  ".join(f"{str(x):>10}" for x in row))
    return "\n".join(lines)


def bucket_table(results: dict, edges=(1.5, 5.0, 15.0)) -> str:
    """Bảng Accin@0.1 theo bucket vận tốc — nơi chứng minh baseline sập ở tốc độ cao."""
    lines = ["Accin@0.1 theo vận tốc (px/s):"]
    for name, (p, g, v, fps) in results.items():
        b = metrics.by_velocity_bucket(p, g, v, fps, edges=edges)
        cells = "  ".join(f"{k}:{d.get('Accin@0.1','-')}(n{d['n']})" for k, d in b.items())
        lines.append(f"  {name:12} {cells}")
    return "\n".join(lines)
