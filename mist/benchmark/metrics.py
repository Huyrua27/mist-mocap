# -*- coding: utf-8 -*-
"""Bộ metric chuẩn cho bài toán sync (khớp bảng trong paper & FormulaSheet).

Owner: P1 (WS1). Mọi method chấm bằng đúng bộ này để công bằng.
"""
from __future__ import annotations
import numpy as np


def _arr(x): return np.asarray(x, dtype=float)


def frm_err(pred, gt) -> float:
    """Sai số trung bình theo đơn vị frame:  mean(|∆f_pred − ∆f_gt|)."""
    return float(np.mean(np.abs(_arr(pred) - _arr(gt))))


def accin(pred, gt, tau=0.1) -> float:
    """Độ chính xác MỊN: tỉ lệ mẫu có |sai số| < tau frame (mặc định 0.1)."""
    return float(np.mean(np.abs(_arr(pred) - _arr(gt)) < tau))


def accex(pred, gt, i=1) -> float:
    """Độ chính xác THÔ: tỉ lệ mẫu sai số frame-NGUYÊN < i."""
    return float(np.mean(np.abs(np.round(_arr(pred)) - np.round(_arr(gt))) < i))


def mae_ms(pred, gt, fps) -> float:
    return float(np.mean(np.abs(_arr(pred) - _arr(gt))) * 1000.0 / fps)


def rmse_ms(pred, gt, fps) -> float:
    e = _arr(pred) - _arr(gt)
    return float(np.sqrt(np.mean(e ** 2)) * 1000.0 / fps)


def summary(pred, gt, fps, taus=(0.1, 0.25, 0.5)) -> dict:
    """Gói toàn bộ metric cho 1 method."""
    d = {"n": len(pred), "Frm.err": round(frm_err(pred, gt), 4)}
    for t in taus:
        d[f"Accin@{t}"] = round(accin(pred, gt, t), 4)
    d["Accex@1"] = round(accex(pred, gt, 1), 4)
    d["MAE_ms"] = round(mae_ms(pred, gt, fps), 3)
    d["RMSE_ms"] = round(rmse_ms(pred, gt, fps), 3)
    return d


def by_velocity_bucket(pred, gt, vel, fps, edges=(1.5, 5.0, 15.0), tau=0.1) -> dict:
    """Chia mẫu theo vận tốc rồi báo Accin@tau + Frm.err từng bucket.

    edges mặc định khớp Bảng 1 paper (đi/chạy/thể thao/cực nhanh).
    """
    pred, gt, vel = _arr(pred), _arr(gt), _arr(vel)
    bounds = [-np.inf, *edges, np.inf]
    labels = ["<{}".format(edges[0])] + \
             [f"{edges[k]}-{edges[k+1]}" for k in range(len(edges) - 1)] + \
             [">{}".format(edges[-1])]
    out = {}
    for k, lab in enumerate(labels):
        m = (vel >= bounds[k]) & (vel < bounds[k + 1])
        if m.sum() == 0:
            out[lab] = {"n": 0}
        else:
            out[lab] = {"n": int(m.sum()),
                        f"Accin@{tau}": round(accin(pred[m], gt[m], tau), 4),
                        "Frm.err": round(frm_err(pred[m], gt[m]), 4)}
    return out
