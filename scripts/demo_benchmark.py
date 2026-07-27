# -*- coding: utf-8 -*-
"""DEMO CHẠY NGAY (không cần Panoptic, không cần torch).

    python scripts/demo_benchmark.py

Sinh dữ liệu keypoint tổng hợp → tiêm offset có nhãn → chạy các baseline → in bảng
metric + bảng theo bucket vận tốc. Đây là bằng chứng harness hoạt động, và là mẫu để
mọi người cắm method của mình vào.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mist.benchmark import synthetic, eval as bench_eval
from mist.benchmark.baselines import ZeroOffset, CrossCorrelation, DTW, CaspiIrani


def main():
    print("== MIST benchmark DEMO (synthetic) ==")
    samples = synthetic.make_dataset(n=240, T=90, fps=30.0, seed=1, max_offset=2.0)
    print(f"Đã sinh {len(samples)} mẫu (30fps, offset ∈ [-2,2] frame, 4 mức vận tốc)\n")

    methods = [ZeroOffset(), CrossCorrelation(), DTW(), CaspiIrani()]
    results = bench_eval.run(samples, methods)

    print(bench_eval.table(results))
    print()
    # edges theo phân vị vận tốc synthetic (px/s) để 4 bucket đều có mẫu
    import numpy as np
    vel = np.array([s.velocity for s in samples])
    edges = tuple(round(float(q), 0) for q in np.quantile(vel, [0.25, 0.5, 0.75]))
    print(f"(vận tốc synthetic {vel.min():.0f}–{vel.max():.0f} px/s; edges={edges})")
    print(bench_eval.bucket_table(results, edges=edges))
    print("\n(DTW/Caspi = nan vì còn stub — điền vào là có số. "
          "ContinuSyncFormer bọc thành SyncMethod rồi thêm vào list là chạy chung.)")


if __name__ == "__main__":
    main()
