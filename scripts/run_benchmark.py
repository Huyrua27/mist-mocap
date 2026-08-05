# -*- coding: utf-8 -*-
"""Chạy benchmark THẬT trên Panoptic (khi loader + desync sẵn sàng).

    python scripts/run_benchmark.py --panoptic <seq_dir>

Luồng dự kiến (P1 hoàn thiện khi panoptic/loader.py xong):
    cams = panoptic.load_sequence(seq_dir)        # {cam: KeypointSequence} đã sync
    samples = build_samples(cams, offsets, ...)   # dùng desync.inject_offset tạo B1-test
    results = eval.run(samples, [CrossCorrelation(), DTW(), CaspiIrani(), ContinuSyncMethod(ckpt)])
    print(eval.table(results)); print(eval.bucket_table(results))
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panoptic", help="thư mục sequence Panoptic")
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()
    if not args.panoptic:
        print("Chưa có Panoptic → chạy demo synthetic thay thế:")
        print("    python scripts/demo_benchmark.py")
        return
    raise NotImplementedError(
        "TODO(P1): nối panoptic.load_sequence + desync.inject_offset + eval.run")


if __name__ == "__main__":
    main()
