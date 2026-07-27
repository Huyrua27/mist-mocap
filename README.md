# MIST — Millisecond-accurate Intra-frame Synchronization

Sub-frame đồng bộ đa camera cho Motion Capture. Repo dùng chung cho nhóm 4 người,
hướng nộp **CSoNet 2026**.

> Vì bộ **IFID không truy cập được**, benchmark chính được dựng bằng **controlled-desync**
> từ CMU Panoptic (đã đồng bộ) + synthetic; data tự quay (5 cam) là tập real-world / downstream.

---

## Chạy thử NGAY (không cần Panoptic, không cần GPU)

```bash
pip install -r requirements.txt
python scripts/demo_benchmark.py
python tests/test_core.py
```

Demo sinh keypoint tổng hợp → tiêm offset có nhãn → chạy baseline → in bảng:

```
    Method           n     Frm.err   Accin@0.1  Accin@0.25      MAE_ms     RMSE_ms
------------------------------------------------------------------------------
 Zero(sàn)         240      0.9796      0.0583      0.1333      32.654      37.705
CC+parabol         240      0.0936       0.725      0.8792       3.118       5.729
       DTW         240         nan         ...  (stub — P1 điền)
```

Nếu bảng này ra được → toàn bộ harness hoạt động, cắm method mới vào là có số.

---

## Cấu trúc

```
mist-sync/
├── mist/
│   ├── core/            # HỢP ĐỒNG dùng chung — đừng đổi tự do
│   │   ├── types.py         KeypointSequence, SyncResult, SyncSample
│   │   └── interfaces.py     SyncMethod (mọi method kế thừa)
│   ├── benchmark/       # WS1 — P1
│   │   ├── desync.py         tiêm ∆t miền keypoint (spline)  ✅ xong
│   │   ├── metrics.py        Frm.err/Accin/Accex/MAE/RMSE + bucket  ✅ xong
│   │   ├── synthetic.py      sinh data demo/test  ✅ xong
│   │   ├── eval.py           harness chạy mọi method  ✅ xong
│   │   └── baselines/        CC ✅ | DTW ⬜ | Caspi-Irani ⬜ | Zero ✅
│   ├── model/          # WS2 — P2 :  continusyncformer.py  ⬜ (skeleton + TODO)
│   ├── panoptic/       # WS1 — P1 :  loader.py  ⬜ (nạp + chiếu 2D)
│   └── realworld/      # WS3 — P3 :  organize.py ⬜ | sync_pipeline.py ⬜
├── tools/              # ĐÃ CHẠY THẬT trên data: sync_groundtruth.py, record_webcams.py, sync_flash.html
├── scripts/            # demo_benchmark.py ✅ | run_benchmark.py
├── tests/              # test_core.py ✅
└── docs/               # ShootSheet, FormulaSheet, Benchmark, PhanCong
```
✅ = chạy được · ⬜ = stub chờ điền (đã có interface + TODO trong file)

---

## Hợp đồng dùng chung (đọc trước khi code)

Mọi thứ xoay quanh 2 file trong `mist/core/`:

- **`KeypointSequence`** — `xy (T,J,2)`, `fps`, `timestamps?`, `name`. Đơn vị input chung.
- **`SyncMethod.predict(a, b) -> SyncResult`** — trả `dt_frames` = offset của `b` so với `a`
  (frame, thực, dấu: `+` là b trễ). Baseline VÀ model đều cài hàm này → harness chạy như nhau.

Muốn thêm method mới (kể cả model đã train):
```python
class MyMethod(SyncMethod):
    name = "My"
    def predict(self, a, b): return SyncResult(dt_frames=..., confidence=...)
# rồi: eval.run(samples, [MyMethod()])  → có số ngay
```

---

## Ai làm gì (khớp docs/MIST_PhanCong.xlsx)

| Người | Thư mục | Nhiệm vụ chính |
|------|---------|----------------|
| **P1** | `benchmark/`, `panoptic/` | Panoptic loader → desync → harness → **baselines DTW/Caspi** |
| **P2** | `model/` | ContinuSyncFormer (RoPE, cross-view, hierarchical head), train, ablation |
| **P3** | `realworld/`, `tools/` | Giải nén/sync 5 cam, calibration, pose 2D, triangulation |
| **P4** | (xuyên suốt) | Survey, phân tích, figures, viết LNCS, **nộp CSoNet**, repo |

TODO cụ thể nằm ngay trong từng file stub (`TODO(Px #ID)` khớp ID task trong sheet).

---

## Cài đặt

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt          # numpy, scipy, opencv
pip install -r requirements-model.txt    # torch — CHỈ P2 cần
```

## Quy ước git
- Nhánh theo workstream: `ws1-benchmark`, `ws2-model`, `ws3-data`, `ws4-writing`.
- PR về `main`, không push thẳng. `python tests/test_core.py` phải PASS trước khi merge.
- **KHÔNG commit data nặng** (video/zip/MIST_data) — đã chặn trong `.gitignore`.
