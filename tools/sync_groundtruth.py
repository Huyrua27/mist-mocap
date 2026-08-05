#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIST — On-set Ground-Truth Sync Verifier
========================================
Kiểm tra ground-truth sync NGAY TẠI CHỖ lúc quay, cho hệ 6 camera dân dụng
+ iPhone/Blackmagic reference. Hai cơ chế:

  (L1/L2) FLASH ROI + TEMPORAL FIRST DERIVATIVE
          - Diệt bệnh baseline: tính brightness TOÀN khung -> false positive.
          - Chỉ tính trong FLASH_ROI (bóng LED trên nền tối).
          - Đỉnh của đạo hàm bậc nhất  dI/dt = I(t)-I(t-1)  = thời điểm tia
            sáng bùng lên (vách đá năng lượng), bỏ qua nền sáng biến thiên chậm.
          - Nội suy parabol quanh đỉnh -> vị trí SUB-FRAME của flash.

  (L3)    ĐỒNG HỒ LED NHỊ PHÂN (binary LED ms-counter)
          - Mỗi frame đọc dãy LED -> timestamp mili-giây trực tiếp.
          - Ground truth "đúng nghĩa" cho Accin@0.1 (xem README budget).

Nguồn vào: video (.mov/.mp4) HOẶC thư mục PNG (khớp pipeline baseline).
Chỉ cần:  numpy, opencv-python.

VÍ DỤ
-----
# 1) Dò 3 nháy flash trong 1 luồng, ROI thủ công:
python sync_groundtruth.py flash --src Cam1_Q.Huy.mov --roi 900,120,80,80 --n-flashes 3

# 2) Chọn ROI bằng chuột trên frame đầu (giữ file .roi.json để tái dùng):
python sync_groundtruth.py pick-roi --src Cam1_Q.Huy.mov

# 3) So khớp offset nhiều camera (mỗi cam 1 nguồn), xuất CSV:
python sync_groundtruth.py align --ref Cam1_Q.Huy.mov \
       --cams Cam2_Kien.mov Cam3_Khoa.mov ... --roi 900,120,80,80 --out offsets.csv

# 4) Đọc đồng hồ LED nhị phân (cấu hình vị trí LED trong file JSON):
python sync_groundtruth.py led --src reference_240fps.mov --led-config led.json
"""
import argparse, json, os, sys, glob, csv
import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("Cần opencv:  pip install opencv-python numpy")


# --------------------------------------------------------------------------- #
#  Nguồn frame: video hoặc thư mục PNG                                          #
# --------------------------------------------------------------------------- #
def iter_frames(src, max_frames=None):
    """Yield (index, gray_frame_uint8, bgr_frame). Hỗ trợ video và folder PNG."""
    if os.path.isdir(src):
        files = sorted(glob.glob(os.path.join(src, "*.png")) +
                       glob.glob(os.path.join(src, "*.jpg")))
        for i, f in enumerate(files):
            if max_frames and i >= max_frames:
                break
            bgr = cv2.imread(f, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            yield i, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), bgr
    else:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            sys.exit(f"Không mở được nguồn: {src}")
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok or (max_frames and i >= max_frames):
                break
            yield i, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), bgr
            i += 1
        cap.release()


def probe_fps(src):
    """FPS thật khai báo trong container (video). Folder PNG -> None (phải log tay)."""
    if os.path.isdir(src):
        return None
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return round(fps, 4) if fps and fps > 0 else None


# --------------------------------------------------------------------------- #
#  L1/L2 — FLASH ROI + TEMPORAL DERIVATIVE                                      #
# --------------------------------------------------------------------------- #
def roi_brightness_series(src, roi, max_frames=None):
    """Chuỗi độ sáng trung bình TRONG ROI (không phải toàn khung)."""
    x, y, w, h = roi
    vals, idxs = [], []
    for i, gray, _ in iter_frames(src, max_frames):
        patch = gray[y:y + h, x:x + w]
        vals.append(float(patch.mean()) if patch.size else 0.0)
        idxs.append(i)
    return np.asarray(idxs), np.asarray(vals, dtype=np.float64)


def _parabolic_subframe(y, k):
    """Nội suy parabol quanh đỉnh rời rạc k -> vị trí sub-frame (float)."""
    if k <= 0 or k >= len(y) - 1:
        return float(k)
    a, b, c = y[k - 1], y[k], y[k + 1]
    denom = (a - 2 * b + c)
    if abs(denom) < 1e-9:
        return float(k)
    return float(k) + 0.5 * (a - c) / denom


def detect_flashes(idxs, brightness, n_flashes=1, min_gap=8, smooth=3):
    """
    Trả về list các đỉnh flash, mỗi cái: dict(frame_int, frame_subpix, dI, baseline, conf).
    Dùng ĐẠO HÀM BẬC NHẤT dI = I(t)-I(t-1); đỉnh = cạnh lên dốc đứng của tia sáng.
    """
    b = brightness.astype(np.float64)
    n = len(b)
    if smooth > 1 and n > smooth:  # làm mượt nhẹ; EDGE-PAD để không tạo bước nhảy giả ở biên
        ker = np.ones(smooth) / smooth
        bpad = np.pad(b, smooth, mode="edge")
        b = np.convolve(bpad, ker, mode="same")[smooth:smooth + n]
    dI = np.diff(b, prepend=b[:1])            # đạo hàm bậc nhất
    baseline = float(np.median(b))
    noise = float(np.median(np.abs(dI - np.median(dI)))) * 1.4826 + 1e-6

    order = np.argsort(dI)[::-1]              # đỉnh đạo hàm lớn nhất trước
    picked = []
    for k in order:
        if all(abs(int(k) - p) >= min_gap for p in picked):
            picked.append(int(k))
        if len(picked) >= n_flashes:
            break
    picked.sort()

    out = []
    for k in picked:
        # k = cạnh lên (đạo hàm) — robust chống nền sáng chậm.
        # Refine SUB-FRAME trên đỉnh BRIGHTNESS (tâm xung đối xứng -> chính xác hơn).
        lo = max(0, k - 2)
        hi = min(len(b), k + smooth + 4)
        j = lo + int(np.argmax(b[lo:hi]))
        sub = _parabolic_subframe(b, j)
        out.append({
            "frame_int": int(idxs[j]),
            "frame_subpix": round(float(idxs[0]) + sub, 4),
            "edge_frame": int(idxs[k]),
            "dI": round(float(dI[k]), 4),
            "baseline": round(baseline, 3),
            "snr": round(float(dI[k]) / noise, 2),
            "confidence": _confidence(float(dI[k]), noise),
        })
    return out


def _confidence(dI, noise):
    snr = dI / noise
    if snr >= 8:   return "RẤT CAO"
    if snr >= 4:   return "TRUNG BÌNH"
    if snr >= 2:   return "THẤP"
    return "RẤT THẤP (nghi false-positive)"


# --------------------------------------------------------------------------- #
#  L3 — ĐỒNG HỒ LED NHỊ PHÂN                                                    #
# --------------------------------------------------------------------------- #
def decode_binary_led(gray, led_boxes, thresh=None, msb_first=True):
    """
    led_boxes: list [x,y,w,h] cho từng bit LED (bit thấp -> cao, hoặc ngược).
    Trả về (value:int, bits:list, sáng_tối rõ ràng? bool).
    """
    lums = []
    for (x, y, w, h) in led_boxes:
        patch = gray[y:y + h, x:x + w]
        lums.append(float(patch.mean()) if patch.size else 0.0)
    lums = np.asarray(lums)
    if thresh is None:  # tự ngưỡng giữa cụm sáng và tối
        thresh = (lums.min() + lums.max()) / 2.0
    bits = (lums > thresh).astype(int).tolist()
    seq = bits if msb_first else bits[::-1]
    value = 0
    for bt in seq:
        value = (value << 1) | bt
    margin = float(lums.max() - lums.min())
    return value, bits, margin


def read_led_clock(src, led_config, max_frames=None):
    """
    Đọc timestamp ms mỗi frame từ đồng hồ LED nhị phân.
    led_config = {"boxes":[[x,y,w,h],...], "msb_first":true, "unit_ms":1,
                  "thresh":null}
    Trả về list dict(frame, raw_value, t_ms, margin).
    """
    boxes = [list(map(int, b)) for b in led_config["boxes"]]
    msb = led_config.get("msb_first", True)
    unit = led_config.get("unit_ms", 1.0)
    thr = led_config.get("thresh", None)
    rows = []
    for i, gray, _ in iter_frames(src, max_frames):
        val, bits, margin = decode_binary_led(gray, boxes, thr, msb)
        rows.append({"frame": i, "raw_value": val,
                     "t_ms": round(val * unit, 3), "margin": round(margin, 1),
                     "bits": "".join(map(str, bits))})
    return rows


def fit_frame_to_time(rows):
    """
    Khớp affine  t = a*frame + b  từ chuỗi LED (cho ra period thực & phase).
    Trả về (a_ms_per_frame, b_ms, residual_std_ms, n).  Dùng để đo clock drift α.
    """
    f = np.array([r["frame"] for r in rows], float)
    t = np.array([r["t_ms"] for r in rows], float)
    # bỏ frame margin thấp (đọc LED không chắc) — dùng median margin làm ngưỡng
    m = np.array([r["margin"] for r in rows], float)
    keep = m > (np.median(m) * 0.5)
    f, t = f[keep], t[keep]
    if len(f) < 3:
        return None
    A = np.vstack([f, np.ones_like(f)]).T
    (a, b), *_ = np.linalg.lstsq(A, t, rcond=None)
    resid = t - (a * f + b)
    return {"ms_per_frame": round(float(a), 5),
            "fps_effective": round(1000.0 / a, 4) if a else None,
            "phase_ms": round(float(b), 3),
            "residual_std_ms": round(float(resid.std()), 4),
            "n_used": int(keep.sum())}


# --------------------------------------------------------------------------- #
#  ROI picker (chuột)                                                           #
# --------------------------------------------------------------------------- #
def pick_roi(src):
    first = next(iter_frames(src, max_frames=1), None)
    if first is None:
        sys.exit("Không đọc được frame đầu.")
    _, _, bgr = first
    r = cv2.selectROI("Chọn FLASH_ROI (Enter=xong, c=huỷ)", bgr, showCrosshair=True)
    cv2.destroyAllWindows()
    x, y, w, h = map(int, r)
    if w == 0 or h == 0:
        sys.exit("Huỷ chọn ROI.")
    roi = [x, y, w, h]
    out = os.path.splitext(src)[0] + ".roi.json"
    json.dump({"roi": roi}, open(out, "w"))
    print(f"ROI = {roi}   (đã lưu {out})")
    return roi


# --------------------------------------------------------------------------- #
#  CLI                                                                          #
# --------------------------------------------------------------------------- #
def _parse_roi(s):
    return list(map(int, s.split(",")))


def cmd_flash(a):
    roi = _parse_roi(a.roi)
    idxs, bright = roi_brightness_series(a.src, roi, a.max_frames)
    flashes = detect_flashes(idxs, bright, a.n_flashes, a.min_gap, a.smooth)
    fps = probe_fps(a.src)
    print(f"\n[{os.path.basename(a.src)}]  fps={fps}  ROI={roi}")
    for j, fl in enumerate(flashes):
        ms = f"{fl['frame_subpix']*1000/fps:.2f} ms" if fps else "n/a"
        print(f"  Flash#{j+1}: frame={fl['frame_int']} "
              f"sub={fl['frame_subpix']} (t≈{ms})  "
              f"dI={fl['dI']} SNR={fl['snr']} -> {fl['confidence']}")
    if not flashes:
        print("  (không tìm thấy flash — kiểm tra ROI/nền có tối không)")
    return flashes


def cmd_align(a):
    roi = _parse_roi(a.roi)
    sources = [a.ref] + a.cams
    results = {}
    for s in sources:
        idxs, bright = roi_brightness_series(s, roi, a.max_frames)
        fl = detect_flashes(idxs, bright, 1, a.min_gap, a.smooth)
        results[s] = (fl[0] if fl else None, probe_fps(s))

    ref_fl, ref_fps = results[a.ref]
    if ref_fl is None:
        sys.exit("Không dò được flash trên REF — không thể canh offset.")
    ref_sub = ref_fl["frame_subpix"]

    rows = []
    print(f"\n=== OFFSET so với REF={os.path.basename(a.ref)} "
          f"(flash sub-frame={ref_sub}) ===")
    for s in sources:
        fl, fps = results[s]
        if fl is None:
            rows.append([os.path.basename(s), fps, "", "", "", "NO-FLASH"])
            print(f"  {os.path.basename(s):22} : KHÔNG dò được flash")
            continue
        d_frame = round(fl["frame_subpix"] - ref_sub, 4)
        d_ms = round(d_frame * 1000 / fps, 3) if fps else ""
        rows.append([os.path.basename(s), fps, fl["frame_subpix"],
                     d_frame, d_ms, fl["confidence"]])
        print(f"  {os.path.basename(s):22} : Δframe={d_frame:+.4f}  "
              f"Δt={d_ms} ms  [{fl['confidence']}]")

    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["camera", "fps", "flash_subframe",
                        "delta_frame_vs_ref", "delta_ms_vs_ref", "confidence"])
            w.writerows(rows)
        print(f"\n-> đã ghi {a.out}")
    return rows


def cmd_led(a):
    cfg = json.load(open(a.led_config, encoding="utf-8"))
    rows = read_led_clock(a.src, cfg, a.max_frames)
    fit = fit_frame_to_time(rows)
    print(f"\n[{os.path.basename(a.src)}]  đọc {len(rows)} frame LED-clock")
    for r in rows[:5]:
        print(f"  frame {r['frame']:>4}: t={r['t_ms']} ms  bits={r['bits']} "
              f"margin={r['margin']}")
    if len(rows) > 5:
        print("  ...")
    if fit:
        print(f"\n  AFFINE t=a*f+b :  ms/frame={fit['ms_per_frame']} "
              f"(fps_eff={fit['fps_effective']})  phase={fit['phase_ms']} ms")
        print(f"  residual_std={fit['residual_std_ms']} ms  "
              f"(clock-drift/nhiễu đọc; càng nhỏ càng tốt)  n={fit['n_used']}")
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "raw_value", "t_ms",
                                              "margin", "bits"])
            w.writeheader()
            w.writerows(rows)
        print(f"-> đã ghi {a.out}")
    return rows


def cmd_pickroi(a):
    pick_roi(a.src)


def build_parser():
    p = argparse.ArgumentParser(description="MIST on-set ground-truth sync verifier")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--src", required=True, help="video HOẶC thư mục PNG")
        sp.add_argument("--max-frames", type=int, default=None)

    f = sub.add_parser("flash", help="dò flash 1 luồng (ROI + đạo hàm)")
    common(f)
    f.add_argument("--roi", required=True, help="x,y,w,h")
    f.add_argument("--n-flashes", type=int, default=1)
    f.add_argument("--min-gap", type=int, default=8)
    f.add_argument("--smooth", type=int, default=3)
    f.set_defaults(func=cmd_flash)

    al = sub.add_parser("align", help="canh offset nhiều camera vs ref")
    al.add_argument("--ref", required=True)
    al.add_argument("--cams", nargs="+", default=[])
    al.add_argument("--roi", required=True, help="x,y,w,h (chung mọi camera)")
    al.add_argument("--min-gap", type=int, default=8)
    al.add_argument("--smooth", type=int, default=3)
    al.add_argument("--max-frames", type=int, default=None)
    al.add_argument("--out", default=None, help="CSV")
    al.set_defaults(func=cmd_align)

    le = sub.add_parser("led", help="đọc đồng hồ LED nhị phân (L3 ground truth)")
    common(le)
    le.add_argument("--led-config", required=True, help="JSON: boxes/msb_first/unit_ms")
    le.add_argument("--out", default=None, help="CSV")
    le.set_defaults(func=cmd_led)

    pr = sub.add_parser("pick-roi", help="chọn ROI bằng chuột, lưu .roi.json")
    pr.add_argument("--src", required=True)
    pr.set_defaults(func=cmd_pickroi)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
