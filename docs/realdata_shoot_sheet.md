# MIST — Shoot sheet: buổi quay data thật (drift + flash GT)

Mục tiêu: quay ≥3 take, mỗi take có (a) drift clock thật giữa các thiết bị, (b) flash
định kỳ làm ground-truth, (c) ChArUco để calib 3D. Xong là chạy `scripts/realdata/`.

**Tiêu chí THÀNH CÔNG của buổi quay (kiểm ngay tại chỗ, không để về mới biết):**
- SNR flash ≥ 4 ở **mọi** camera (chạy `detect_flash_gt.py` trên clip thử 10s).
- Mỗi camera thấy trọn người suốt take + thấy bảng ChArUco lúc calib.
- Take dài ≥ 5 phút, flash loé đều mỗi ~20s.

---

## 1. Thiết bị & checklist mang theo
- [ ] 3 điện thoại (khác model) — **C0, C2, C4**. Cài sẵn **Open Camera**.
- [ ] 2 webcam + laptop (cáp USB đủ dài) — **C1, C3**, chạy `record_webcams.py`.
- [ ] 1 **đèn pin LED sáng** làm nguồn flash (KHÔNG dùng màn hình điện thoại).
- [ ] Bảng **ChArUco** (ghi sẵn: squares_x, squares_y, square_len, marker_len — cần cho calib).
- [ ] 5 tripod + giá kẹp điện thoại.
- [ ] 1 điện thoại/đồng hồ bấm giây cho người bắn flash.
- [ ] Băng dán sàn (đánh dấu vùng diễn + chân tripod), sạc/pin dự phòng, thẻ nhớ trống.
- [ ] Laptop đã cài repo + venv (để chạy QC tại chỗ).

## 2. Không gian & bố trí camera
Người diễn ở **tâm**. 5 camera trên cung ~120°, bán kính **~3 m**, cách nhau **~30°**,
cao **~1.35 m**, chúc xuống ~10–15°. Laptop đặt sau lưng cung (giữa), webcam nối vào đó.

```
        C0(-60°)   C1(-30°)   C2(0°)   C3(+30°)   C4(+60°)
        PhoneA     Webcam1    PhoneB   Webcam2    PhoneC
              \       \        |       /        /
               \       \       |      /       /
                \       \      |     /       /
                        ◎  người + vùng diễn ~2×2 m
                     (đánh dấu băng dán sàn)
```
| Vị trí | Góc | Thiết bị | Ghi hình bằng | Clock |
|---|---|---|---|---|
| C0 | −60° | Phone A | Open Camera | riêng |
| C1 | −30° | Webcam 1 | record_webcams.py | laptop |
| C2 | 0° (chính diện) | Phone B | Open Camera | riêng |
| C3 | +30° | Webcam 2 | record_webcams.py | laptop |
| C4 | +60° | Phone C | Open Camera | riêng |

- **3 phone = 3 clock độc lập → nguồn drift chính.** 2 webcam chung clock laptop (cặp gần-đồng-bộ, dùng làm baseline + triangulation).
- Cặp kề ~30° (tốt cho sync, in-distribution của model ≤60°); cặp xa (C0–C4 ~120°) tốt cho triangulation.
- **Kiểm preview:** mỗi camera phải thấy **trọn người kể cả lúc nhảy** (không cụt đầu/chân). Chưa đủ → lùi camera ra ~3.2 m hoặc quay dọc (portrait).

## 3. Cấu hình từng thiết bị (BẮT BUỘC khoá tay)
**Điện thoại (Open Camera):**
- [ ] FPS = **30** cố định.
- [ ] **Khoá phơi sáng (exposure/ISO)** — nếu không, camera tự chỉnh sáng khi flash loé → hỏng detect (đúng lỗi lần trước).
- [ ] **Khoá lấy nét (focus)** vào vùng diễn.
- [ ] **Khoá cân bằng trắng (WB)**.
- [ ] Độ phân giải 1080p, quay ngang (hoặc dọc nếu cần chiều cao).

**Webcam (`record_webcams.py`):**
```bash
python tools/record_webcams.py --cams 0 1 --fps 30 --width 1280 --height 720 --duration 400
```
- [ ] Xác nhận cả 2 webcam mở đúng, MJPG, 30fps (script tự verify).

**Đừng bao giờ inter-sync các thiết bị** (timecode/genlock) — sẽ xoá mất drift cần đo.

## 4. Nguồn flash (fix vụ phòng sáng)
- Đèn pin LED **chiếu lên TRẦN/tường sáng** → cả phòng loé 1 nhịp → mọi camera bắt được.
- **Giảm bớt đèn phòng** để tương phản transient cao.
- ROI = mảng trần/tường sáng trong mỗi khung (chọn ở bước QC bằng `pick-roi`).
- **Nhịp:** người bắn flash **gõ 3 nhịp nhanh mỗi ~20 giây** (xem đồng hồ bấm giây). 3-nhịp để khớp chắc; 1 nhịp cũng được nếu vội.
- Đèn pin là **thiết bị thứ 6 riêng**, KHÔNG phải 1 trong 5 camera.

## 5. Phân vai (4 người: em + Khải, Nam, Phong)
- **Đạo diễn/bấm máy (em):** hô lệnh, kiểm preview, ra hiệu bắt đầu/kết thúc.
- **Kỹ thuật laptop (Khải):** chạy `record_webcams.py`, chạy QC 10s tại chỗ.
- **Người bắn flash (Nam):** cầm đèn pin, gõ 3-nhịp mỗi 20s theo đồng hồ.
- **Diễn viên (Phong / luân phiên):** thực hiện choreography chuyển động.

---

## 6. QUY TRÌNH QC 10 GIÂY (làm TRƯỚC mọi take thật — quan trọng nhất)
1. Bật tất cả camera quay, bắn **3 nhịp flash**, quay ~10s, dừng.
2. Offload nhanh 5 clip lên laptop.
3. Chọn ROI mỗi cam: `python tools/sync_groundtruth.py pick-roi --src <clip>`
4. Chạy: `python scripts/realdata/detect_flash_gt.py --videos <5 clip> --ref <cam C2> --n-flashes 3 --out /tmp/qc.json`
5. **Đọc SNR in ra.** Mọi cam ≥ 4 → OK, quay thật. Có cam "LOW" → giảm đèn phòng / kéo ROI sát mảng sáng hơn / tăng độ sáng đèn pin, rồi thử lại.

**Không đạt QC thì KHÔNG quay take thật.** Đây là chỗ chết lần trước.

## 7. Run-of-show — một take (~6 phút)
| Mốc | Ai | Hành động |
|---|---|---|
| −10s | Khải | Bắt đầu `record_webcams.py` (chạy nền, đủ --duration) |
| −5s | em | Hô "bắt đầu", tất cả phone bấm REC |
| 0:00 | Nam | **Vỗ tay 1 cái to** (slate thô, mốc thời gian dự phòng) |
| 0:00–0:30 | Phong | **Vẫy bảng ChArUco** khắp vùng, nghiêng đủ hướng, chậm rãi, mọi cam thấy rõ |
| 0:30–0:40 | Phong | **Đặt bảng ChArUco ĐỨNG YÊN** giữa vùng, mọi cam thấy (10s cho extrinsics) — ghi lại frame index này! |
| 0:40 | Nam | **3 nhịp flash mở màn** |
| 0:40–5:40 | Phong | **Choreography chuyển động** (mục 8); Nam **3 nhịp flash mỗi ~20s** |
| 5:40 | Nam | **3 nhịp flash kết thúc** |
| 5:45 | em | Hô "cắt", dừng tất cả |

Quay **3–4 take**, đổi diễn viên/tốc độ để có clip đa dạng (cho error bar).

## 8. Choreography chuyển động (đa tốc độ = tín hiệu sync mạnh)
Trong vùng ~2×2 m, mỗi động tác ~30–40s, đan xen nhanh/chậm:
1. Đi bộ tại chỗ / qua lại (chậm).
2. Vẫy hai tay biên độ lớn (vừa).
3. Squat / đứng lên lặp lại (vừa).
4. **Đấm gió / boxing nhanh** (nhanh — quan trọng, motion nhanh cho sync tốt).
5. **Nhảy tại chỗ / bật cao** (nhanh; giữ trong khung).
6. Đá chân trước/sau (nhanh).
7. Xoay người 360° (vừa).
8. Lặp lại vòng 2 với tốc độ khác.
→ Xen kẽ nhanh–chậm giúp cả sync (cần nhanh) lẫn có đoạn Q1–Q4 velocity.

## 9. Đặt tên file & offload
- Phone: đổi tên ngay sau mỗi take → `s1_t1_c0_phoneA.mp4`, `s1_t1_c2_phoneB.mp4`, `s1_t1_c4_phoneC.mp4`.
- Webcam: `record_webcams.py` xuất sẵn + CSV timestamp → gom vào `s1_t1_c1_web1.*`, `s1_t1_c3_web2.*`.
- Cấu trúc: `data/realdata/s1_t1/` chứa 5 video + (sau này) `pose/`, `gt_drift.json`, `calib.json`.
- **Ghi sổ mỗi take:** frame index lúc bảng đứng yên (cho `--static-frame`), số nhịp flash tổng cộng (cho `--n-flashes`).

## 10. Bảng xử lý sự cố nhanh
| Triệu chứng | Nguyên nhân | Xử lý |
|---|---|---|
| SNR flash thấp | Phòng sáng / ROI rộng / đèn yếu | Giảm đèn phòng, ROI sát mảng sáng, đèn pin mạnh hơn, chiếu trần |
| Flash nhòe, khó bắt | Auto-exposure bật | Khoá exposure trên Open Camera |
| Số flash phát hiện khác nhau giữa cam | Cam nào đó miss/false peak | Tăng `--min-gap`, chọn lại ROI, kiểm cam đó thấy flash không |
| Người bị cụt lúc nhảy | Khung hẹp | Lùi camera / quay dọc / nhảy thấp hơn |
| Webcam rớt frame | Băng thông USB | Giữ MJPG, giảm còn 720p, tách 2 webcam ra 2 cổng USB khác hub |
| Drift đo được ~0 | Take quá ngắn | Quay ≥5 phút để drift tích luỹ tới mức sub-frame |

## 11. Sau buổi quay → chạy pipeline
Theo `scripts/realdata/README.md`: pick-roi → detect_flash_gt → extract_pose2d →
run_realdata_sync (headline) → (tùy chọn) calibrate_charuco → triangulate_realdata.
