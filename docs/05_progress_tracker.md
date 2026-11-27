# Bảng Theo Dõi Tiến Trình Thực Chiến (Đội 18 · `TEAM_ID = 23`)
**Cập nhật lần cuối:** `12:10 · 27/11/2026` · **Thời gian còn lại đến đóng cổng (`10:30 28/11`):** `~22 giờ 20 phút`

---

## 1. Các Mốc Đã Hoàn Thành (100% Verified)

| Bước | Trạng thái | Mốc hoàn thành | Kết quả & Bằng chứng cụ thể |
|---|---|---|---|
| **Bước 0** | ✅ XONG | `10:30 27/11` | Giải phẫu toàn bộ website `competition.antco.ai`, `labeling_guideline.md`, `train_and_export.ipynb` (`TEAM_ID=23`, `YOLO26n-seg`, `imgsz=640`, 11 class, 50% Private LB + 50% Slide). |
| **Bước 1** | ✅ XONG | `11:07 27/11` | Dựng trọn bộ project tại `D:\2026\VinUni_Lv2\robotaxi-hackathon-team18`, viết & kiểm thử **6/6 unit test passed** cho lõi `seg_utils.py` + 4 script (`01` → `04`) + notebook chạy Colab/Kaggle. |
| **Bước 2** | ✅ XONG | `11:30 27/11` | Chốt 7 URL YouTube 4K đường phố VN tách tuyến Val/Train trong `configs/vn_youtube_sources.yaml`; xác minh schema thật của HF `star092304/Traffic-sign-detection-VietNam` (905 MB, CC-BY-NC-4.0) & Kaggle `manjotpahwa/indian-driving-dataset` (5.51 GB). |
| **Bước 3A** | ✅ XONG | `11:43 27/11` | Gom xong **500 ảnh đường phố VN thật (0% leakage)**: **200 ảnh Val** (từ `/valid/`) + **300 ảnh Train** (từ `/train/`) kèm GT bbox biển báo VN & `harvest_manifest.csv`. *(YouTube bị chặn IP datacenter Colab nên tự chuyển sang kho 12.189 ảnh VN đã cache sẵn trên Colab!)* |
| **Bước 3B** | ✅ XONG | `12:00 27/11` | Chạy xong **Pre-label 3 tầng trên Colab Pro Tesla T4 GPU** (`YOLO11x-seg` + `YOLO-Worldv2` + `SAM2.1-b` + Rider-Bike Union + Non-Overlap):<br>• `val_200_prelabel_cvat.zip` (**19 MB**, 200 ảnh, **1.750 instances**)<br>• `train_b1_200_prelabel_cvat.zip` (**17 MB**, 200 ảnh, **1.639 instances**) |
| **Bước 3C** | ✅ XONG | `12:01 27/11` | Chuyển đổi xong **600 ảnh IDD tinh lọc** (từ 8.089 ảnh gốc): **14.910 instances ĐỦ CẢ 11 LỚP**, tỉ lệ imbalance chỉ **21.0x** (`Other vehicle`: 2.833, `Motorcyclist`: 2.043, `Sign`: 660, `Traffic light`: 222, `Cyclist`: 135). |
| **Backup** | ✅ XONG | `12:07 27/11` | Đã backup toàn bộ sang Google Drive (`robotaxi-hackathon-team18_step3`) và tải 2 file zip CVAT về local (`data/processed/step3_cvat_zips/`). |

---

## 2. Giai Đoạn Tiếp Theo — Chiến Lược Song Song 2 Luồng (Từ `12:15` Hôm Nay)

### ⚡ LUỒNG 1 (GPU Colab Pro — V1 Baseline ĐANG TRAIN LIVE TRÊN T4 GPU!)
- [x] **1.1 — Dọn gọn Google Drive & đồng bộ code mới nhất (`12:20`):** Xong! Toàn bộ đã gộp vào `Colab Notebooks/robotaxi-hackathon-team18/`.
- [x] **1.2 — Nâng cấp IDD lên 1.200 ảnh (`12:25`):** Xong! **27.682 instances** đủ 11 lớp (`runs/v1/idd_1200_provenance.md`).
- [x] **1.3 — Chạy Script #4 đóng gói `dataset_v1.zip` (`12:27`):** Xong! `281.93 MiB`, SHA-256 `73ef1f20...`, **1.380 Train (28.783 inst) / 200 Val (1.750 inst)**, `0` leakage, `20` dup đã lọc, imbalance `30.7x`.
- [ ] **1.4 — Train & Nộp V1 Baseline (`12:28 → ~12:55`):** **ĐANG TRAIN LIVE** trên Tesla T4 GPU (`runs/v1/train_v1.ipynb`, `TEAM_ID = 23`, `EPOCHS = 50`, đã qua sạch Bước 0→1→2). Chờ xong Bước 4→5→6 → lấy `submission.zip` nộp Lượt 1/20 & đọc Bảng AP 11 lớp!

### 🎯 LUỒNG 2 (Đội Gán Nhãn CVAT — Chạy song song NGAY lúc `12:15`!)
- [ ] **2.1 — Kiểm tra nhanh `val_preview.jpg` & `train_b1_preview.jpg`** (1 phút) để phát hiện lỗi hệ thống của pre-label (biển quảng cáo, cặp `Motorcycle`/`Motorcyclist`) và nhắn cả nhóm lưu ý.
- [ ] **2.2 — Tạo Project CVAT 11 lớp đúng thứ tự ID `0 → 10`** (theo `configs/classes.yaml`).
- [ ] **2.3 — Nạp `val_200_prelabel_cvat.zip` (`YOLO Segmentation 1.0`) → Dual-QC 2 người 100%** chốt **Golden VN Val Set** chuẩn tuyệt đối.
- [ ] **2.4 — Nạp `train_b1_200_prelabel_cvat.zip` → sửa sạch 200 ảnh VN Train Batch 1** (ưu tiên soi kỹ 3 lớp yếu nhất từ Bước 5 của V1!).
- [ ] **2.5 — Xuất 2 file sạch từ CVAT** (`YOLO Segmentation 1.0`) đẩy lên Drive `data/processed/cvat_clean/`.

### 🚀 CHẶNG BỨT TỐC TOP 1 (Chiều → Sáng mai `28/11`)
- [ ] **V2 (`15:00 – 19:30`):** Thay bộ nhãn pre-label bằng **Golden Val sạch + 200 VN Train Batch 1 sạch** + pre-label & sửa thêm **300–400 ảnh VN Batch 2** (từ kho 10.157 ảnh VN + cảnh đêm/mưa/Hard Negatives) → Train V2 (`75 epochs`, `cos_lr=True`, `copy_paste=0.1`) → Nộp lượt `#2–4`.
- [ ] **V3 (`20:00 – 01:30`):** Viết & chạy **Script #5 (Active Learning lọc 250 ảnh VN khó nhất bằng `best.pt` V2 + Targeted Copy-Paste bơm lớp hiếm lên `≥ 300 instances/lớp`)** → Train V3 (`80 epochs`) → Nộp lượt `#5–7`.
- [ ] **V4 (`02:00 – 07:30`):** Data Cleaning + chạy song song **V4-A (Full 100 ep)** và **V4-B (2-Stage Domain Fine-Tune 20 ep cuối `lr0=0.0005` trên 80% VN)** → Nộp lượt `#8–12`.
- [ ] **Chốt bài & Slide (`07:30 – 09:30`):** Dừng nộp trước `09:30`, **tick chọn 2 bài Private LB**, gom gói tái lập (`train_and_export.ipynb`, `dataset_v4.zip`, `run_report.json`), hoàn thiện **Slide 5 phần** cho `11:30` thuyết trình!
