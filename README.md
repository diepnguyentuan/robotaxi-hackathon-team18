# AIInAction RoboTaxi Data-Centric Competition — Đội 18 (`TEAM_ID = 23`)

Kho trung tâm gom toàn bộ **tài liệu chính thức**, **kế hoạch tác chiến 24 giờ**, **thiết kế logic Data Engine**, **Label Contract & Provenance Log**, và **bộ script Python thực chiến** chạy trên **Kaggle / Google Colab** cho Đội 18.

---

## Thông số cứng của Cuộc thi (Không được sai lệch)
- **Đội:** Đội 18 – Nguyễn Văn Bảo · **Seed bắt buộc:** `TEAM_ID = 23`
- **Bài toán:** Instance Segmentation **11 lớp** (định dạng YOLO-seg polygon, tọa độ `[0, 1]`, tối thiểu 3 đỉnh)
- **Model khóa cứng:** `YOLO26n-seg` (`ultralytics==8.4.214`, `imgsz=640`, xuất ONNX FP32 `opset=17`, `simplify=True`, **không** `half`, **không** `nms`)
- **Chấm điểm:** `Final Rank = 50% Private LB (100 × mask mAP@[.5:.95]) + 50% Thuyết trình Giải pháp`
- **Deadline đóng nộp bài:** **`10:30 28/11/2026`** (Chốt nộp trước `09:30` & chọn **2 bài** cho Private LB!)
- **Công bố Private:** `11:00 28/11/2026` · **Thuyết trình:** `11:30 28/11/2026`

---

## Cấu trúc Project

```text
robotaxi-hackathon-team18/
├── README.md                              # Hướng dẫn tổng thể & Quickstart
├── requirements.txt                       # Thư viện Python cho Local / Colab / Kaggle
├── docs/
│   ├── official/
│   │   ├── labeling_guideline.md          # Quy ước gán nhãn chính thức từ BTC
│   │   └── train_and_export.ipynb         # Starter Notebook chính thức từ BTC (v3.0.0)
│   ├── 01_championship_blueprint.md       # Kế hoạch 24h, nguồn Public Datasets, Best Choice GPU
│   ├── 02_data_engine_architecture.md     # Phân tích logic & thiết kế kỹ thuật từng Script
│   ├── 03_team18_label_contract.md        # Label Contract, Edge-Case Log & Bảng kê Public Datasets (Slide)
│   └── 04_experiment_log.md               # Bảng theo dõi V1->V4 + run_report.json (Slide)
├── configs/
│   ├── classes.yaml                       # 11 lớp cố định + bảng ánh xạ Mapillary Vistas v2 & IDD
│   └── vn_youtube_sources.yaml            # Danh sách video YouTube 4K VN tách riêng VAL vs TRAIN
├── scripts/
│   ├── common/
│   │   └── seg_utils.py                   # Lõi hình học: dhash, area@640, Rider-Bike Union, Non-Overlap, YOLO-seg I/O
│   ├── 01_vn_frame_harvester.py           # [Ưu tiên #1] Cắt frame video VN + lọc blur + lọc dhash + tách scene Val/Train
│   ├── 02_auto_prelabel_sam2.py           # [Ưu tiên #2] Pre-label YOLO + YOLO-World + SAM2 + Rider-Bike Union -> CVAT zip
│   ├── 03_public_converter.py             # [Ưu tiên #3] Chuyển Mapillary Vistas v2 (HF) & IDD (Kaggle) -> 11 lớp + provenance CSV
│   └── 04_auto_qc_and_packager.py         # [Ưu tiên #4] Auto-QC (<64px², shapely fix, khử overlap, check leakage) -> dataset_vX.zip
├── notebooks/
│   └── colab_kaggle_runner.ipynb          # Notebook 1-click chạy toàn bộ pipeline trên Kaggle / Google Colab
└── tests/
    └── test_seg_utils_and_qc.py           # Bộ kiểm thử tự động xác minh 100% logic hình học & QC
```

---

## Quy trình Chạy Nhanh (Quickstart)

1. **Đọc thiết kế logic trước khi chạy:** xem `docs/02_data_engine_architecture.md`.
2. **Bước A — Gom ảnh VN thật ngay tại phòng (Script #1):**
   Điền link video dashcam 4K VN vào `configs/vn_youtube_sources.yaml` (nhớ tách riêng `val_videos` và `train_videos`), sau đó chạy:
   ```bash
   python scripts/01_vn_frame_harvester.py --config configs/vn_youtube_sources.yaml --out data/raw/vn_frames
   ```
3. **Bước B — Chạy Pre-label SAM2 trên GPU Kaggle/Colab để nạp vào CVAT (Script #2):**
   ```bash
   python scripts/02_auto_prelabel_sam2.py --images data/raw/vn_frames/val --out data/processed/cvat_val_prelabel --use-sam2 --use-yoloworld
   ```
4. **Bước C — Chuẩn hóa Public Dataset (IDD trên Kaggle / Mapillary Vistas v2 trên HF) (Script #3):**
   ```bash
   python scripts/03_public_converter.py --source idd --input /kaggle/input/indian-driving-dataset --out data/processed/public_idd --max-images 600
   python scripts/03_public_converter.py --source mapillary_hf --out data/processed/public_mapillary --max-images 1200
   ```
5. **Bước D — Chạy Auto-QC & Đóng gói `dataset_vX.zip` trước MỌI lần train (Script #4):**
   ```bash
   python scripts/04_auto_qc_and_packager.py \
     --train-dirs data/processed/public_mapillary data/processed/public_idd data/processed/cvat_vn_train \
     --val-dir data/processed/cvat_golden_val \
     --out-zip data/processed/dataset_v1.zip
   ```
6. **Bước E — Train & Xuất ONNX:** Nạp `dataset_v1.zip` vào `docs/official/train_and_export.ipynb` với `TEAM_ID = 23` trên **Kaggle GPU (P100/T4×2)** hoặc **Colab L4/T4**, giữ lại `submission.zip` và `run_report.json`!
