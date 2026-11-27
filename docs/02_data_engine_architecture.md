# Phân Tích Logic & Thiết Kế Kiến Trúc Data Engine (Đội 18)

Tài liệu này phân tích kỹ **bản chất kỹ thuật, đầu vào/đầu ra, công thức toán học, các trường hợp biên (edge cases) và lý do chọn giải pháp** cho từng module **trước khi triển khai mã nguồn**.

---

## 1. Các Ràng Buộc Toán Học Từ Hệ Thống Chấm (`train_and_export.ipynb`)

Mọi script trong dự án phải bám sát 100% các hằng số nội bộ đã giải phẫu từ `detscore` trong notebook chính thức:

1. **Thang đo diện tích tối thiểu (`MIN_GT_AREA_PX = 64.0` ở `imgsz = 640`):**
   - Server letterbox ảnh gốc kích thước $(W, H)$ về hình vuông $640 \times 640$ với hệ số thu phóng:
     $$r = \min\left(\frac{640}{W}, \frac{640}{H}\right)$$
   - Diện tích của một đa giác ở thang `640` được tính chính xác là:
     $$A_{640} = A_{\text{norm}} \times W \times H \times r^2$$
   - *Hệ quả thiết kế:* Nếu ảnh gốc là `1280×720` ($r = 0.5$), vật thể $16 \times 16 = 256\text{ px}^2$ trên ảnh gốc đúng bằng $64\text{ px}^2$ ở thang 640. Mọi nhãn nhỏ hơn `64 px²` đều bị bộ chấm **bỏ qua** (tạo nhiễu gradient cho `YOLO26n-seg`). Do đó `seg_utils.py` bắt buộc lọc các polygon có $A_{640} < 64.0$.

2. **Chống trùng lặp & Rò rỉ Train/Val (`DUP_HAMMING_MAX = 2`):**
   - Bộ kiểm tra Bước 2 của BTC dùng **64-bit Perceptual dHash** (resize ảnh xám $9 \times 8$, so sánh pixel kề ngang `px[:, 1:] > px[:, :-1]`) và gắn cờ nếu khoảng cách Hamming $\le 2$.
   - *Hệ quả thiết kế:*
     - `01_vn_frame_harvester.py` áp ngưỡng chặt hơn gấp 4 lần (**Hamming $> 8$** khi trích frame video) + bắt buộc tách biệt thư mục video nguồn (`val_videos` vs `train_videos`).
     - `04_auto_qc_and_packager.py` quét chéo `dhash` giữa toàn bộ tập `train` và `val` với ngưỡng bảo vệ **Hamming $\le 4$** để loại bỏ 100% nguy cơ rò rỉ.

3. **Biểu diễn Multi-Polygon (Vật bị cột/dây điện cắt đôi) trong YOLO-seg `.txt`:**
   - Luật BTC: *"Đối tượng bị che bởi vật mảnh (cột điện, dây, lan can, cành cây) vẫn là một instance."*
   - Thực tế kỹ thuật: Chuẩn YOLO-seg `.txt` chỉ nhận **1 vòng polygon liên tục trên mỗi dòng**. Nếu ghi 2 mảnh thành 2 dòng riêng, Ultralytics hiểu là 2 đối tượng riêng biệt (tạo False Positive và giảm IoU!).
   - *Giải pháp trong `seg_utils.py`:* Khi một đối tượng bị tách thành `MultiPolygon` (do Rider-Bike Union hoặc trừ vùng chồng lấn):
     - Bước 1: Loại bỏ mảnh vụn nhiễu nhỏ hơn `5%` diện tích mảnh lớn nhất hoặc nhỏ hơn `32 px²` ở thang 640.
     - Bước 2: Nếu vẫn còn $\ge 2$ mảnh đáng kể, sử dụng thuật toán **Shortest-Bridge Ring Merger** (tìm cặp đỉnh gần nhất giữa 2 đa giác rồi nối thành 1 vòng duy nhất bằng đường cầu đi–về có diện tích bằng 0, giống chuẩn `merge_multi_segment` của Ultralytics) → giữ trọn vẹn **1 instance duy nhất** mà không bị lỗi hình học!

---

## 2. Thiết Kế Chi Tiết Từng Module & Lập Luận Logic

### Module 0 — `scripts/common/seg_utils.py` (Thư viện Lõi Hình Học)
- **Trách nhiệm:** Tập trung toàn bộ thuật toán hình học & kiểm chuẩn về 1 nơi duy nhất để Script #1, #2, #3, #4 dùng chung, tránh lệch logic.
- **Các hàm cốt lõi:**
  1. `compute_dhash(img)` & `hamming_distance(h1, h2)`: 64-bit dHash chuẩn, tương thích 100% với Bước 2 của Notebook BTC.
  2. `poly_area_at_640(pts_norm, w, h)`: Tính diện tích chính xác tại scale letterbox 640.
  3. `clean_polygon(pts_norm)` & `merge_multipolygon_to_single_ring(geom)`: Sửa lỗi đa giác tự cắt bằng `shapely.make_valid`, lọc mảnh rác, nối `MultiPolygon` thành 1 vòng đơn hợp lệ và đơn giản hóa đỉnh vừa phải (`simplify` giữ sắc nét biên mà không làm nặng file `.txt`).
  4. `merge_riders_and_vehicles(instances, mode)`:
     - **Chế độ `explicit_rider` (Mapillary Vistas v2 / IDD):** Vì nguồn đã có nhãn `rider`/`motorcyclist`/`bicyclist` riêng, thuật toán tìm chiếc `motorcycle`/`bicycle` gần nhất có giao nhau hoặc nằm ngay dưới chân `rider` (mở rộng bbox xe 15% theo chiều rộng & 35% lên phía trên). Hỗ trợ **nhiều người cùng ngồi trên 1 xe máy** (gom tất cả `rider` khớp cùng 1 xe vào `unary_union` thành 1 `Motorcyclist` ID `5`).
     - **Chế độ `generic_person` (Pre-label YOLO11x / COCO):** Áp dụng **4 điều kiện hình học chặt** để tránh gộp nhầm người đi bộ đứng sau xe đỗ:
       1. Tâm ngang `person_cx` nằm trong vùng `[veh_x1 - 0.15*w, veh_x2 + 0.15*w]`.
       2. Đỉnh đầu `person_y1 < veh_y1 + 0.25*veh_h` và chân `person_y2` nằm trong `[veh_y1 + 0.15*veh_h, veh_y2 + 0.15*veh_h]`.
       3. Tỉ lệ chiều cao hợp lý `0.35 <= person_h / veh_h <= 2.5` (loại bỏ người nhỏ ở xa đứng sau xe máy gần camera).
       4. Bbox giao nhau (`IoU >= 0.10` hoặc `intersection / min_area >= 0.20`) hoặc khoảng cách biên polygon `< 0.02`.
  5. `enforce_non_overlap(instances, w, h)`:
     - Thực thi nguyên tắc số 2 của BTC (*"vật phía trước giữ pixel, vật phía sau chỉ phủ phần còn lại"*).
     - **Quy tắc Z-order (sắp thứ tự từ gần đến xa):** Với camera gắn trên xe nhìn ngang mặt đường, vật có **đáy bbox `y_max` lớn hơn** (gần mép dưới ảnh hơn) nằm gần camera hơn. Riêng lớp `Sign` (9) và `Traffic light` (10) treo trên cao có `y_max` nhỏ hơn nên tự nhiên không cắt mất phần thân của xe buýt/ô tô chạy phía trước!
     - **Bảo vệ chống nuốt nhầm:** Nếu 2 instance **cùng lớp** chồng nhau `IoU >= 0.65` → loại bỏ bản trùng (dedup). Nếu khác lớp → vật phía trước giữ nguyên, vật phía sau nhận `behind_geom.difference(front_geom)`; nếu phần còn lại `< 64 px²` ở scale 640 thì loại bỏ.

---

### Script #1 — `scripts/01_vn_frame_harvester.py` (Hành động ngay lập tức #1)
- **Mục tiêu:** Cung cấp ngay kho ảnh đường phố Việt Nam thật (`val/` tách riêng `train_pool/`) **ngồi tại phòng 100%** để đội gán nhãn CVAT có việc làm ngay trong 30 phút đầu.
- **Nguồn hỗ trợ:**
  1. Video local / URL YouTube 4K dashcam VN (tự tải bằng `yt-dlp` nếu có, hoặc đọc trực tiếp file `.mp4` tải sẵn).
  2. Thư mục ảnh sẵn có (ví dụ trích từ HuggingFace `star092304/Traffic-sign-detection-VietNam` hoặc Zalo Traffic Sign).
- **Pipeline xử lý mỗi video:**
  - Bước nhảy thời gian cố định `--interval-sec 4.0` (chống lấy frame liên tiếp).
  - Bộ lọc độ nét **Laplacian Variance** (`--min-sharpness 60.0`) → tự động loại bỏ frame bị nhòe nặng do xe xóc/rẽ gắt.
  - Bộ lọc **dHash (`--min-dhash-dist 8`)** so với tất cả frame đã lưu trong cùng split → loại sạch cảnh dừng đèn đỏ đứng yên 30 giây bị lặp 7 frame giống hệt nhau!
  - Ghi log `harvest_manifest.csv` (tên file, nguồn video, timestamp, độ nét, dHash) để kiểm soát 0% leakage và chứng minh nguồn gốc trên Slide.

---

### Script #2 — `scripts/02_auto_prelabel_sam2.py` (Hành động ngay lập tức #2)
- **Mục tiêu:** Tăng tốc gán nhãn trên CVAT lên **10 lần** (40–50 ảnh/người/giờ) với viền polygon sắc nét đạt chuẩn IoU `0.75–0.95`.
- **Tại sao chọn Kiến trúc 3 tầng thuần Ultralytics (không dùng GroundingDINO phức tạp)?**
  - Cài đặt GroundingDINO trên Colab/Kaggle thường hay lỗi biên dịch CUDA C++ mất 30–60 phút vô ích.
  - Thay vào đó, **100% chạy trong hệ sinh thái `ultralytics`** (cài 1 dòng duy nhất):
    1. **Tầng 1 (Base Detector/Segmenter):** `yolo11x-seg.pt` (hoặc `yolo26x-seg.pt`) bắt cực chuẩn các lớp động (`car`, `bus`, `truck`, `motorcycle`, `bicycle`, `person`, `traffic light`).
    2. **Tầng 2 (Open-Vocabulary Booster — tùy chọn `--use-yoloworld`):** `yolov8x-worldv2.pt` chạy prompt chuyên trị các lớp COCO bị thiếu:
       - `Sign` (9): `"road traffic sign"`, `"street sign plate"`
       - `Other vehicle` (3): `"auto rickshaw"`, `"tuk tuk"`, `"cyclo"`, `"three-wheeled cargo cart"`, `"excavator"`, `"forklift"`, `"shipping container trailer"`
       - *(Đặc biệt: hỗ trợ đọc thêm `--extra-sign-boxes` nếu ảnh lấy từ bộ `star092304/Traffic-sign-detection-VietNam` để dùng trực tiếp GT box biển báo VN!)*
    3. **Tầng 3 (SAM2 Mask Refiner — tùy chọn `--use-sam2`):** Lấy toàn bộ bounding box hợp nhất từ Tầng 1 + Tầng 2 đẩy làm box-prompt vào `SAM("sam2.1_b.pt")` → sinh mặt nạ cực ôm viền cho cả 11 lớp!
  - Sau đó đi qua `merge_riders_and_vehicles(..., mode="generic_person")` → `enforce_non_overlap` → lọc `< 64 px²` → đóng gói trực tiếp định dạng `YOLO Segmentation 1.0` (`images/`, `labels/`, `data.yaml`, `obj.names`) để import 1 phát vào CVAT!

---

### Script #3 — `scripts/03_public_converter.py` (Hành động ngay lập tức #3)
- **Mục tiêu:** Tạo ngay **1.500 – 1.800 ảnh Public chuẩn 11 lớp** sạch, giàu lớp hiếm, có file chứng minh nguồn gốc (`public_provenance.csv`) cho Slide.
- **2 bộ chuyển đổi tích hợp sẵn:**
  1. **IDD Converter (`--source idd`):** Đọc trực tiếp cấu trúc `leftImg8bit/` + `gtFine/*_polygons.json` từ Kaggle (`/kaggle/input/indian-driving-dataset` hoặc Part 2).
     - Chấm điểm ưu tiên (**Rare-Class Priority Score**) từng ảnh: ảnh có `autorickshaw`/`trailer`/`caravan` (`+5`), `bicycle` (`+4`), `bus` (`+3`), `traffic light`/`traffic sign` (`+3`), `rider`+`motorcycle` (`+2`) → tự động tuyển chọn Top `N` ảnh giàu giá trị nhất thay vì lấy ngẫu nhiên toàn ô tô!
     - Chạy `merge_riders_and_vehicles(..., mode="explicit_rider")` + `enforce_non_overlap` + lọc `< 64 px²`.
  2. **Mapillary Vistas v2 Streaming Converter (`--source mapillary_hf`):**
     - Stream trực tiếp từ HuggingFace `candylion/mapillary-vistas-v2` (không tải toàn bộ 20GB!).
     - Giải mã instance mask / panoptic PNG (hỗ trợ cả định dạng Mapillary v1.2/v2.0 `instance_id = raw_pixel % 256`, `class_id = raw_pixel // 256` kèm tự kiểm tra schema ngay ảnh đầu tiên để không bị chạy mù).
     - Chấm điểm ưu tiên lớp hiếm (`Other vehicle`, `Bicycle`, `Cyclist`, `Sign back/front`, `Traffic light`, `Bus`) → chỉ lưu đúng `N` ảnh tốt nhất + xuất `public_provenance.csv`.

---

### Script #4 — `scripts/04_auto_qc_and_packager.py` (Lá chắn bắt buộc trước MỌI lần train)
- **Mục tiêu:** Biến hỗn hợp nhiều nguồn (`Public + CVAT VN Train + Hard Negatives + Golden Val`) thành 1 file `dataset_vX.zip` **0 lỗi nặng, 0 rò rỉ, 0 chồng lấn, đạt chuẩn 100% Bước 1 & Bước 2 của BTC**.
- **6 cổng kiểm định tự động:**
  1. Chuẩn hóa tên file có prefix nguồn (`map_`, `idd_`, `vn_`, `hn_`) để **không bao giờ ghi đè trùng tên** (như `0001.jpg` từ 2 nguồn khác nhau!).
  2. Đọc & kiểm tra mọi dòng `.txt`: ID thuộc `0..10`, số tọa độ chẵn `>= 6`, kẹp về `[0, 1]`, sửa đa giác lỗi bằng `clean_polygon`.
  3. Lọc mask `< 64 px²` tại scale 640 và chạy `enforce_non_overlap` (có thể bật/tắt bằng cờ `--enforce-non-overlap`).
  4. Quét chéo `dhash` toàn bộ `train` với `val` (`--leak-hamming-max 4`) và nội bộ `train` (`--dup-hamming-max 2`): **tự động loại bỏ ảnh train trùng/rò rỉ**, bảo vệ nguyên vẹn `Golden Val`.
  5. Kiểm định KPI so với ngưỡng BTC (`MIN_TRAIN_INSTANCES >= 100`, `MAX_IMBALANCE_RATIO <= 30`, `val >= 10/lớp`) và xuất bảng thống kê Markdown (`qc_report.md`) để dán thẳng vào `docs/04_experiment_log.md` và Slide.
  6. Sinh ảnh lưới kiểm tra bằng mắt (`qc_preview_train.jpg`, `qc_preview_val.jpg`) + đóng gói `dataset_vX.zip` đúng cấu trúc `images/{train,val}` + `labels/{train,val}` + `data.yaml`.
