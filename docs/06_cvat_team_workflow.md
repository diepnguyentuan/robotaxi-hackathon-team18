# Kế Hoạch & Hướng Dẫn Thực Chiến Luồng 2 Trên CVAT (Đội 18)

Tài liệu cầm tay chỉ việc từng cú click cho Đội 18 bắt đầu từ trạng thái **chưa tạo gì trên CVAT**, đã chuẩn bị sẵn mọi file sạch tại thư mục:
`D:\2026\VinUni_Lv2\robotaxi-hackathon-team18\data\processed\step3_cvat_ready\`

---

## 1. Các File Đã Chuẩn Bị Sẵn Cho CVAT (Không Cần Tự Tách/Sửa)

| File / Thư mục trong `step3_cvat_ready/` | Dùng để làm gì trên CVAT? |
|---|---|
| `cvat_11_labels_raw.json` | Copy-paste 1 lần vào tab **Raw** khi tạo Project CVAT → tạo đủ **11 lớp polygon chuẩn chính tả + màu dễ nhìn trong 3 giây** (không bao giờ gõ nhầm!). |
| `val_200_images/` *(200 ảnh `.jpg`)* | Kéo thả tạo **Task 1 (`Golden_VN_Val_200`)**. |
| `val_200_annotations_for_cvat.zip` *(~150 KB)* | Upload vào Task 1 bằng **`Ultralytics YOLO Segmentation 1.0`** (đã tạo sẵn `train.txt` + `data.yaml` chuẩn 100% CVAT!). |
| `train_b1_200_images/` *(200 ảnh `.jpg`)* | Kéo thả tạo **Task 2 (`VN_Train_Batch1_200`)**. |
| `train_b1_200_annotations_for_cvat.zip` *(~150 KB)* | Upload vào Task 2 bằng **`Ultralytics YOLO Segmentation 1.0`**. |

---

## 2. Quy Trình Setup CVAT Trong 5 Phút (Leader Làm 1 Lần Cho Cả Nhóm)

### Bước A — Tạo Project 11 lớp (30 giây)
1. Mở CVAT (`https://cvat.edufun.ai/` để cả nhóm làm chung online, hoặc CVAT local nếu làm cùng mạng).
2. Vào mục **Projects** → bấm **`+`** → **Create a new project**.
3. Đặt tên: `Team18_RoboTaxi_11Classes`.
4. Ở mục **Labels**, bấm chuyển từ tab `Constructor` sang tab **`Raw`** → mở file `step3_cvat_ready/cvat_11_labels_raw.json`, **copy toàn bộ dán vào** → bấm **Done** → **Submit & Open**.
   - Kiểm tra nhanh đủ 11 nhãn: `Car`, `Bus`, `Truck`, `Other vehicle`, `Motorcycle`, `Motorcyclist`, `Bicycle`, `Cyclist`, `Pedestrian`, `Sign`, `Traffic light`.

### Bước B — Tạo Task 1 (`Golden_VN_Val_200`) & Chia Job Song Song (2 phút)
1. Trong Project `Team18_RoboTaxi_11Classes`, bấm **`+`** → **Create a new task**.
2. Đặt tên: `Task1_Golden_VN_Val_200`.
3. Mục **Select files**: mở thư mục `step3_cvat_ready/val_200_images/`, nhấn `Ctrl+A` chọn toàn bộ 200 ảnh kéo thả vào.
4. **CỰC QUAN TRỌNG — Chia Job cho cả nhóm làm song song:** Bấm mở **`Advanced configuration`** → điền **`Segment size = 40`** (nếu đội 5 người → tạo đúng 5 Job) hoặc **`50`** (nếu đội 4 người → tạo đúng 4 Job).
5. Chọn Image quality = `95` → bấm **Submit & Open**.
6. Khi Task tạo xong, bấm nút **`Actions`** (góc trên bên phải Task) → **`Upload annotations`** → chọn format **`Ultralytics YOLO Segmentation 1.0`** → chọn file `step3_cvat_ready/val_200_annotations_for_cvat.zip` → chờ 5 giây là toàn bộ **1.750 polygon SAM2** hiện lên sẵn trên 200 ảnh!

### Bước C — Tạo Task 2 (`VN_Train_Batch1_200`) (2 phút)
Làm y hệt Bước B nhưng với:
- Tên Task: `Task2_VN_Train_Batch1_200`
- Ảnh: `step3_cvat_ready/train_b1_200_images/` (`Segment size = 40` hoặc `50`)
- Nhãn pre-label: `step3_cvat_ready/train_b1_200_annotations_for_cvat.zip` (**1.639 polygon SAM2**)

*(Mẹo: Chụp 1 ảnh kiểm tra khung đầu tiên của Task 1 xem polygon có khớp đúng từng xe/biển báo không — nếu khớp 100% là phân công giao Job cho cả nhóm ngay!)*

---

## 3. Chiến Thuật Phân Công & Tối Ưu Tốc Độ Nhóm (40–50 Ảnh / Người / Giờ)

### Thứ tự ưu tiên tuyệt đối
1. **Giờ 1 (`12:20 – 13:20`): Dồn 100% quân số chốt xong `Task 1: Golden_VN_Val_200`**
   - Mỗi người nhận 1 Job (40–50 ảnh) → sửa xong trong ~45 phút.
   - **Dual-QC (Đổi chéo Job trong 15 phút cuối):** Người A kiểm tra nhanh Job của Người B, Người B kiểm tra của Người C... Vì đây là **tập Val chuẩn (la bàn đo điểm thật)** nên phải đạt độ sạch **99%+**!
2. **Giờ 2 (`13:20 – 14:20`): Cả nhóm dồn sang `Task 2: VN_Train_Batch1_200`**
   - Lúc này **V1 Baseline trên Colab đã chấm xong (`~13:15`)** và có **Bảng AP từng lớp ở Bước 5**!
   - Leader gửi ngay **Top 3 lớp yếu nhất của V1** vào nhóm chat → anh em khi sửa Task 2 sẽ **soi siêu kỹ đúng 3 lớp yếu đó** (tạo bước nhảy điểm cực mạnh ở V2!).

### Quy trình "4 Thao Tác 30 Giây / Ảnh" trên CVAT (Đừng vẽ lại từ đầu!)
Khi mở 1 ảnh đã có sẵn pre-label SAM2, mắt quét đúng 4 thứ theo thứ tự sau:

1. **Quét XÓA lỗi False Positive (5 giây — phím `Delete`):**
   - Xóa sạch **biển hiệu cửa hàng, bảng quảng cáo, băng rôn, số nhà** mà YOLO-World lỡ bắt nhầm thành `Sign`.
   - Xóa đèn đường, đèn pha xe, LED trang trí lỡ bắt thành `Traffic light`.
   - Xóa vật thể sau kính showroom, ảnh người/xe trên áp phích, động vật, vật quá nhỏ `< 16 px`.
2. **Quét ĐỔI NHÃN cặp dễ nhầm (10 giây — click polygon → đổi label):**
   - Xe máy có người ngồi (kể cả dừng đèn đỏ, ngồi chờ) → **`Motorcyclist`** (phải phủ **cả người + xe thành 1 mask**! Nếu pre-label lỡ tách thành `Motorcycle` + `Pedestrian`, xóa 1 cái rồi dùng SAM/kéo đỉnh gộp lại thành 1 `Motorcyclist`).
   - Xe máy đỗ không người hoặc đang dắt → **`Motorcycle`** + người dắt tách riêng là **`Pedestrian`**.
   - Xe đạp có người đi → **`Cyclist`** (1 mask gộp); xe đạp đỗ → **`Bicycle`**.
   - Xe đầu kéo kéo container → Cabin = **`Truck`**, Container/rơ-moóc = **`Other vehicle`**. Ba gác/xích lô/xe công trình = **`Other vehicle`**.
   - Xe 16 chỗ trở lên = **`Bus`**; xe 7–15 chỗ = **`Car`**.
3. **Quét BỔ SUNG vật bị sót bằng SAM / Polygon (10 giây):**
   - Đặc biệt để ý **4 nhóm hay bị pre-label sót**:
     - Biển báo nhìn từ **mặt sau (lưng biển)** hoặc nhiều biển nhỏ trên cùng 1 cột → chỉ vẽ **mặt biển**, bỏ cột.
     - **Hộp đèn giao thông** (kể cả hộp đèn người đi bộ & đếm ngược gắn liền) → vẽ **cả hộp vỏ**, không vẽ riêng bóng sáng.
     - **Người đi bộ ở xa** (vẫn lớn hơn ~16 px cạnh ngắn) và **xe đạp / ba gác**.
   - **Phím tắt tăng tốc x3 trên CVAT:**
     - `F` / `D`: Chuyển ảnh tiếp / ảnh trước.
     - `N`: Vẽ xong polygon mới (bấm `Shift` kéo để vẽ nhanh hoặc dùng **AI Tools → Interactors → SAM** bấm 1 click chuột trái vào vật là tự bo viền siêu sát!).
     - `Ctrl + Z`: Hoàn tác · `Ctrl + S`: Lưu (nhớ bấm lưu sau mỗi 5–10 ảnh!).
     - `Shift + click` vào cạnh polygon để thêm đỉnh, `Alt + click` vào đỉnh để xóa đỉnh.
4. **Kiểm tra biên & chống chồng lấn (5 giây):**
   - Không cần ngồi tỉa từng pixel chồng lấn quá lâu ở những chỗ chênh nhau nhẹ — **Script #4 (`04_auto_qc_and_packager.py`) sẽ tự động khử chồng lấn theo thứ tự độ sâu Z-order** cho đội! Chỉ cần đảm bảo **viền ngoài của mỗi vật ôm sát vật thể, không thừa nền và không trùm lên bóng đổ**.

---

## 4. Cách Export Sạch Từ CVAT Để Chạy V2 (Cực Quan Trọng!)

Khi hoàn tất Task 1 và Task 2:
1. Vào từng Task → bấm **`Actions`** → **`Export task dataset`**.
2. Chọn Format: **`Ultralytics YOLO Segmentation 1.0`**.
3. **BẮT BUỘC TICK:** **`Save images = ON`** (để file zip chứa cả ảnh lẫn nhãn, đảm bảo khớp 100% tên file!).
4. Đặt tên file tải về:
   - `cvat_golden_val_clean.zip`
   - `cvat_vn_train_b1_clean.zip`
5. Upload 2 file zip này lên Google Drive chung vào thư mục:
   `Colab Notebooks/robotaxi-hackathon-team18/data/processed/cvat_clean/`
6. Nhắn mình một câu **"đã up CVAT sạch"** → mình chạy **Script #4** kiểm tra `qc_report.md` (0 lỗi, 0 leakage, kiểm tra đúng thứ tự ID 0→10) và lập tức train **V2**!
