# Team 18 Label Contract, Edge-Case Log & Public Dataset Provenance
*(Tài liệu sống dùng để cả đội gán nhãn nhất quán 100% VÀ xuất trực tiếp sang Slide 50% điểm Thuyết trình)*

---

## 1. Bảng Minh Bạch Public Datasets (Bắt buộc có trên Slide)

| # | Tên Dataset | Đường dẫn (URL) | Giấy phép (License) | Số ảnh dùng (Train) | Ánh xạ sang 11 lớp & Xử lý kỹ thuật |
|---:|---|---|---|---:|---|
| 1 | **Mapillary Vistas v2** | `https://huggingface.co/datasets/candylion/mapillary-vistas-v2` (gốc: `https://www.mapillary.com/dataset/vistas`) | Mapillary Vistas Research License (Non-Commercial / Academic) | *(điền từ `public_provenance.csv`)* | Đủ 11 lớp gồm `traffic-sign--front/back`, `trailer`→`Other vehicle`; chạy thuật toán **Rider-Bike Polygon Union** tạo `Motorcyclist` (5) & `Cyclist` (7). |
| 2 | **Indian Driving Dataset (IDD)** | `https://www.kaggle.com/datasets/manjotpahwa/indian-driving-dataset` (gốc: `http://insaan.iiit.ac.in/`) | IIIT Hyderabad Academic / Non-Commercial | **600 ảnh** *(quét 8.089 ảnh, 14.910 instances đủ 11 lớp, imbalance 21.0x)* | `autorickshaw`/`caravan`/`trailer`→`Other vehicle` (3: 2.833); chạy **Rider-Bike Union** cho `Motorcyclist` (5: 2.043) & `Cyclist` (7: 135); `Sign` (9: 660), `Traffic light` (10: 222). |
| 3 | **Vietnam Traffic Sign Images** *(ảnh đường phố VN)* | `https://huggingface.co/datasets/star092304/Traffic-sign-detection-VietNam` | CC-BY-NC-4.0 | **200 Val** (`/valid/`) + **300 Train** (`/train/`, lô 1 đã pre-label 200 ảnh) | Lấy ảnh đường phố VN + GT box biển báo prompt qua **SAM2.1-b** tạo polygon `Sign` (9), kết hợp `YOLO11x-seg` + `YOLO-Worldv2` + Rider-Bike Union + QC tay trên CVAT. |
| 4 | **VN Dashcam 4K Frames** *(tự trích lọc)* | *(điền danh sách URL YouTube đã dùng trong `configs/vn_youtube_sources.yaml`)* | Public / CC video frames trích lọc | *(điền số ảnh Train & Golden Val)* | Cắt cách 4–5s, lọc `dhash > 8`, tách biệt 100% video Val vs Train; pre-label SAM2 + Dual-QC CVAT. |

---

## 2. Quy Ước Gán Nhãn Nhanh Cho Toàn Đội (Tóm tắt từ `docs/official/labeling_guideline.md`)

### 5 Quy tắc "Bất Di Bất Dịch" (Vi phạm là mất điểm ngay)
1. **Motorcyclist (5) vs Motorcycle (4):** Hễ có **≥ 1 người ngồi trên yên** (kể cả dừng đèn đỏ, ngồi chờ, mặc áo mưa, chở 2–3 người + hàng buộc chắc) → **1 mask gộp duy nhất** (`5`). Xe không người ngồi (đang đỗ / đang dắt) → `Motorcycle` (`4`), người dắt tách riêng là `Pedestrian` (`8`).
2. **Cyclist (7) vs Bicycle (6):** Tương tự xe máy. **Xe máy điện không bàn đạp** thuộc nhóm `Motorcycle`/`Motorcyclist`; chỉ xe có bàn đạp/khung xe đạp mới thuộc `Bicycle`/`Cyclist`.
3. **Đầu kéo & Container:** Phần đầu kéo (cabin) = `Truck` (`2`); phần rơ-moóc / container kéo theo = **`Other vehicle` (`3`)** (2 instance riêng!). Ba gác, xích lô, xe công trình = `Other vehicle` (`3`).
4. **Sign (9) & Traffic light (10):**
   - `Sign`: Chỉ vẽ **mặt biển báo** (KHÔNG vẽ cột), vẽ **cả mặt sau (lưng biển)**, nhiều tấm trên 1 cột = mỗi tấm 1 instance. **Cấm vẽ** biển quảng cáo, biển tên shop, số nhà.
   - `Traffic light`: Vẽ **nguyên hộp vỏ đèn** (gồm đèn người đi bộ & đếm ngược gắn liền), vẽ cả trước & sau. **Cấm vẽ** đèn xe, đèn đường, đèn LED trang trí.
5. **Chỉ vẽ phần nhìn thấy, KHÔNG chồng lấn, bỏ vật quá nhỏ:**
   - Vật trước giữ pixel, vật sau chỉ vẽ phần lộ ra.
   - Bỏ vật có cạnh ngắn bbox `< 1.25%` cạnh dài ảnh (~16 px ở ảnh 1280 px; tương đương `< 64 px²` ở scale 640).
   - Bỏ vật sau kính showroom/cửa hàng, hành khách ngồi trong ô tô, hình người/xe trên áp phích, động vật, xe đẩy tay không động cơ.

---

## 3. Team 18 Edge-Case Log (Ghi lại ca khó đường Việt Nam để đưa lên Slide)

| Mã ca | Tình huống thực tế gặp trên ảnh VN | Quyết định thống nhất của Đội 18 | Tham chiếu Guideline |
|---|---|---|---|
| `EC-01` | Xe máy chở hàng cồng kềnh (sọt hàng, thùng xốp buộc sau yên) | Hàng buộc chắc trên xe: gộp vào cùng mask `Motorcyclist` (5). | Mục 4, bảng tình huống |
| `EC-02` | Người ngồi trên xe máy đang đỗ ở vỉa hè bấm điện thoại | Đã ngồi trên yên → gán `Motorcyclist` (5) (1 mask gộp). | Mục 3, Motorcyclist |
| `EC-03` | Xe 16 chỗ Ford Transit / Solati trên đường phố VN | Từ 16 chỗ trở lên → `Bus` (1); xe 7–15 chỗ dạng van nhỏ → `Car` (0). | Mục 3, Bus vs Car |
| `EC-04` | Cụm cột đèn giao thông ngã tư VN có 1 hộp đèn 3 màu + 1 hộp đếm ngược tách rời bên cạnh | Gắn liền trong cùng 1 vỏ hộp = 1 instance; tách thành 2 hộp vỏ riêng = 2 instance `Traffic light` (10). | Mục 3, Traffic light |
| `EC-05` | *(Đội bổ sung thêm ca mới gặp trong quá trình QC CVAT...)* | | |
