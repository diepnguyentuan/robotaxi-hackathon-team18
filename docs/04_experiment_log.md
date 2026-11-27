# Nhật Ký Thí Nghiệm & Cải Tiến Data-Centric (Đội 18 · `TEAM_ID = 23`)

> **Quy tắc vàng:** Mọi vòng đều giữ nguyên **`TEAM_ID = 23`** và tập **Golden VN Val cố định** để mọi thay đổi điểm đều so sánh được công bằng. Sau mỗi vòng, lưu `submission.zip` và `run_report.json` vào thư mục `runs/vX/` (và backup Google Drive).

## 1. Bảng Theo Dõi Vòng Lặp (Đưa thẳng lên Slide Thuyết trình)

| Vòng | Thời gian | Thay đổi dữ liệu / Huấn luyện cốt lõi | Train (Public / VN / CP) | Golden Val `mAP@[.5:.95]` | AP50 / AP75 | Top 3 lớp yếu nhất (Bước 5) | Public LB | Trạng thái chọn Private |
|---|---|---|---:|---:|---:|---|---:|---|
| **V1** | `27/11 12:28` | Baseline thông pipeline (`dataset_v1.zip` 282MB, 0 leak, imbalance 30.7x): 1.200 IDD + 180 VN Pre-label (50 ep) | `1200 / 180 / 0` *(1380)* | *(đang train)* | — | — | — | [ ] |
| **V2** | `27/11 __:__` | +400 ảnh VN thật (đêm/mưa/ngã tư/hiếm) + 60 Hard Negatives | `1500 / 550 / 0` | — | — | — | — | [ ] |
| **V3** | `28/11 __:__` | +300 ảnh Active Learning + Copy-Paste lớp hiếm + Non-Overlap QC | `1500 / 850 / 200` | — | — | — | — | [ ] |
| **V4-A** | `28/11 __:__` | Final Clean V4 + Train 100 epochs (`cos_lr`, `copy_paste=0.15`) | `1500 / 900 / 250` | — | — | — | — | [ ] |
| **V4-B** | `28/11 __:__` | 2-Stage Domain Fine-Tune (20 ep cuối `lr0=0.0005` trên 80% VN) | `80% VN / 20% Rare` | — | — | — | — | [ ] |

---

## 2. Nhật Ký Phân Tích Lỗi Bước 5 (Error Analysis Log)

### Sau Vòng 1 (V1)
- **3 lớp yếu nhất:** `...`
- **Cặp nhầm lẫn phổ biến nhất:** `...`
- **Hành động dữ liệu cho V2:** `...`

### Sau Vòng 2 (V2)
- **3 lớp yếu nhất:** `...`
- **Cặp nhầm lẫn phổ biến nhất:** `...`
- **Hành động dữ liệu cho V3:** `...`

### Sau Vòng 3 (V3)
- **3 lớp yếu nhất:** `...`
- **Cặp nhầm lẫn phổ biến nhất:** `...`
- **Hành động làm sạch dữ liệu cho V4:** `...`
