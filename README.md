# project_name

Pipeline: camera/video data → 3D reconstruction → BIM parsing → registration → progress evaluation.

## Structure

- `data/`: local datasets and generated outputs; large data should not be pushed to Git.
- `src/`: pipeline scripts.
- `notebooks/`: demonstration notebook.
- `requirements.txt`: Python dependencies.

## Cài đặt

```
python -m venv venv
venv\Scripts\activate          # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```
Open3D cần Python 3.10 hoặc 3.11.

---

## Member 2 — BIM parsing & Registration (`03_bim_parser.py`, `04_registration.py`)

### Quy ước chung (Member 1 và Member 3 cần theo)

| Quy ước | Giá trị |
|---|---|
| Đơn vị | **mét** |
| Hệ toạ độ | **hệ toạ độ của BIM**, trục **Z hướng lên** (Z = cao độ) |
| Point cloud sau căn chỉnh | `data/reg/registered_pc.ply`, đã nằm trong hệ BIM |
| Cấu kiện BIM | `data/bim/prepared/elements/*.ply` (mesh), tra cứu trong `elements_manifest.csv` |

### ⚠️ Những điều quan trọng về file BIM của đồ án

1. **Đây là mô hình KẾT CẤU, không phải kiến trúc.** Mô hình chỉ có cột, dầm, sàn, vách (chỉ ở tầng hầm), cọc, đài móng, bê tông lót, cốp pha và thép.
   **Không có cửa, cửa sổ hay tường gạch.** Vì vậy phần đánh giá tiến độ phải làm trên **cột / dầm / sàn / vách**, không phải "tường / cột / cửa" như kế hoạch ban đầu.
2. **Tên cấu kiện là mã vật liệu tiếng Việt**, ví dụ `VL.BT_COT_B40` (cột), `VL.BT_DAM_B40` (dầm), `VL.BT_SAN_B40` (sàn), `VL.BT_VACH_B40` (vách), `01.COFF_SAN` (cốp pha sàn). `03_bim_parser.py` tự phân loại các mã này vào cột `category`.
3. **Quy trình export:** Navisworks Manage → FBX → Blender → OBJ (tick *Object Groups*). Navisworks không xuất thẳng ra OBJ. File OBJ dùng đơn vị mét và trục Y hướng lên; script tự đổi sang trục Z hướng lên.
4. **Tòa nhà khoảng 16 tầng cộng tầng hầm, rộng 69 m × 118 m, gồm 6.254 cấu kiện.** Video chỉ quay một phần, nên **phải cắt BIM theo tầng mà video quay** (`--zmin/--zmax`). Nếu không, ICP dễ khớp nhầm sang tầng khác vì các tầng giống hệt nhau.
   Cao độ mặt sàn (m): tầng hầm khoảng −3.6, tầng trệt khoảng 0–1, sau đó 5.5 · 8.8 · 12.2 · 15.4 · 18.8 · 22.1 · 25.4 · 28.6 · 32.0 · 35.2 · 38.5 · 41.8 · 45.5 · 49.3 · 52.9, mái khoảng 54.5.
5. Những thứ camera không thấy được (cọc, đài móng, bê tông lót) và đồ tạm (cốp pha) **không** được đưa vào đích căn chỉnh (`bim_full_points.ply`). Chúng vẫn được xuất ra trong `elements/` và manifest.

### Bước 1 — Chuẩn bị BIM: `src/03_bim_parser.py` (chỉ cần numpy)

```
# xem danh sách cao độ sàn
python src/03_bim_parser.py --input data/bim/building.obj --list-levels

# cắt 1 tầng: --zmin = sàn tầng đó - 0.5 ; --zmax = sàn tầng trên + 0.2
python src/03_bim_parser.py --input data/bim/building.obj --out data/bim/prepared --zmin 8.3 --zmax 12.4
```

Output trong `data/bim/prepared/`:

| File | Dùng cho | Nội dung |
|---|---|---|
| `bim_full_points.ply` | Registration | Point cloud của cột/dầm/sàn/vách trong tầng đã cắt |
| `bim_full_mesh.ply` | Xem / M3 | Mesh mọi cấu kiện trong tầng đã cắt |
| `elements/XXXX_<category>_<name>.ply` | **M3** | Mesh từng cấu kiện |
| `elements_manifest.csv` | **M3** | `element_id, name, category, file, area_m2, in_registration_target, bbox_min, bbox_max` |

### Bước 2 — Căn chỉnh: `src/04_registration.py` (Open3D)

```
# (0) tự kiểm tra bằng dữ liệu giả, không cần dữ liệu thật
python src/04_registration.py demo --out data/demo --show

# (1) chọn >= 4 cặp điểm tương ứng (SHIFT + click), theo CÙNG thứ tự ở 2 cửa sổ
python src/04_registration.py pick --source data/pc_output/fused.ply --target data/bim/prepared/bim_full_points.ply --picks data/reg/picks.json

# (2) làm sạch + căn chỉnh thô (có scale) + ICP + đo sai số
python src/04_registration.py run --source data/pc_output/fused.ply --target data/bim/prepared/bim_full_points.ply --picks data/reg/picks.json --out data/reg --show
```

Quy trình bên trong `run`: lọc nhiễu thống kê → căn chỉnh đồng dạng (Umeyama, ước lượng **scale** vì point cloud COLMAP không có đơn vị thật) → ICP point-to-plane nhiều mức (0.10 → 0.05 → 0.02 m) → đo fitness / inlier RMSE trước và sau ICP.

Output trong `data/reg/`:

| File | Nội dung |
|---|---|
| `registered_pc.ply` | Point cloud đã làm sạch và nằm trong hệ BIM → **đầu vào của M3** |
| `transformation.txt` | Ma trận 4×4 (đã gồm scale) đưa point cloud COLMAP về hệ BIM |
| `cleaned_pc.ply` | Point cloud đã làm sạch, chưa căn chỉnh |
| `metrics.json` | scale, sai số của các điểm chọn, fitness/RMSE trước và sau ICP |

Các tuỳ chọn hay dùng: `--std-ratio` (nhỏ hơn thì lọc nhiễu mạnh hơn), `--voxels 0.3 0.1 0.05` (khi căn chỉnh thô còn lệch xa), `--refine-scale` (cho ICP chỉnh tiếp cả scale).

Kết quả demo: scale tìm được 2.690 (đáp án 2.703), góc xoay lệch 0.02°, fitness 0.88 → 0.999, RMSE 3.2 cm → 1.9 cm.

### Lưu ý

- Cảnh báo `[Open3D WARNING] [ViewControl] SetViewPoint() failed ...` trên Windows là vô hại.
- `data/` bị bỏ qua trong `.gitignore` (trừ các `.gitkeep`). **Không push** file BIM, frame hay point cloud lên Git.
