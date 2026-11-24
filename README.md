# DATN-261-262 — Đánh giá tiến độ thi công bằng video đối chiếu BIM

Pipeline: video → frames (01) → COLMAP point cloud (02) → BIM parser (03) → registration (04) → evaluation (05).

## Cài đặt (Windows)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dữ liệu (không push lên Git)

Đặt vào `data/`: `data/bim/PLM.ifc`, video vào `data/`, point cloud vào `data/pc_output/`.
Đơn vị thống nhất: **mét**, cùng hệ toạ độ với IFC.

## Chạy toàn bộ pipeline

```powershell
python run_pipeline.py --check     # xem trạng thái từng bước (chưa code / thiếu file / sẵn sàng)
python run_pipeline.py             # chạy mọi bước chạy được, theo thứ tự 01 -> 05
python run_pipeline.py --steps 3 5 # chỉ chạy bước 3 và 5
python run_pipeline.py --demo      # chạy bước 5 trên dữ liệu giả
```

Đường dẫn và tham số của mọi bước nằm trong `config.yaml`. Khi viết xong script của mình,
mỗi thành viên chỉ cần sửa phần `args` của bước đó trong `config.yaml`.

## Chạy từng bước

```powershell
# 03 - xem các tầng, rồi xuất cột/dầm/sàn của một tầng ra data/bim/
python src/03_bim_parser.py --list-storeys
python src/03_bim_parser.py --ifc data/bim/PLM.ifc --storey "TẦNG 2-"

# 04 - căn chỉnh point cloud vào BIM (xem mục "Bước 04" bên dưới)
python src/04_registration.py pick --source data/pc_output/colmap.ply --target data/bim/bim_merged.ply --picks data/reg/picks.json
python src/04_registration.py --source data/pc_output/colmap.ply --target data/bim/bim_merged.ply --picks data/reg/picks.json --out data/pc_output/registered.ply --crop-margin 2

# 05 - chạy thử trên dữ liệu giả
python src/05_evaluate.py --demo

# 05 - chạy với point cloud đã registration (từ 04)
python src/05_evaluate.py --pcd data/pc_output/registered.ply --bim-dir data/bim --out results
```

Kết quả của 05: `results.csv` (Detected / Partial / Missing / Out of scan, coverage, sai số),
`heatmap_deviation.ply` (đỏ = lệch > 5 cm), `bim_status.ply`, `deviation_hist.png`.

## Bước 04 — Registration (Member 2)

Point cloud COLMAP **không có tỷ lệ thật** và nằm ở hệ toạ độ tuỳ ý. `src/04_registration.py` đưa nó về
hệ toạ độ của BIM (mét): **căn chỉnh thô** từ ≥ 3 cặp điểm chọn tay (thuật toán Umeyama, tính cả scale) →
**ICP point-to-plane** nhiều mức (thô → mịn) → đo fitness / inlier RMSE trước và sau ICP.

| Lệnh | Việc |
|---|---|
| `demo` | Tự kiểm tra trên căn phòng giả có đáp án |
| `pick` | Mở cửa sổ 3D, chọn **cùng thứ tự** 4–6 điểm trên point cloud rồi trên BIM → `picks.json`. In sai số từng cặp và cảnh báo cặp nghi chọn sai |
| `run` (hoặc không ghi lệnh con, như `run_pipeline.py` gọi) | Làm sạch + căn chỉnh thô + ICP → `registered.ply` |
| `perturb` | Làm lệch một cloud **đã khớp sẵn** với BIM + lưu đáp án, để đo sai số thật bằng `run --gt` |

- **Đầu vào `--source`**: `.ply` (kể cả loại có element lạ / header Windows), `.pcd`, `.xyz/.txt/.csv/.pts`,
  `.las/.laz` (cần `pip install laspy`), **thư mục mô hình COLMAP** (`sparse/0`, tự lọc điểm xấu và điểm rác
  dính sát camera do chữ in trên video), mesh. Đơn vị khác mét → `--source-scale` (vd. mm: `0.001`).
- **Đích `--target`**: `data/bim/bim_merged.ply` của bước 03 (mesh → tự lấy mẫu điểm trên bề mặt).
- **Đầu ra** khi `--out data/pc_output/registered.ply`: file đó + thư mục `data/pc_output/registration/`
  (`transformation.txt` ma trận 4×4 có scale, `metrics.json`, `cleaned_pc.ply`).
- Tuỳ chọn hay dùng: `--crop-margin 2` (bỏ cây cối / giàn giáo ngoài hộp bao BIM ± 2 m), `--voxels 0.5 0.2 0.1`
  (khi căn chỉnh thô còn lệch xa, hoặc cloud nhiễu), `--show` (xem kết quả chồng lên BIM).
- **Chọn điểm**: góc sắc nét có ở **cả hai** bên (góc mái, chân cột, góc sàn), rải rộng, **không cùng một mặt
  phẳng**; tránh giàn giáo, cây, chi tiết lặp lại. Sau `pick`, scale phải hợp lý và không có cảnh báo "nghi cặp …".

### Đã kiểm chứng trên dữ liệu công khai Schependomlaan

Bộ dữ liệu: TU Eindhoven và cộng sự, **CC BY 4.0** —
https://github.com/openBIMstandards/Archive-DataSetSchependomlaan (DOI 10.17605/OSF.IO/NE2YU).
Point cloud SfM tuần 30 đã nằm sẵn trong hệ BIM → làm lệch (scale 0,37, xoay 35°, dịch 5) rồi căn chỉnh lại
bằng 5 cặp điểm + ICP: **scale sai 2,4 %, lệch góc 1,0°, độ lệch điểm trung bình 0,45 m** — cùng bậc với
độ chênh giữa công trình thực tế tuần 30 và mô hình thiết kế (mái thấp hơn 0,2–0,7 m).

```powershell
python src/03_bim_parser.py --ifc data/schependomlaan/schependomlaan.ifc --out data/bim_schep --types IfcWall IfcSlab IfcColumn IfcBeam
python src/04_registration.py perturb --source data/schependomlaan/week30_sfm_m.ply --target data/bim_schep/bim_merged.ply --out data/schep_test/week30_perturbed.ply
python src/04_registration.py pick --source data/schep_test/week30_perturbed.ply --target data/bim_schep/bim_merged.ply --picks data/schep_test/picks.json
python src/04_registration.py run --source data/schep_test/week30_perturbed.ply --target data/bim_schep/bim_merged.ply --picks data/schep_test/picks.json --out data/schep_test/reg --voxels 0.5 0.2 0.1 --crop-margin 2 --gt data/schep_test/week30_perturbed_gt_transform.txt
```


## Structure

- `data/`: dữ liệu và output (không push).
- `src/`: các bước pipeline.
- `notebooks/`: notebook demo.
