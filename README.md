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

# 05 - chạy thử trên dữ liệu giả
python src/05_evaluate.py --demo

# 05 - chạy với point cloud đã registration (từ 04)
python src/05_evaluate.py --pcd data/pc_output/registered.ply --bim-dir data/bim --out results
```

Kết quả của 05: `results.csv` (Detected / Partial / Missing / Out of scan, coverage, sai số),
`heatmap_deviation.ply` (đỏ = lệch > 5 cm), `bim_status.ply`, `deviation_hist.png`.

## Structure

- `data/`: dữ liệu và output (không push).
- `src/`: các bước pipeline.
- `notebooks/`: notebook demo.
