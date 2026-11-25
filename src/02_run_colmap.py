import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# ===== Cấu hình (khớp với cấu trúc thư mục của bạn) =====
BASE = Path("D:/DATN/DATN-261-262/data")
IMAGE_DIR = BASE / "frames"
DB_PATH = BASE / "datn.db"
SPARSE_DIR = BASE / "sparse"            # model làm từ GUI (nằm trực tiếp trong này)
MAPPER_OUT = BASE / "sparse_auto"       # nếu chạy lại mapper, ghi ra đây để không đè kết quả cũ
PLY_OUT = BASE / "pc_output" / "sparse_room.ply"
COLMAP = BASE / "bin" / "colmap.exe"    # hoặc r"D:\...\COLMAP.bat"
USE_GPU = "0"                           # máy không có NVIDIA/CUDA


def run(cmd):
    cmd = [str(c) for c in cmd]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def find_model(root: Path) -> Path:
    """Model nằm trực tiếp trong root, hoặc trong thư mục con 0,1,2... (lấy model lớn nhất)."""
    if (root / "cameras.bin").exists():
        return root
    subs = [p for p in root.iterdir() if p.is_dir() and (p / "images.bin").exists()]
    if not subs:
        sys.exit(f"Không tìm thấy model COLMAP trong {root}")
    return max(subs, key=lambda p: (p / "images.bin").stat().st_size)


def step_extract():
    print("\n=== [1] FEATURE EXTRACTION (CPU) ===")
    run([COLMAP, "feature_extractor",
         "--database_path", DB_PATH,
         "--image_path", IMAGE_DIR,
         "--ImageReader.camera_model", "OPENCV",
         "--ImageReader.single_camera", "1",
         "--FeatureExtraction.use_gpu", USE_GPU])


def step_match():
    print("\n=== [2] SEQUENTIAL MATCHING (CPU) ===")
    run([COLMAP, "sequential_matcher",
         "--database_path", DB_PATH,
         "--SequentialMatching.overlap", "15",
         "--SequentialMatching.loop_detection", "0",
         "--FeatureMatching.use_gpu", USE_GPU])


def step_mapper():
    print("\n=== [3] MAPPER ===")
    MAPPER_OUT.mkdir(parents=True, exist_ok=True)
    run([COLMAP, "mapper",
         "--database_path", DB_PATH,
         "--image_path", IMAGE_DIR,
         "--output_path", MAPPER_OUT])


def step_export(model_root: Path):
    print("\n=== [4] THỐNG KÊ + XUẤT PLY ===")
    model = find_model(model_root)
    print("Model:", model)
    run([COLMAP, "model_analyzer", "--path", model])   # in số ảnh đăng ký, số điểm 3D, reprojection error
    PLY_OUT.parent.mkdir(parents=True, exist_ok=True)
    run([COLMAP, "model_converter",
         "--input_path", model,
         "--output_path", PLY_OUT,
         "--output_type", "PLY"])
    total = len(list(IMAGE_DIR.glob("*.jpg")))
    print(f"\nTổng ảnh đầu vào: {total}. Ghi số 'Registered images' ở trên vào bảng metric.")
    print("PLY đã lưu tại:", PLY_OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-step", type=int, default=4, choices=[1, 2, 3, 4],
                    help="1=extract 2=match 3=mapper 4=thống kê+xuất PLY (mặc định)")
    a = ap.parse_args()

    if a.from_step <= 1: step_extract()
    if a.from_step <= 2: step_match()
    if a.from_step <= 3: step_mapper()
    step_export(MAPPER_OUT if a.from_step <= 3 else SPARSE_DIR)
    print("\nHOÀN TẤT")