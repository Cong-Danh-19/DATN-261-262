import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ===== Cấu hình tự động đa nền tảng (Windows & Linux) =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE = PROJECT_ROOT / "data"

# Tự động tìm COLMAP trong biến môi trường (Linux/Conda) hoặc thư mục bin nội bộ (Windows)
_local_colmap = BASE / "bin" / ("colmap.exe" if os.name == "nt" else "colmap")
COLMAP = str(_local_colmap) if _local_colmap.exists() else (shutil.which("colmap") or "colmap")

IMAGE_DIR = BASE / "frames"
VOCAB_TREE = BASE / "vocab_tree_flickr100K_words32K.bin"

WORKSPACE = BASE / "colmap_workspace"
DB_PATH = WORKSPACE / "datn.db"
SPARSE_DIR = WORKSPACE / "sparse"
DENSE_DIR = WORKSPACE / "dense"

PC_OUT_DIR = BASE / "pointclouds"
SPARSE_PLY = PC_OUT_DIR / "sparse_room.ply"
FUSED_PLY = PC_OUT_DIR / "dense_room_fused.ply"
MESHED_PLY = PC_OUT_DIR / "dense_room_meshed.ply"


def run_cmd(cmd):
    cmd_str = [str(c) for c in cmd]
    print(f"\n[RUNNING]: {' '.join(cmd_str)}")
    subprocess.run(cmd_str, check=True)


def find_best_sparse_model(sparse_root: Path) -> Path:
    """Tự động tìm model COLMAP lớn nhất (nhiều ảnh đăng ký nhất) thay vì fix cứng '0'."""
    if not sparse_root.exists():
        sys.exit(f"[LỖI] Không tìm thấy thư mục sparse: {sparse_root}")
    if (sparse_root / "cameras.bin").exists():
        return sparse_root
    subs = [p for p in sparse_root.iterdir() if p.is_dir() and (p / "images.bin").exists()]
    if not subs:
        sys.exit(f"[LỖI] Không có model COLMAP hợp lệ trong {sparse_root}")
    if len(subs) > 1:
        print(f"[CẢNH BÁO] Có {len(subs)} sub-models tách rời, đang chọn model lớn nhất.")
    return max(subs, key=lambda p: (p / "images.bin").stat().st_size)


def run_colmap_pipeline(
    from_step: int = 1,
    stop_after_undistort: bool = False,
    matcher: str = "sequential",
    use_gpu: str = "1",
    max_image_size: str = "1920"
):
    # Kiểm tra các file đầu vào bắt buộc
    if not Path(COLMAP).exists() and not shutil.which(COLMAP):
        sys.exit(f"[LỖI] Không tìm thấy lệnh COLMAP ({COLMAP}). Hãy kích hoạt conda env hoặc kiểm tra PATH!")
    if not IMAGE_DIR.exists() or not any(IMAGE_DIR.glob("*.jpg")):
        sys.exit(f"[LỖI] Thư mục ảnh {IMAGE_DIR} trống. Hãy chạy src/01_prep_video.py trước!")

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    SPARSE_DIR.mkdir(parents=True, exist_ok=True)
    DENSE_DIR.mkdir(parents=True, exist_ok=True)
    PC_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Feature Extraction (Trích xuất đặc trưng SIFT mật độ cao)
    if from_step <= 1:
        if DB_PATH.exists():
            backup_db = WORKSPACE / f"datn_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            DB_PATH.rename(backup_db)
            print(f"[INFO] Đã backup database cũ sang: {backup_db.name}")

        run_cmd([
            COLMAP, "feature_extractor",
            "--database_path", DB_PATH,
            "--image_path", IMAGE_DIR,
            "--ImageReader.camera_model", "OPENCV",
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", use_gpu,
            "--SiftExtraction.max_num_features", "8192",
            "--SiftExtraction.estimate_affine_shape", "1",
            "--SiftExtraction.domain_size_pooling", "1"
        ])

    # 2. Feature Matching (Khớp điểm tương đồng)
    if from_step <= 2:
        if matcher == "sequential":
            match_args = [
                COLMAP, "sequential_matcher",
                "--database_path", DB_PATH,
                "--SiftMatching.use_gpu", use_gpu,
                "--SequentialMatching.overlap", "15",
                "--SequentialMatching.quadratic_overlap", "1"
            ]
            if VOCAB_TREE.exists():
                match_args.extend([
                    "--SequentialMatching.loop_detection", "1",
                    "--SequentialMatching.vocab_tree_path", str(VOCAB_TREE)
                ])
            else:
                print(f"[CẢNH BÁO] Không thấy {VOCAB_TREE.name}, tắt loop_detection.")
            run_cmd(match_args)
        else:
            run_cmd([
                COLMAP, "exhaustive_matcher",
                "--database_path", DB_PATH,
                "--SiftMatching.use_gpu", use_gpu
            ])

    # 3. Sparse Reconstruction (SfM + Bundle Adjustment)
    if from_step <= 3:
        if SPARSE_DIR.exists():
            shutil.rmtree(SPARSE_DIR)
        SPARSE_DIR.mkdir(parents=True, exist_ok=True)

        run_cmd([
            COLMAP, "mapper",
            "--database_path", DB_PATH,
            "--image_path", IMAGE_DIR,
            "--output_path", SPARSE_DIR,
            "--Mapper.ba_global_max_num_iterations", "75",
            "--Mapper.ba_global_max_refinements", "3"
        ])

    best_model = find_best_sparse_model(SPARSE_DIR)
    print(f"\n[INFO] Model thưa được chọn: {best_model}")
    run_cmd([COLMAP, "model_analyzer", "--path", best_model])
    run_cmd([
        COLMAP, "model_converter",
        "--input_path", best_model,
        "--output_path", SPARSE_PLY,
        "--output_type", "PLY"
    ])

    # 4. Image Undistortion (Khử méo ống kính & đóng gói workspace cho bước Dense)
    if from_step <= 4:
        if DENSE_DIR.exists():
            shutil.rmtree(DENSE_DIR)
        DENSE_DIR.mkdir(parents=True, exist_ok=True)

        run_cmd([
            COLMAP, "image_undistorter",
            "--image_path", IMAGE_DIR,
            "--input_path", best_model,
            "--output_path", DENSE_DIR,
            "--output_type", "COLMAP",
            "--max_image_size", max_image_size
        ])

    if stop_after_undistort:
        print(f"\n[TẠM DỪNG SAU BƯỚC 4] Workspace đã sẵn sàng tại: {DENSE_DIR}")
        print("Khi có GPU, chạy tiếp lệnh: python src/02_run_colmap.py --from-step 5")
        return

    # 5. PatchMatch Stereo (Tái tạo bản đồ độ sâu dày đặc - yêu cầu GPU NVIDIA CUDA)
    if from_step <= 5:
        run_cmd([
            COLMAP, "patch_match_stereo",
            "--workspace_path", DENSE_DIR,
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.geom_consistency", "true"
        ])

    # 6. Stereo Fusion -> Đám mây điểm 3D dày đặc (Dense Point Cloud)
    if from_step <= 6:
        run_cmd([
            COLMAP, "stereo_fusion",
            "--workspace_path", DENSE_DIR,
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_path", FUSED_PLY
        ])

    # 7. Poisson Meshing -> Mô hình lưới bề mặt 3D dày đặc (Dense 3D Mesh)
    if from_step <= 7:
        run_cmd([
            COLMAP, "poisson_mesher",
            "--input_path", FUSED_PLY,
            "--output_path", MESHED_PLY,
            "--PoissonMeshing.trim", "7"
        ])

    print("\n================ HOÀN TẤT TOÀN BỘ PIPELINE ================")
    print(f"1. Sparse Point Cloud : {SPARSE_PLY}")
    print(f"2. Dense Point Cloud  : {FUSED_PLY}")
    print(f"3. Dense 3D Mesh      : {MESHED_PLY}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy toàn bộ pipeline COLMAP từ frames -> Dense 3D Model")
    parser.add_argument("--from-step", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7],
                        help="Bắt đầu từ bước mấy (1=đầu đến cuối, 5=chạy tiếp phần Dense trên GPU)")
    parser.add_argument("--stop-after-undistort", action="store_true",
                        help="Dừng sau bước 4 (dùng khi đang chạy tạm trên CPU chưa có GPU)")
    parser.add_argument("--cpu-sfm", action="store_true",
                        help="Dùng CPU cho bước 1 & 2 (nếu máy chưa lắp GPU)")
    parser.add_argument("--matcher", type=str, default="sequential", choices=["sequential", "exhaustive"],
                        help="Kiểu khớp ảnh: sequential (mặc định cho video) hoặc exhaustive")
    args = parser.parse_args()

    run_colmap_pipeline(
        from_step=args.from_step,
        stop_after_undistort=args.stop_after_undistort,
        matcher=args.matcher,
        use_gpu="0" if args.cpu_sfm else "1"
    )