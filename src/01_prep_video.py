import cv2
import csv
import shutil
import numpy as np
from pathlib import Path

# ===== Cấu hình tự động theo thư mục gốc dự án =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VIDEO_PATH = DATA_DIR / "video1.mp4"
OUTPUT_DIR = DATA_DIR / "frames"

TARGET_FPS = 1.5            # Tần suất lấy mẫu cửa sổ ban đầu
MIN_BLUR_PERCENTILE = 20    # Bỏ 20% cửa sổ có độ nét thấp nhất
MIN_MOTION_DIFF = 4.5       # Ngưỡng sai khác pixel để loại bỏ các frame đứng yên trùng lặp
MAX_FRAMES = 450            # Nâng giới hạn lên 800 frames cho server GPU
MAX_IMAGE_DIM = 1920        # Chuẩn hóa cạnh dài về Full HD 1920px
SCORE_WIDTH = 640           # Kích thước ảnh thu nhỏ để tính độ nét và chuyển động
JPEG_QUALITY = 95

# Làm sạch thư mục frames cũ để tránh lẫn ảnh giữa các lần chạy
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def open_video():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise IOError(f"Không mở được video: {VIDEO_PATH}")
    return cap


cap = open_video()
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if fps <= 0:
    raise ValueError("Không đọc được FPS từ video")
print(f"Video gốc: {w}x{h}, {fps:.3f} fps")

# ===== Lượt 1: Tính điểm nét (Laplacian) và lưu thumbnail xám để lọc trùng =====
score_h = int(SCORE_WIDTH * h / w)
scores = []
thumbs = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (SCORE_WIDTH, score_h), interpolation=cv2.INTER_AREA)
    scores.append(cv2.Laplacian(small, cv2.CV_64F).var())
    thumbs.append(small)
cap.release()

scores = np.array(scores)
n = len(scores)
print(f"Đã đọc {n} frames ({n / fps:.1f} giây)")

# ===== Bước 1: Chọn frame nét nhất mỗi cửa sổ & lọc theo phân vị độ nét =====
window = max(1, round(fps / TARGET_FPS))
chosen = [
    start + int(np.argmax(scores[start:start + window]))
    for start in range(0, n, window)
]
chosen_scores = scores[chosen]
threshold = np.percentile(chosen_scores, MIN_BLUR_PERCENTILE)
sharp_candidates = sorted(i for i in chosen if scores[i] >= threshold)
print(f"Sau lọc mờ: {len(sharp_candidates)}/{len(chosen)} frames (ngưỡng blur = {threshold:.1f})")

# ===== Bước 2: Lọc bỏ các frame đứng yên (trùng lặp góc nhìn) =====
dedup_keep = []
last_thumb = None
for idx in sharp_candidates:
    if last_thumb is None:
        dedup_keep.append(idx)
        last_thumb = thumbs[idx]
    else:
        diff = float(np.mean(cv2.absdiff(thumbs[idx], last_thumb)))
        if diff >= MIN_MOTION_DIFF:
            dedup_keep.append(idx)
            last_thumb = thumbs[idx]

print(f"Sau lọc đứng yên trùng lặp (diff >= {MIN_MOTION_DIFF}): còn {len(dedup_keep)} frames")

# ===== Bước 3: Lấy mẫu đều nếu số frame vượt quá MAX_FRAMES =====
if len(dedup_keep) > MAX_FRAMES:
    sel_indices = np.linspace(0, len(dedup_keep) - 1, MAX_FRAMES, dtype=int)
    keep = [dedup_keep[i] for i in sel_indices]
    print(f"Rút gọn đều xuống {len(keep)} frames tối ưu cho COLMAP")
else:
    keep = dedup_keep

# Giải phóng bộ nhớ thumbnails
del thumbs
keep_set = set(keep)

# Tính kích thước xuất ảnh (giữ nguyên tỷ lệ, giới hạn cạnh dài <= MAX_IMAGE_DIM)
scale = min(1.0, MAX_IMAGE_DIM / max(w, h))
out_w = int(round(w * scale))
out_h = int(round(h * scale))
print(f"Kích thước ảnh xuất: {out_w}x{out_h} (scale = {scale:.3f})")

# ===== Lượt 2: Đọc tuần tự, resize và lưu các frame được chọn =====
cap = open_video()
last = keep[-1]
saved = 0
log_path = OUTPUT_DIR / "frames_log.csv"
with open(log_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "frame_idx", "time_s", "blur_score"])
    for idx in range(last + 1):
        if idx in keep_set:
            ret, frame = cap.read()
            if not ret:
                break
            if scale < 1.0:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            name = f"frame_{saved:05d}.jpg"
            cv2.imwrite(
                str(OUTPUT_DIR / name),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            writer.writerow([name, idx, f"{idx / fps:.2f}", f"{scores[idx]:.1f}"])
            saved += 1
        else:
            if not cap.grab():
                break
cap.release()
print(f"HOÀN THÀNH: Đã lưu {saved} frames chất lượng cao vào {OUTPUT_DIR}")