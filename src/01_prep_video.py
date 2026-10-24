import cv2
import os
import csv
import numpy as np

# ===== Cấu hình =====
VIDEO_PATH = "D:/DATN/DATN-261-262/data/video.mp4"
OUTPUT_DIR = "D:/DATN/DATN-261-262/data/frames"
TARGET_FPS = 1.5            # ~900 cửa sổ cho video 10 phút
MIN_BLUR_PERCENTILE = 20    # bỏ 20% cửa sổ có frame tốt nhất vẫn mờ
SCORE_WIDTH = 640           # tính độ nét trên ảnh thu nhỏ cho ổn định
JPEG_QUALITY = 95

os.makedirs(OUTPUT_DIR, exist_ok=True)

def open_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Không mở được video: {VIDEO_PATH}")
    return cap

cap = open_video()
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if fps <= 0:
    raise ValueError("Không đọc được FPS")
print(f"Video: {w}x{h}, {fps:.3f} fps")

# ===== Lượt 1: tính điểm nét của mọi frame (không lưu ảnh) =====
score_h = int(SCORE_WIDTH * h / w)
scores = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (SCORE_WIDTH, score_h), interpolation=cv2.INTER_AREA)
    scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())
cap.release()

scores = np.array(scores)
n = len(scores)
print(f"Đã đọc {n} frames ({n / fps:.1f} s)")

# ===== Chọn frame nét nhất mỗi cửa sổ, rồi lọc theo phân vị =====
window = max(1, round(fps / TARGET_FPS))
chosen = [start + int(np.argmax(scores[start:start + window]))
          for start in range(0, n, window)]
chosen_scores = scores[chosen]
threshold = np.percentile(chosen_scores, MIN_BLUR_PERCENTILE)
keep = sorted(i for i in chosen if scores[i] >= threshold)
keep_set = set(keep)
print(f"{len(chosen)} cửa sổ, ngưỡng blur = {threshold:.1f}, giữ {len(keep)} frames")

# ===== Lượt 2: đọc tuần tự, chỉ decode + lưu frame được chọn =====
cap = open_video()
last = keep[-1]
saved = 0
with open(os.path.join(OUTPUT_DIR, "frames_log.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "frame_idx", "time_s", "blur_score"])
    for idx in range(last + 1):
        if idx in keep_set:
            ret, frame = cap.read()
            if not ret:
                break
            name = f"frame_{saved:05d}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, name), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            writer.writerow([name, idx, f"{idx / fps:.2f}", f"{scores[idx]:.1f}"])
            saved += 1
        else:
            if not cap.grab():
                break
cap.release()
print(f"Hoàn thành: lưu {saved} frames vào {OUTPUT_DIR}")