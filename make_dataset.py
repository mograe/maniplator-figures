import cv2
import numpy as np
import os
import math
from pathlib import Path

# --- настройки ---

VIDEOS = {
    "cone":     "cone.mp4",
    "cube":     "cube.mp4",
    "cylinder": "cylinder.mp4",
    "pyramid":  "pyramid.mp4",
    # если будет отдельный ролик без фигур:
    # "nothing": "nothing.mp4",
}

OUTPUT_DIR = "dataset_raw"         # сюда будем сохранять кадры
FRAME_STRIDE = 3                   # брать каждый 3-й кадр
MIN_AREA = 1500                    # минимальная площадь красного контура
MIN_MOVE_PX = 15                   # минимальное смещение центра между сохранёнными кадрами
MIN_FOCUS = 20                     # порог резкости (variance of Laplacian)
CROP_MARGIN = 10                   # отступ вокруг bbox, в пикселях
OUT_SIZE = (224, 224)              # размер картинки для нейросети

NOTHING_CLASS = "nothing"
NOTHING_STRIDE = 6                 # как часто сохранять nothing-кадры
SKIN_MAX_RATIO = 0.35              # макс. доля кожи внутри кропа, иначе кадр выбрасываем


# --- функции ---

def get_red_mask(frame):
    """Маска красной фигурки (чуть более мягкая по насыщенности/яркости)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # красный: убираем супер-жёсткую границу по S
    lower1 = np.array([0,   80, 40], dtype=np.uint8)
    upper1 = np.array([10,  255, 255], dtype=np.uint8)
    lower2 = np.array([170, 80, 40], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask



def get_skin_mask(frame):
    """Маска кожи в YCrCb (обычный диапазон из примеров OpenCV)."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_red_object(frame):
    """bbox, центр и маска самого большого красного объекта (или None)."""
    mask = get_red_mask(frame)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, mask

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    if area < MIN_AREA:
        # объект слишком маленький — считаем, что "ничего нет"
        return None, None, mask

    x, y, w, h = cv2.boundingRect(c)
    M = cv2.moments(c)
    cx = int(M["m10"] / (M["m00"] + 1e-6))
    cy = int(M["m01"] / (M["m00"] + 1e-6))

    return (x, y, w, h), (cx, cy), mask


def is_blurry(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < MIN_FOCUS


def process_video(label, video_path, out_root):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[{label}] Не удалось открыть видео: {video_path}")
        return

    class_dir = out_root / label
    class_dir.mkdir(parents=True, exist_ok=True)

    nothing_dir = out_root / NOTHING_CLASS
    nothing_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved_idx = 0
    nothing_saved_idx = 0
    last_center = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # пропускаем часть кадров
        if frame_idx % FRAME_STRIDE != 0:
            continue

        if is_blurry(frame):
            continue

        bbox, center, red_mask = detect_red_object(frame)

        # --- если НЕТ красной фигуры — кадр в nothing ---
        if bbox is None:
            if frame_idx % NOTHING_STRIDE != 0:
                continue
            resized = cv2.resize(frame, OUT_SIZE)
            out_path = nothing_dir / f"{NOTHING_CLASS}_{label}_{nothing_saved_idx:04d}.jpg"
            cv2.imwrite(str(out_path), resized)
            nothing_saved_idx += 1
            continue
        # ------------------------------------------------

        # есть фигура — проверяем, не дубль ли по положению
        if last_center is not None:
            dx = center[0] - last_center[0]
            dy = center[1] - last_center[1]
            dist = math.hypot(dx, dy)
            if dist < MIN_MOVE_PX:
                continue

        x, y, w, h = bbox
        H, W, _ = frame.shape
        x0 = max(0, x - CROP_MARGIN)
        y0 = max(0, y - CROP_MARGIN)
        x1 = min(W, x + w + CROP_MARGIN)
        y1 = min(H, y + h + CROP_MARGIN)

        crop = frame[y0:y1, x0:x1]

        # ---- выкидываем кадры, где много руки ----
        skin_crop = get_skin_mask(crop)
        skin_area = cv2.countNonZero(skin_crop)
        crop_area = crop.shape[0] * crop.shape[1]
        skin_ratio = skin_area / float(crop_area)

        if skin_ratio > SKIN_MAX_RATIO:
            # рука занимает слишком много места — кадр не используем
            continue
        # ------------------------------------------

        crop = cv2.resize(crop, OUT_SIZE)
        out_path = class_dir / f"{label}_{saved_idx:04d}.jpg"
        cv2.imwrite(str(out_path), crop)
        saved_idx += 1
        last_center = center

    cap.release()
    print(f"[{label}] сохранено кадров: {saved_idx}, nothing: {nothing_saved_idx}")


def main():
    out_root = Path(OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / NOTHING_CLASS).mkdir(parents=True, exist_ok=True)

    for label, video in VIDEOS.items():
        process_video(label, video, out_root)


if __name__ == "__main__":
    main()
