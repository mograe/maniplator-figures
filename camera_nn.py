import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

# ---------- настройки ----------
MIN_AREA = 1500
CROP_MARGIN = 10
MIN_FOCUS = 20
# -------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- загрузка модели ---

ckpt = torch.load("shape_classifier.pt", map_location=device)
class_names = ckpt["class_names"]

model = models.resnet18(weights=None)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, len(class_names))
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# --- функции для поиска фигуры (как в make_dataset.py) ---

def get_red_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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


def detect_red_object(frame):
    mask = get_red_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < MIN_AREA:
        return None, mask

    x, y, w, h = cv2.boundingRect(c)
    return (x, y, w, h), mask


# --- инференс из кадра ---

def predict_from_frame(frame):
    bbox, mask = detect_red_object(frame)
    if bbox is None:
        return "no object", frame, mask

    x, y, w, h = bbox
    H, W, _ = frame.shape
    x0 = max(0, x - CROP_MARGIN)
    y0 = max(0, y - CROP_MARGIN)
    x1 = min(W, x + w + CROP_MARGIN)
    y1 = min(H, y + h + CROP_MARGIN)

    crop = frame[y0:y1, x0:x1]
    # BGR -> RGB
    img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    tensor = preprocess(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        pred_class = class_names[pred_idx]
        confidence = probs[pred_idx].item()

    # рисуем bbox
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
    label = f"{pred_class} ({confidence:.2f})"
    cv2.putText(frame, label, (x0, max(30, y0-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                cv2.LINE_AA)

    return pred_class, frame, mask


def main():
    cap = cv2.VideoCapture("pyramid.mp4")  # или путь к видео

    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pred_class, vis_frame, mask = predict_from_frame(frame)

        cv2.imshow("frame", vis_frame)
        if mask is not None:
            cv2.imshow("red_mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
