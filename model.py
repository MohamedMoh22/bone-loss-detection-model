import cv2
import io
from ultralytics import YOLO
from PIL import Image
import numpy as np

CONFIDENCE_THRESHOLD = 0.45
HEALTHY_CLASS_ID = 0
YOLO_OPTIMAL_SIZE = 640

COLOR_MAP = {
    1: (0, 255, 255),
    2: (255, 0, 0),
    3: (0, 0, 255)
}

# ✅ Load model once (IMPORTANT)
model_path = r"v11m.pt"
model = YOLO(model_path)


def process_image_bytes(image_bytes: bytes):
    # Convert bytes → OpenCV image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_image = np.array(image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

    orig_h, orig_w = original_image.shape[:2]

    scale_x = orig_w / YOLO_OPTIMAL_SIZE
    scale_y = orig_h / YOLO_OPTIMAL_SIZE

    base_scale = min(orig_h, orig_w) / 800.0
    BOX_THICKNESS = max(1, int(base_scale * 1.5))
    FONT_SCALE = base_scale * 0.60
    FONT_THICKNESS = max(1, int(base_scale * 1.5))

    # Resize for YOLO
    img_640 = cv2.resize(
        original_image, (YOLO_OPTIMAL_SIZE, YOLO_OPTIMAL_SIZE))

    results = model(img_640)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < CONFIDENCE_THRESHOLD or cls_id == HEALTHY_CLASS_ID:
                continue

            x1_640, y1_640, x2_640, y2_640 = map(int, box.xyxy[0])

            x1 = int(x1_640 * scale_x)
            y1 = int(y1_640 * scale_y)
            x2 = int(x2_640 * scale_x)
            y2 = int(y2_640 * scale_y)

            color = COLOR_MAP.get(cls_id, (255, 255, 255))

            cv2.rectangle(original_image, (x1, y1), (x2, y2),
                          color, thickness=BOX_THICKNESS)

            conf_text = f"{conf:.2f}"
            offset = int(12 * base_scale)
            text_y = y1 - 3 if y1 - 3 > offset else y1 + offset

            cv2.putText(
                original_image,
                conf_text,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                color,
                FONT_THICKNESS,
                cv2.LINE_AA
            )

    # Convert output → bytes
    _, buffer = cv2.imencode(".png", original_image)
    return io.BytesIO(buffer.tobytes())
