# utils.py
import io
import cv2
import numpy as np
from PIL import Image

def cv2_from_uploaded(uploaded_file):
    """
    Convert a Streamlit uploaded file into an OpenCV BGR image.
    """
    bytes_data = uploaded_file.read()
    image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def crop_with_padding(image_bgr, x1, y1, x2, y2, pad=6):
    """
    Crop a bounding box region with optional padding.
    """
    h, w = image_bgr.shape[:2]
    x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
    x2p, y2p = min(w, x2 + pad), min(h, y2 + pad)
    return image_bgr[y1p:y2p, x1p:x2p].copy()

def draw_detections(image_bgr, boxes, labels, scores):
    """
    Draw bounding boxes with labels and scores on an image.
    """
    for (x1, y1, x2, y2), lab, sc in zip(boxes, labels, scores):
        # Draw box
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label text
        text = f"{lab} | {sc*100:.1f}%"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        y_text = max(0, y1 - 4)

        # Background rectangle
        cv2.rectangle(
            image_bgr,
            (x1, y_text - th - baseline),
            (x1 + tw, y_text),
            (0, 255, 0),
            -1,
        )

        # Put text
        cv2.putText(
            image_bgr,
            text,
            (x1, y_text - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )
