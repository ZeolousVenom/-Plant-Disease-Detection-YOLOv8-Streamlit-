# detect.py
import cv2
import torch
from ultralytics import YOLO

class RiceDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.35, device: str = None):
        self.conf_threshold = conf_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.last_result_plot = None  # will store the last annotated image

        try:
            self.model.overrides["conf"] = conf_threshold
        except Exception:
            pass

    def detect(self, image_bgr):
        # Convert to RGB for YOLO
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Run detection
        results = self.model.predict(
            source=image_rgb,
            device=self.device,
            conf=self.conf_threshold,
            verbose=False
        )

        dets = []
        r0 = results[0]
        names = r0.names
        boxes = r0.boxes

        for b in boxes:
            conf = float(b.conf.item())
            if conf < self.conf_threshold:
                continue

            xyxy = b.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            cls_id = int(b.cls.item())
            name = names.get(cls_id, str(cls_id))

            dets.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "name": name
            })

        # Save annotated result (numpy array, RGB)
        self.last_result_plot = r0.plot()  # already in BGR
        self.last_result_plot = cv2.cvtColor(self.last_result_plot, cv2.COLOR_BGR2RGB)

        return dets
