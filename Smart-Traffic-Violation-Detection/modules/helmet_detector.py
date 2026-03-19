import cv2
import numpy as np
import os

HELMET_CLASSES = ['Helmet', 'No Helmet']


class HelmetDetector:
    def __init__(self, model_path='models/helmet_model.pt'):
        self.model = None
        if os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                print(f"[HelmetDetector] Model loaded: {model_path}")
            except Exception as e:
                print(f"[HelmetDetector] Could not load model: {e}")
        else:
            # Fallback to base YOLOv8 for person detection
            try:
                from ultralytics import YOLO
                self.model = YOLO('yolov8n.pt')
                print("[HelmetDetector] Using YOLOv8n for person detection (demo mode).")
            except Exception:
                print("[HelmetDetector] No model available — demo mode.")
                self.model = None

        self.frame_count = 0

    def detect(self, frame):
        """Detect helmets/no-helmets. Returns annotated frame + violation list."""
        self.frame_count += 1
        violations = []

        if self.model is None:
            return frame, violations

        try:
            results = self.model(frame, imgsz=320, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls   = int(box.cls[0])
                    conf  = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # For custom helmet model: cls 0 = Helmet, cls 1 = No Helmet
                    # For base YOLOv8: cls 0 = person (demo)
                    label = HELMET_CLASSES[cls] if cls < len(HELMET_CLASSES) else "Person"
                    color = (0, 255, 0) if label == "Helmet" else (0, 0, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}",
                                (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, color, 2)

                    if label == "No Helmet" and conf > 0.5:
                        violations.append({
                            "type": "No Helmet",
                            "confidence": round(conf, 2),
                            "bbox": [x1, y1, x2, y2]
                        })
                        # Red warning box
                        cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2),
                                      (0, 0, 255), 3)
        except Exception as e:
            print(f"[HelmetDetector] Detection error: {e}")

        return frame, violations
