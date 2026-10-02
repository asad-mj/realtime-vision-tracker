import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    """Wrapper around YOLOv8/YOLO11 models for batch and stream inference."""

    def __init__(self, model_name: str = "yolov8n.pt", conf_thresh: float = 0.35):
        self.model = YOLO(model_name)
        self.conf_thresh = conf_thresh

    def detect(self, frame: np.ndarray):
        """
        Runs object detection on a single frame.
        Returns:
            boxes: np.ndarray (N, 4) in [x1, y1, x2, y2]
            confidences: np.ndarray (N,)
            class_ids: np.ndarray (N,)
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            verbose=False
        )[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        return boxes, confs, classes
