"""SafeCity AI: Object Detection Module.

Wraps YOLOv8 inference with COCO class filtering, device auto-detection
(Apple Silicon MPS / CUDA / CPU), and normalized coordinate handling.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("safecity.detector")


class TrafficDetector:
    """YOLOv8 wrapper configured specifically for urban traffic surveillance."""

    # Default target classes from COCO:
    # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
    DEFAULT_TARGET_CLASSES = {
        0: "pedestrian",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        target_classes: Optional[Dict[int, str]] = None,
        device: str = "auto",
        img_size: int = 640
    ) -> None:
        self.model_name = model_name
        self.conf_thresh = confidence_threshold
        self.iou_thresh = iou_threshold
        self.target_classes = target_classes or self.DEFAULT_TARGET_CLASSES
        self.img_size = img_size
        self.device = self._resolve_device(device)

        logger.info(f"Initializing YOLOv8 model '{model_name}' on device: {self.device}")
        from ultralytics import YOLO  # Lazy import
        self.model = YOLO(model_name)

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolves optimal compute device, prioritizing Apple Silicon MPS on macOS."""
        if device != "auto":
            return device

        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Performs detection on a single BGR frame.

        Returns a list of dicts:
            {
                "bbox": [x1, y1, x2, y3], # Absolute pixel coordinates
                "conf": float,
                "class_id": int,
                "class_name": str,
                "bottom_center": [cx, y2] # Approximate road contact point
            }
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.img_size,
            device=self.device,
            classes=list(self.target_classes.keys()),
            verbose=False
        )

        detections: List[Dict[str, Any]] = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())

            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            bottom_center = [(x1 + x2) / 2.0, y2]

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "class_id": cls_id,
                "class_name": self.target_classes.get(cls_id, f"class_{cls_id}"),
                "bottom_center": bottom_center
            })

        return detections
