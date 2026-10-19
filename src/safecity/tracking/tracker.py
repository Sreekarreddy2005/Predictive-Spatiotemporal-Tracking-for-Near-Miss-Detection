"""SafeCity AI: Multi-Object Tracking Module.

Integrates ByteTrack tracking with persistent ID histories, road contact point
tracking, and smooth velocity estimation.
"""

from __future__ import annotations

from collections import deque
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("safecity.tracker")


class TrackedObject:
    """Represents an active or historical object track."""

    def __init__(self, track_id: int, class_id: int, class_name: str, max_history: int = 60) -> None:
        self.track_id = track_id
        self.class_id = class_id
        self.class_name = class_name
        self.history: deque = deque(maxlen=max_history)  # [(frame_idx, cx, cy_bottom, w, h)]
        self.last_seen_frame: int = 0
        self.velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) in pixels/sec or m/s

    def update(self, frame_idx: int, bbox: List[float], dt: float = 0.08) -> None:
        """Updates track with a new observed bounding box."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy_bottom = y2  # Ground contact point
        w = x2 - x1
        h = y2 - y1

        if len(self.history) > 0 and dt > 0:
            prev_frame, prev_cx, prev_cy, _, _ = self.history[-1]
            vx = (cx - prev_cx) / dt
            vy = (cy_bottom - prev_cy) / dt
            # Exponential moving average for velocity smoothing
            self.velocity = (
                0.7 * self.velocity[0] + 0.3 * vx,
                0.7 * self.velocity[1] + 0.3 * vy
            )

        self.history.append((frame_idx, cx, cy_bottom, w, h))
        self.last_seen_frame = frame_idx

    @property
    def current_position(self) -> Optional[Tuple[float, float]]:
        if not self.history:
            return None
        _, cx, cy, _, _ = self.history[-1]
        return cx, cy

    @property
    def trail(self) -> List[Tuple[float, float]]:
        """Returns sequence of historical ground contact points."""
        return [(item[1], item[2]) for item in self.history]


class TrafficTracker:
    """Manages ByteTrack tracking state and track persistence across video frames."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        tracker_type: str = "bytetrack",
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        device: str = "auto",
        max_history: int = 60
    ) -> None:
        self.model_name = model_name
        self.tracker_type = tracker_type
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = self._resolve_device(device)
        self.max_history = max_history

        # Active tracks keyed by integer track ID
        self.tracks: Dict[int, TrackedObject] = {}

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
                return "0"
        except Exception:
            pass
        return "cpu"

    def track_frame(self, frame: np.ndarray, frame_idx: int, dt: float = 0.08) -> List[Dict[str, Any]]:
        """Runs detector + tracker on a single frame.

        Returns list of active tracked objects for the current frame:
            {
                "track_id": int,
                "bbox": [x1, y1, x2, y2],
                "conf": float,
                "class_id": int,
                "class_name": str,
                "trail": [(x, y), ...],
                "velocity": (vx, vy)
            }
        """
        # Run tracking using ByteTrack
        results = self.model.track(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )

        current_frame_tracks: List[Dict[str, Any]] = []
        if not results or len(results) == 0:
            return current_frame_tracks

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            return current_frame_tracks

        xyxys = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        track_ids = boxes.id.cpu().numpy().astype(int)
        names = self.model.names

        for xyxy, conf, cls_id, track_id in zip(xyxys, confs, cls_ids, track_ids):
            bbox = [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]
            class_name = names.get(cls_id, f"class_{cls_id}")

            if track_id not in self.tracks:
                self.tracks[track_id] = TrackedObject(
                    track_id=track_id,
                    class_id=cls_id,
                    class_name=class_name,
                    max_history=self.max_history
                )

            track_obj = self.tracks[track_id]
            track_obj.update(frame_idx=frame_idx, bbox=bbox, dt=dt)

            current_frame_tracks.append({
                "track_id": track_id,
                "bbox": bbox,
                "conf": float(conf),
                "class_id": int(cls_id),
                "class_name": class_name,
                "trail": track_obj.trail,
                "velocity": track_obj.velocity
            })

        return current_frame_tracks
